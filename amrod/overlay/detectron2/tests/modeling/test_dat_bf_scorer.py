import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import SemSegEvaluator

_TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", "tools")
_spec = importlib.util.spec_from_file_location("dat_bf_score_predictions", os.path.join(_TOOLS, "dat_bf_score_predictions.py"))
scorer = importlib.util.module_from_spec(_spec)
sys.modules["dat_bf_score_predictions"] = scorer
_spec.loader.exec_module(scorer)

C, H, W = 19, 24, 40


def _make_dataset(root, name, n, seed):
    rng = np.random.RandomState(seed)
    records, gts, preds = [], [], []
    for i in range(n):
        gt = rng.randint(0, C, size=(H, W)).astype(np.uint8)
        gt[rng.rand(H, W) < 0.1] = 255
        pred = np.where(rng.rand(H, W) < 0.6, np.where(gt == 255, 0, gt), rng.randint(0, C, size=(H, W))).astype(np.uint8)
        gpath = os.path.join(root, f"{name}_{i}_gt.png")
        Image.fromarray(gt).save(gpath)
        records.append({"file_name": f"/fake/{name}/{i}.png", "sem_seg_file_name": gpath, "height": H, "width": W})
        gts.append(gt)
        preds.append(pred)
    DatasetCatalog.register(name, lambda records=records: records)
    MetadataCatalog.get(name).set(
        evaluator_type="sem_seg", ignore_label=255, stuff_classes=[f"c{k}" for k in range(C)], sem_seg_root=root,
    )
    return records, gts, preds


def _write_run(run_dir, records, preds, tag_frames=None, log_names=None):
    """Write frame log + prediction PNGs for the first ``tag_frames`` frames of the concatenated stream."""
    n = len(records)
    tag_frames = n if tag_frames is None else tag_frames
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "dat_bf_frames.jsonl"), "w") as f:
        for k in range(tag_frames):
            f.write(json.dumps({"frame": k + 1, "file_name": (log_names or [r["file_name"] for r in records])[k]}) + "\n")
    for tag in ("primary", "plain_view"):
        d = os.path.join(run_dir, "dat_bf_predictions", tag)
        os.makedirs(d, exist_ok=True)
        for k in range(tag_frames):
            p = preds[k] if tag == "primary" else (preds[k] + 1) % C
            Image.fromarray(p.astype(np.uint8)).save(os.path.join(d, f"{k + 1:06d}.png"))


def _numpy_miou(gts, preds):
    conf = np.zeros((C, C), dtype=np.int64)
    for gt, pr in zip(gts, preds):
        m = gt != 255
        np.add.at(conf, (gt[m], pr[m]), 1)
    iou = np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf))
    return 100 * np.nanmean(iou)


class TestScorer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_agrees_with_online_evaluator_and_independent_numpy(self):
        records, gts, preds = _make_dataset(self.root, "fake_a", 6, 0)
        run = os.path.join(self.root, "run")
        _write_run(run, records, preds)
        res = scorer.score_run(run, [("fake_a", None)])
        # online path: model-style one-hot (C,H,W) float outputs
        ev = SemSegEvaluator("fake_a", distributed=False, output_dir=None)
        ev.reset()
        for rec, pr in zip(records, preds):
            onehot = torch.nn.functional.one_hot(torch.from_numpy(pr.astype(np.int64)), C).permute(2, 0, 1).float()
            ev.process([{"file_name": rec["file_name"]}], [{"sem_seg": onehot}])
        online = {"fake_a": ev.evaluate()}
        worst, _ = scorer.compare_online(res["primary"], online)
        self.assertLessEqual(worst, 1e-12)
        self.assertAlmostEqual(res["primary"]["fake_a"]["mIoU"], _numpy_miou(gts, preds), places=6)
        self.assertTrue(res["primary"]["fake_a"]["complete"])
        # secondary stream is scored separately and differs
        self.assertNotAlmostEqual(res["plain_view"]["fake_a"]["mIoU"], res["primary"]["fake_a"]["mIoU"], places=3)

    def test_truncated_online_name_is_matched(self):
        records, _, preds = _make_dataset(self.root, "fake_b", 4, 1)
        run = os.path.join(self.root, "run")
        _write_run(run, records, preds)
        res = scorer.score_run(run, [("fake_b", 3)])
        self.assertEqual(res["primary"]["fake_b"]["frames_expected"], 3)
        online = {"fake_b_first3": {"sem_seg": {k: res["primary"]["fake_b"][k] for k in scorer.METRIC_KEYS}}}
        worst, detail = scorer.compare_online(res["primary"], online)
        self.assertEqual(worst, 0.0)
        self.assertTrue(detail)

    def test_partial_scoring_after_abort(self):
        records, gts, preds = _make_dataset(self.root, "fake_c", 5, 2)
        run = os.path.join(self.root, "run")
        _write_run(run, records, preds, tag_frames=3)
        res = scorer.score_run(run, [("fake_c", None)])
        r = res["primary"]["fake_c"]
        self.assertEqual((r["frames_scored"], r["frames_expected"], r["complete"]), (3, 5, False))
        self.assertAlmostEqual(r["mIoU"], _numpy_miou(gts[:3], preds[:3]), places=6)

    def test_two_datasets_sequential_frame_mapping(self):
        ra, ga, pa = _make_dataset(self.root, "fake_d1", 3, 3)
        rb, gb, pb = _make_dataset(self.root, "fake_d2", 4, 4)
        run = os.path.join(self.root, "run")
        _write_run(run, ra + rb, pa + pb)
        res = scorer.score_run(run, [("fake_d1", None), ("fake_d2", None)])
        self.assertAlmostEqual(res["primary"]["fake_d1"]["mIoU"], _numpy_miou(ga, pa), places=6)
        self.assertAlmostEqual(res["primary"]["fake_d2"]["mIoU"], _numpy_miou(gb, pb), places=6)

    def test_misalignment_is_detected(self):
        records, _, preds = _make_dataset(self.root, "fake_e", 3, 5)
        run = os.path.join(self.root, "run")
        names = [r["file_name"] for r in records]
        names[1] = "/fake/other.png"
        _write_run(run, records, preds, log_names=names)
        with self.assertRaises(RuntimeError):
            scorer.score_run(run, [("fake_e", None)])

    def test_online_log_parser_matches_run_log_format(self):
        sample = (
            "[10/13 10:10:25 d2.engine.defaults]: Evaluation results for acdc_fog_semseg in csv format:\n"
            "[10/13 10:10:25 d2.evaluation.testing]: copypaste: Task: sem_seg\n"
            "[10/13 10:10:25 d2.evaluation.testing]: copypaste: mIoU,fwIoU,mACC,pACC\n"
            "[10/13 10:10:25 d2.evaluation.testing]: copypaste: 47.0513,70.1664,57.8035,81.2495\n"
            "[10/13 10:19:08 d2.engine.defaults]: Evaluation results for acdc_night_semseg in csv format:\n"
            "[10/13 10:19:08 d2.evaluation.testing]: copypaste: Task: sem_seg\n"
            "[10/13 10:19:08 d2.evaluation.testing]: copypaste: mIoU,fwIoU,mACC,pACC\n"
            "[10/13 10:19:08 d2.evaluation.testing]: copypaste: 21.1823,44.8627,29.3865,60.8469\n"
        )
        path = os.path.join(self.root, "log.txt")
        open(path, "w").write(sample)
        online = scorer.parse_online_log(path)
        self.assertEqual(online["acdc_fog_semseg"]["mIoU"], 47.0513)
        self.assertEqual(online["acdc_night_semseg"]["pACC"], 60.8469)


    def test_repeated_domains_do_not_overwrite_each_other(self):
        # long-term stream: A B A B, each domain scored on its own frames, in stream order
        ra, ga, pa = _make_dataset(self.root, "fake_lt_a", 3, 10)
        rb, gb, pb = _make_dataset(self.root, "fake_lt_b", 2, 11)
        # second visit of the same dataset sees the same records but different predictions
        pa2 = [(p + 3) % C for p in pa]
        pb2 = [(p + 5) % C for p in pb]
        run = os.path.join(self.root, "run")
        _write_run(run, ra + rb + ra + rb, pa + pb + pa2 + pb2)
        specs = [("fake_lt_a", None), ("fake_lt_b", None), ("fake_lt_a", None), ("fake_lt_b", None)]
        self.assertEqual(scorer.spec_keys(specs), ["000_fake_lt_a", "001_fake_lt_b", "002_fake_lt_a", "003_fake_lt_b"])
        res = scorer.score_run(run, specs)["primary"]
        self.assertEqual(list(res), ["000_fake_lt_a", "001_fake_lt_b", "002_fake_lt_a", "003_fake_lt_b"])
        self.assertAlmostEqual(res["000_fake_lt_a"]["mIoU"], _numpy_miou(ga, pa), places=6)
        self.assertAlmostEqual(res["002_fake_lt_a"]["mIoU"], _numpy_miou(ga, pa2), places=6)
        self.assertAlmostEqual(res["003_fake_lt_b"]["mIoU"], _numpy_miou(gb, pb2), places=6)
        self.assertNotAlmostEqual(res["000_fake_lt_a"]["mIoU"], res["002_fake_lt_a"]["mIoU"], places=3)
        self.assertTrue(all(r["name"].startswith("fake_lt_") for r in res.values()))

    def test_online_log_parser_keeps_repeated_datasets_in_order(self):
        block = (
            "[x d2.engine.defaults]: Evaluation results for {n} in csv format:\n"
            "[x d2.evaluation.testing]: copypaste: Task: sem_seg\n"
            "[x d2.evaluation.testing]: copypaste: mIoU,fwIoU,mACC,pACC\n"
            "[x d2.evaluation.testing]: copypaste: {v}\n"
        )
        text = block.format(n="fog_semseg", v="1.0000,2.0000,3.0000,4.0000") + block.format(
            n="snow_semseg", v="5.0000,6.0000,7.0000,8.0000") + block.format(n="fog_semseg", v="9.0000,9.5000,9.7000,9.9000")
        path = os.path.join(self.root, "lt.log")
        open(path, "w").write(text)
        self.assertEqual([n for n, _ in scorer.parse_online_log_list(path)], ["fog_semseg", "snow_semseg", "fog_semseg"])
        keyed = scorer.parse_online_log(path)
        # same rule as spec_keys: plain name when unique in the stream, positional when repeated
        self.assertEqual(list(keyed), ["000_fog_semseg", "snow_semseg", "002_fog_semseg"])
        self.assertEqual(keyed["002_fog_semseg"]["mIoU"], 9.0)

    def test_parity_requires_every_offline_entry_to_be_matched(self):
        offline = {"000_a": {"name": "a", "frames_expected": 3, "mIoU": 1.0, "fwIoU": 2.0, "mACC": 3.0, "pACC": 4.0},
                   "001_b": {"name": "b", "frames_expected": 3, "mIoU": 1.0, "fwIoU": 2.0, "mACC": 3.0, "pACC": 4.0}}
        online = {"000_a": {"mIoU": 1.0, "fwIoU": 2.0, "mACC": 3.0, "pACC": 4.0}}
        worst, detail = scorer.compare_online(offline, online)
        self.assertEqual(worst, 0.0)
        self.assertEqual(len(detail), 1)  # the caller treats len(detail) != len(offline) as DISAGREE


if __name__ == "__main__":
    unittest.main()
