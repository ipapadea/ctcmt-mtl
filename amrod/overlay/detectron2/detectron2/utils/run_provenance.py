# Provenance logging for test-time-adaptation baseline runs.
#
# Every new CTTA baseline run must record, in its own log:
#   git SHA, dirty/clean state, resolved MODEL.WEIGHTS, checkpoint MD5,
#   adaptation method/config, seed and target stream/order.
import hashlib
import logging
import os
import subprocess

__all__ = ["log_run_provenance", "file_md5"]


def file_md5(path, chunk_size=1 << 20):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo_dir, *args):
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir] + list(args),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _repo_dir():
    # .../detectron2/detectron2/utils/run_provenance.py -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def log_run_provenance(cfg, method, extra=None, logger=None):
    """
    Print a single provenance block. Returns the dict that was logged.
    """
    logger = logger or logging.getLogger(__name__)
    repo = _repo_dir()
    sha = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    weights = cfg.MODEL.WEIGHTS
    info = {
        "method": method,
        "git_sha": sha or "unknown",
        "git_state": ("unknown" if status is None else ("dirty" if status else "clean")),
        "model_weights": weights,
        "model_weights_md5": file_md5(weights) or "unavailable",
        "seed": cfg.SEED,
        "meta_architecture": cfg.MODEL.META_ARCHITECTURE,
        "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH,
        "output_dir": cfg.OUTPUT_DIR,
        "stream_len": len(cfg.DATASETS.TEST),
        "stream": list(cfg.DATASETS.TEST),
    }
    if extra:
        info.update(extra)
    lines = ["=" * 70, "RUN PROVENANCE"]
    for k, v in info.items():
        lines.append("  {:<22}: {}".format(k, v))
    lines.append("=" * 70)
    msg = "\n".join(lines)
    logger.info(msg)
    return info
