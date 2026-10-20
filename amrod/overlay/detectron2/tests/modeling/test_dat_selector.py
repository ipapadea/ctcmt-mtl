import math
import unittest

from detectron2.modeling.meta_arch.dat_pfn import DATTensorSelector

TOTAL = 28425555
CAP = 2842555


class TestDATTensorSelector(unittest.TestCase):
    def test_cap_arithmetic_matches_locked_value(self):
        self.assertEqual(int(math.floor(0.10 * TOTAL)), CAP)
        self.assertEqual(max(1, int(0.001 * 92)), 1)
        self.assertEqual(int(0.001 * 92), 0)  # why the minimum-one porting rule exists

    def test_k_must_be_one(self):
        with self.assertRaises(ValueError):
            DATTensorSelector([1, 2], 10, k=2)

    def test_empty_group_selects_nothing(self):
        sel = DATTensorSelector([5, 5], 100)
        self.assertEqual(sel.consider(None)["outcome"], "empty_group")
        self.assertEqual(sel.selected, [])

    def test_zero_score_selects_nothing(self):
        sel = DATTensorSelector([5, 5], 100)
        self.assertEqual(sel.consider([0.0, 0.0])["outcome"], "zero_score")
        self.assertEqual(sel.selected, [])

    def test_adds_top_ranked_whole_tensor(self):
        sel = DATTensorSelector([5, 7, 3], 100)
        res = sel.consider([1.0, 9.0, 2.0])
        self.assertEqual((res["winner"], res["outcome"]), (1, "added"))
        self.assertEqual((sel.selected, sel.selected_elements), ([1], 7))

    def test_ties_resolve_to_registration_order(self):
        sel = DATTensorSelector([1, 1, 1], 100)
        self.assertEqual(sel.consider([3.0, 3.0, 3.0])["winner"], 0)

    def test_already_selected_winner_adds_nothing_and_does_not_search_further(self):
        sel = DATTensorSelector([5, 7, 3], 100)
        sel.consider([1.0, 9.0, 2.0])
        res = sel.consider([1.0, 9.0, 8.0])  # tensor 2 is second best and unselected
        self.assertEqual(res["outcome"], "already_selected")
        self.assertEqual(sel.selected, [1])

    def test_does_not_fit_adds_nothing_and_does_not_search_further(self):
        sel = DATTensorSelector([10, 4, 3], 12)
        sel.consider([0.0, 5.0, 0.0])  # 4 elements used
        res = sel.consider([9.0, 1.0, 8.0])  # winner needs 10 > 8 remaining; tensor 2 would fit
        self.assertEqual(res["outcome"], "does_not_fit")
        self.assertEqual(sel.selected, [1])
        self.assertEqual(sel.selected_elements, 4)

    def test_cap_is_strict_and_exact_fit_is_allowed(self):
        sel = DATTensorSelector([6, 6, 1], 12)
        self.assertEqual(sel.consider([2.0, 1.0, 0.0])["outcome"], "added")
        self.assertEqual(sel.consider([1.0, 2.0, 0.0])["outcome"], "added")
        self.assertEqual(sel.selected_elements, 12)
        self.assertEqual(sel.consider([0.0, 0.0, 5.0])["outcome"], "does_not_fit")
        self.assertLessEqual(sel.selected_elements, sel.cap_elements)

    def test_same_tensor_chosen_by_dsp_and_trp_is_added_once(self):
        sel = DATTensorSelector([5, 7], 100)
        self.assertEqual(sel.consider([1.0, 9.0])["outcome"], "added")  # DSP
        self.assertEqual(sel.consider([1.0, 9.0])["outcome"], "already_selected")  # TRP
        self.assertEqual(sel.selected, [1])
        self.assertEqual(sel.selected_elements, 7)

    def test_selected_set_only_grows(self):
        sel = DATTensorSelector([1] * 6, 100)
        seen = []
        for winner in (2, 5, 2, 0, 5):
            scores = [0.0] * 6
            scores[winner] = 1.0
            sel.consider(scores)
            seen.append(list(sel.selected))
        for a, b in zip(seen, seen[1:]):
            self.assertEqual(b[: len(a)], a)
        self.assertEqual(sel.selected_sorted, [0, 2, 5])


if __name__ == "__main__":
    unittest.main()
