import unittest

import numpy as np

from scripts.run_longmemeval import session_records
from worldmodelsoc.experiment_io import paired_summary, prediction_metrics
from worldmodelsoc.memory.ctmc import MemoryRecord, TokenCodec, integer_shares, pack_context, rank_weights
from worldmodelsoc.synthetic_experiments import TransitionStore


class AllocationTests(unittest.TestCase):
    def test_normalization_order_and_uniform_limit(self):
        for tau in (0, 0.25, 1, 2):
            values = rank_weights(20, tau)
            self.assertAlmostEqual(float(values.sum()), 1)
            self.assertTrue(np.all(np.diff(values) <= 0))
        np.testing.assert_allclose(rank_weights(20, 0), np.full(20, 1 / 20))

    def test_core_mass_increases_without_changing_boundary(self):
        masses = [rank_weights(20, tau)[:3].sum() for tau in (0, 0.25, 0.5, 1, 2)]
        self.assertTrue(all(left < right for left, right in zip(masses, masses[1:])))

    def test_integer_budget_is_conserved(self):
        for budget in (0, 1, 17, 512):
            shares = integer_shares(rank_weights(13, 1), budget)
            self.assertEqual(int(shares.sum()), budget)
            self.assertTrue(np.all(shares >= 0))

    def test_serializations_preserve_ranking_and_stay_within_budget(self):
        codec = TokenCodec()
        records = [MemoryRecord(str(i), "evidence " * 100, metadata={"prev": "s", "action": str(i), "next": "n", "count": 1}) for i in range(10)]
        views = []
        for encoding in ("compact", "verbose"):
            context, allocation = pack_context(records, "ctmc", encoding, 128, 3, 1, codec)
            self.assertLessEqual(codec.count(context), 128)
            self.assertGreater(allocation["tail_budget"], 0)
            views.append(allocation)
        self.assertEqual(views[0]["ranked_memory_ids"], views[1]["ranked_memory_ids"])
        self.assertEqual(views[0]["allocation_weights"], views[1]["allocation_weights"])


class EvaluationTests(unittest.TestCase):
    def test_tail_error_conditions_on_query_state(self):
        records = [
            {"prev": "rare", "next": "common", "prediction": "common"},
            {"prev": "common", "next": "rare", "prediction": "wrong"},
            {"prev": "common", "next": "common", "prediction": "wrong"},
        ]
        metrics = prediction_metrics(records)
        self.assertEqual(metrics["rare_states"], ["rare"])
        self.assertEqual(metrics["tail_visits"], 1)
        self.assertEqual(metrics["tail_prediction_error"], 0)

    def test_empty_tail_is_not_reported_as_success(self):
        metrics = prediction_metrics([], rare_states=set())
        self.assertIsNone(metrics["tail_prediction_error"])

    def test_pairing_ignores_unmatched_runs_and_rejects_duplicates(self):
        rows = [
            {"pair_id": "a", "method": "graph_memory", "tokens": 100},
            {"pair_id": "a", "method": "ctmc", "tokens": 75},
            {"pair_id": "b", "method": "graph_memory", "tokens": 1000},
        ]
        summary = paired_summary(rows, "graph_memory", "ctmc", ["tokens"], resamples=20)
        self.assertEqual(summary["n_pairs"], 1)
        self.assertEqual(summary["n_unpaired"], 1)
        self.assertEqual(summary["metrics"]["tokens"]["reduction_percent"], 25)
        with self.assertRaises(ValueError):
            paired_summary(rows + [rows[0]], "graph_memory", "ctmc", ["tokens"])

    def test_ground_truth_labels_are_excluded_from_session_memory(self):
        example = {
            "question_id": "q", "answer": "HIDDEN_ANSWER", "answer_session_ids": ["LABEL_ONLY"],
            "haystack_session_ids": ["session"], "haystack_dates": ["2025-01-01"],
            "haystack_sessions": [[{"role": "user", "content": "I moved to Boston.", "has_answer": "HIDDEN_MARKER"}]],
        }
        record = session_records(example)[0]
        self.assertIn("I moved to Boston.", record.text)
        for marker in ("HIDDEN_ANSWER", "HIDDEN_MARKER", "LABEL_ONLY", "has_answer"):
            self.assertNotIn(marker, record.text)

    def test_store_capacity_and_historical_coverage_are_distinct(self):
        store = TransitionStore(1)
        store.write("s0", "a", "s1", 0, [])
        store.write("s1", "a", "s2", 1, [])
        self.assertEqual(len(store.records), 1)
        self.assertEqual(len(store.history), 2)
        self.assertEqual(len(store.visited_transitions), 2)


if __name__ == "__main__":
    unittest.main()
