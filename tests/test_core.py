import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np

from scripts.run_frozen_replay import replay_on_trajectory
from scripts.run_method_comparison import CostAccountant, llm_pick_action, session_records
from worldmodelsoc.env.synthetic_graph_world import (
    GRAPH_TYPES,
    action_index_for_neighbor,
    build_graph,
    build_state_payloads,
    describe_action_options,
    neighbor_segment,
)
from worldmodelsoc.llm_config import get_llm_api_base_url, load_llm_api_key
from worldmodelsoc.memory import backends, backends_ctwm
from worldmodelsoc.memory.backends_ctwm import MemoryRecord, TokenCodec, TransitionStore, integer_shares, pack_context, rank_weights
from worldmodelsoc.memory.reservoir import (
    StateAwareReservoirMemory,
    TauReservoirMemory,
    summary_stats,
)
from worldmodelsoc.pipeline.modules import paired_summary, prediction_metrics


ROOT = Path(__file__).resolve().parents[1]


def _write_transitions(memory, count=8):
    for i in range(count):
        memory.write_transition(f"s{i}", "go", f"n{i}", i)


class GraphWorldTests(unittest.TestCase):
    def test_all_graph_families_are_reproducible_and_connected(self):
        for graph_type in GRAPH_TYPES:
            first = build_graph(graph_type, 30, seed=9)
            second = build_graph(graph_type, 30, seed=9)
            self.assertTrue(nx.is_strongly_connected(first), graph_type)
            self.assertEqual(set(first.edges), set(second.edges), graph_type)

    def test_toy_graph_fixture_is_tracked_and_well_formed(self):
        with (ROOT / "data" / "toy_graph.json").open(encoding="utf-8") as handle:
            toy_graph = json.load(handle)
        states = {state["state_id"] for state in toy_graph["states"]}
        self.assertEqual(states, set(toy_graph["canonical_state_ids"]))
        for previous, _action, next_state in toy_graph["edges_ground_truth"]:
            self.assertIn(previous, states)
            self.assertIn(next_state, states)

    def test_invalid_graph_size_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least 6"):
            build_graph("scale_free", 1)

    def test_action_descriptions_match_neighbor_segments(self):
        graph = build_graph("scale_free", 30, seed=5)
        payloads = build_state_payloads(graph, seed=5)
        node = next(iter(graph))
        neighbors = list(graph.successors(node))
        options = describe_action_options(graph, payloads, node)
        self.assertEqual(len(options), len(payloads[node].actions))
        for action_idx, option in enumerate(options):
            segment = neighbor_segment(
                neighbors,
                len(payloads[node].actions),
                action_idx,
            )
            self.assertTrue(any(str(target) in option for target in segment))
            for target in segment:
                self.assertEqual(
                    action_index_for_neighbor(
                        neighbors,
                        len(payloads[node].actions),
                        target,
                    ),
                    action_idx,
                )


class MemoryBackendTests(unittest.TestCase):
    def test_flat_retrieval_serializes_the_entries_it_selected(self):
        for module in (backends, backends_ctwm):
            memory = module.B3_FlatRetrieval(top_k=3, seed=17)
            _write_transitions(memory)
            events = memory.retrieve_hints("unused", step=10)
            selected = [event["memory_id"] for event in events]
            self.assertEqual(selected, memory.retrieve_no_side_effects("unused"))
            context = memory.context_string("unused")
            for transition_id in selected:
                self.assertIn(memory.entries[transition_id]["content"], context)

    def test_graph_memory_searches_newest_episode_first(self):
        for module in (backends, backends_ctwm):
            memory = module.B7_GraphMemory(episode_length=2, top_k=2)
            memory.write_transition("s", "old", "old_next", 0)
            memory.write_transition("x", "close", "y", 1)
            memory.write_transition("s", "recent", "recent_next", 2)
            picks = memory._episodic_top_k("s")
            self.assertEqual(
                picks,
                [
                    ("s", "recent", "recent_next"),
                    ("s", "old", "old_next"),
                ],
            )

    def test_coverage_uses_retained_entries(self):
        for module in (backends, backends_ctwm):
            memory = module.B4_FrequencyCache(capacity=1, top_k=1)
            memory.write_transition("s0", "go", "n0", 0)
            memory.write_transition("s1", "go", "n1", 1)
            walker_transitions = {"s0::go::n0", "s1::go::n1"}
            walker_states = {"s0", "n0", "s1", "n1"}
            self.assertEqual(memory.coverage_trans(walker_transitions), 0.5)
            self.assertEqual(memory.coverage_state(walker_states), 0.5)

    def test_ctwm_sampling_is_seeded(self):
        for module in (backends, backends_ctwm):
            first = module.B8_CTWM(core_slots=1, tail_slots=3, seed=21)
            second = module.B8_CTWM(core_slots=1, tail_slots=3, seed=21)
            _write_transitions(first, count=20)
            _write_transitions(second, count=20)
            self.assertEqual(
                first.retrieve_hints("missing", 20),
                second.retrieve_hints("missing", 20),
            )

    def test_reservoir_sampling_has_no_duplicates(self):
        first = TauReservoirMemory(
            capacity=20,
            rng=random.Random(4),
            tau=1.0,
            K_pool=10,
            M_pass=5,
        )
        second = TauReservoirMemory(
            capacity=20,
            rng=random.Random(4),
            tau=1.0,
            K_pool=10,
            M_pass=5,
        )
        for memory in (first, second):
            for i in range(10):
                memory.write(
                    f"m{i}",
                    f"transition {i}",
                    prev=f"s{i}",
                    action="go",
                    nxt=f"n{i}",
                    step=i,
                )
        first_ids = [event["memory_id"] for event in first.retrieve("none")]
        second_ids = [event["memory_id"] for event in second.retrieve("none")]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))

    def test_invalid_memory_configuration_fails_early(self):
        with self.assertRaises(ValueError):
            StateAwareReservoirMemory(capacity=0)
        with self.assertRaises(ValueError):
            TauReservoirMemory(tau=-1)
        with self.assertRaises(ValueError):
            backends.B8_CTWM(weights=(1, 2))
        with self.assertRaises(ValueError):
            StateAwareReservoirMemory().retrieve("s", k=-1)

    def test_constant_frequency_summary_is_valid(self):
        self.assertEqual(summary_stats([1, 1, 1])["skew"], 0.0)


class ExperimentTests(unittest.TestCase):
    def test_policy_prompt_contains_semantic_state_and_action_context(self):
        class FakeCompletions:
            def __init__(self):
                self.request = None

            def create(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=2,
                    ),
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"action_idx": 0}'
                            )
                        )
                    ],
                )

        completions = FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        result = llm_pick_action(
            client,
            "s",
            ["act_0"],
            [1],
            [],
            "(empty)",
            CostAccountant(budget_usd=1.0),
            random.Random(3),
            state_context="constraints=['temperature<50']",
            action_options=["act_0 -> target battery>=0.2"],
            retries=1,
        )
        self.assertEqual(result, (0, True))
        user_prompt = completions.request["messages"][1]["content"]
        self.assertIn("temperature<50", user_prompt)
        self.assertIn("target battery>=0.2", user_prompt)

    def test_llm_configuration_supports_standard_and_azure_endpoints(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_API_BASE_URL": "",
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            },
            clear=False,
        ):
            self.assertEqual(
                get_llm_api_base_url(),
                "https://example.openai.azure.com/openai/v1",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            empty_key = Path(temp_dir) / "empty.key"
            empty_key.write_text("", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {
                    "LLM_API_KEY": "",
                    "OPENAI_API_KEY": "",
                    "AZURE_OPENAI_API_KEY": "",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "empty"):
                    load_llm_api_key(str(empty_key))

    def test_llm_fallback_uses_the_supplied_rng(self):
        def fallback(seed):
            accountant = CostAccountant(budget_usd=1.0)
            return llm_pick_action(
                object(),
                "s",
                ["a0", "a1", "a2"],
                [1, 2, 3],
                [],
                "(empty)",
                accountant,
                random.Random(seed),
                retries=0,
            )

        self.assertEqual(fallback(33), fallback(33))

    def test_frozen_replay_is_reproducible(self):
        actions = [
            {"prev": f"s{i % 3}", "action": "go", "next": f"s{(i + 1) % 3}"}
            for i in range(12)
        ]
        first = replay_on_trajectory(actions, "first", seed=8)
        second = replay_on_trajectory(actions, "second", seed=8)
        comparable_keys = {
            "tail_correct_count",
            "tail_total_count",
            "tail_pred_accuracy",
            "tail_pred_error",
            "n_predictions",
        }
        self.assertEqual(
            {key: first[key] for key in comparable_keys},
            {key: second[key] for key in comparable_keys},
        )


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
