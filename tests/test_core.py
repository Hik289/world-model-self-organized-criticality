import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from scripts.run_frozen_replay import replay_on_trajectory
from scripts.run_method_comparison import CostAccountant, llm_pick_action
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
from worldmodelsoc.memory.reservoir import (
    StateAwareReservoirMemory,
    TauReservoirMemory,
    summary_stats,
)


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


if __name__ == "__main__":
    unittest.main()
