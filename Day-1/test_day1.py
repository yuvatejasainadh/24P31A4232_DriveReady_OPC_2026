"""Automated unit and integration checks for Day 1 (Demo Mode)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import task


class TestDay1UniversalBusinessAgent(unittest.TestCase):
    """Verify validation, grounding, memory, token reporting, and profile switching."""

    def setUp(self) -> None:
        self.profiles, self.default_id = task.load_business_profiles(task.DEFAULT_PROFILES_PATH)
        self.backend = task.GeminiAgentBackend(force_demo=True)

    def test_profiles_loaded_and_default_valid(self) -> None:
        self.assertIn("dripzoid", self.profiles)
        self.assertIn("edusmart", self.profiles)
        self.assertEqual(self.default_id, "dripzoid")

    def test_prompt_template_validation_and_rendering(self) -> None:
        rendered = task.build_business_prompt_template(
            business_name="Dripzoid",
            business_industry="E-commerce",
            business_description="Online clothing platform",
            business_tone="Friendly",
            user_query="What is your return policy?",
        )
        self.assertIn("Dripzoid", rendered.system_instruction)
        self.assertIn("Friendly", rendered.system_instruction)
        self.assertIn("What is your return policy?", rendered.user_prompt)

        with self.assertRaises(task.ValidationError):
            task.build_business_prompt_template(
                business_name="   ",
                business_industry="E-commerce",
                business_description="Desc",
                business_tone="Friendly",
                user_query="Hello",
            )

    def test_grounding_supported_and_unsupported_queries(self) -> None:
        agent = task.UniversalBusinessAgent(
            profile=self.profiles["dripzoid"],
            backend=self.backend,
        )
        supported = task.query_stateless(agent, "What is your return policy?")
        self.assertTrue(supported.is_mock)
        self.assertIn("[MOCK DEMO", supported.text)
        self.assertIn("7 days", supported.text)

        unsupported = task.query_stateless(
            agent,
            "Do you offer 30-minute drone delivery and a lifetime warranty?",
        )
        self.assertIn("do not have enough information", unsupported.text.lower())

    def test_stateless_vs_conversation_history(self) -> None:
        agent = task.UniversalBusinessAgent(
            profile=self.profiles["dripzoid"],
            backend=self.backend,
        )
        turn_1 = "Tell me about the Dripzoid Oversized Graphic Hoodie."
        follow_up = "Does it qualify for free standard shipping?"

        # Stateless cannot resolve "it"
        stateless_reply = task.query_stateless(agent, follow_up)
        self.assertIn("do not have previous conversation context", stateless_reply.text.lower())

        # Conversation-aware carries Turn 1 context into Turn 2
        history: list[task.ConversationTurn] = []
        _, history = task.query_with_history(agent, turn_1, history)
        aware_reply, history = task.query_with_history(agent, follow_up, history)
        self.assertEqual(len(history), 4)
        self.assertIn("1,799", aware_reply.text)
        self.assertIn("1,499", aware_reply.text)

    def test_profile_switching_across_industries(self) -> None:
        agent = task.UniversalBusinessAgent(
            profile=self.profiles["dripzoid"],
            backend=self.backend,
        )
        dripzoid_res = task.query_stateless(agent, "What products do you offer?")
        self.assertIn("Dripzoid", dripzoid_res.text)
        self.assertIn("Hoodie", dripzoid_res.text)

        agent.switch_profile(self.profiles["edusmart"])
        edusmart_res = task.query_stateless(agent, "What courses do you offer?")
        self.assertIn("EduSmart", edusmart_res.text)
        self.assertIn("Bootcamp", edusmart_res.text)

    def test_token_metrics_never_fabricate_api_counts_in_demo_mode(self) -> None:
        agent = task.UniversalBusinessAgent(
            profile=self.profiles["dripzoid"],
            backend=self.backend,
        )
        result = task.query_stateless(agent, "What are your support hours?")
        self.assertGreater(result.token_metrics.estimated_prompt_tokens, 0)
        self.assertGreater(result.token_metrics.estimated_response_tokens, 0)
        self.assertIsNone(result.token_metrics.api_prompt_tokens)
        self.assertIsNone(result.token_metrics.api_candidates_tokens)
        self.assertIsNone(result.token_metrics.api_total_tokens)

        report = task.format_token_report(result, label="Test")
        self.assertIn("ESTIMATED", report)
        self.assertIn("Unavailable in mock/demo mode", report)

    def test_error_handling_for_invalid_inputs_and_configs(self) -> None:
        with self.assertRaises(task.ValidationError):
            task.validate_temperature(3.5)

        with self.assertRaises(task.ValidationError):
            task.validate_temperature(True)

        with self.assertRaises(task.ConfigurationError):
            task.load_business_profiles(Path("non_existent_profiles.json"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_json = Path(tmp_dir) / "bad.json"
            bad_json.write_text("{invalid json", encoding="utf-8")
            with self.assertRaises(task.ConfigurationError):
                task.load_business_profiles(bad_json)

            incomplete_json = Path(tmp_dir) / "incomplete.json"
            incomplete_json.write_text(
                json.dumps({"profiles": {"x": {"business_name": "OnlyName"}}}),
                encoding="utf-8",
            )
            with self.assertRaises(task.ConfigurationError):
                task.load_business_profiles(incomplete_json)

        exit_code = task.main(["--demo", "--profile", "unknown_biz"])
        self.assertEqual(exit_code, 1)

        exit_code_empty = task.main(["--demo", "--query", "   "])
        self.assertEqual(exit_code_empty, 1)


if __name__ == "__main__":
    unittest.main()
