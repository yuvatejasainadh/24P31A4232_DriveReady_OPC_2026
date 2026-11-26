"""Automated Day 2 test suite covering all 17 required verification cases."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

DAY2_DIR = Path(__file__).resolve().parent
if str(DAY2_DIR) not in sys.path:
    sys.path.insert(0, str(DAY2_DIR))

from agent import UniversalToolAgent, evaluate_agent_cases  # noqa: E402
from business_config import (  # noqa: E402
    GeminiAgentBackend,
    ValidationError,
    load_business_contexts,
)
from business_tools import ToolRegistryError, create_default_tool_registry  # noqa: E402
from intent_classifier import classify_intent  # noqa: E402
from memory_manager import ConversationMemoryManager  # noqa: E402
from structured_extraction import extract_structured_entities  # noqa: E402
import task  # noqa: E402


class TestDay2UniversalBusinessAgent(unittest.TestCase):
    """Verify intent classification, entity extraction, tools, memory, and logging."""

    def setUp(self) -> None:
        self.contexts, self.default_id = load_business_contexts()
        self.backend = GeminiAgentBackend(force_demo=True)
        self.registry = create_default_tool_registry()
        self.dripzoid = self.contexts["dripzoid"]
        self.edusmart = self.contexts["edusmart"]

    # 1. Product inquiry classification
    def test_01_product_inquiry_classification(self) -> None:
        res = classify_intent(
            "Find black hoodies under 1000",
            backend=self.backend,
            context=self.dripzoid,
        )
        self.assertEqual(res.intent, "product_inquiry")
        self.assertGreaterEqual(res.confidence, 0.8)
        self.assertFalse(res.requires_clarification)

    # 2. Order status classification
    def test_02_order_status_classification(self) -> None:
        res = classify_intent(
            "Where is my order DZ1024?",
            backend=self.backend,
            context=self.dripzoid,
        )
        self.assertEqual(res.intent, "order_status")
        self.assertEqual(res.entities.get("order_id"), "DZ1024")
        self.assertFalse(res.requires_clarification)

    # 3. Refund request classification
    def test_03_refund_request_classification(self) -> None:
        res = classify_intent(
            "I want a refund for my order",
            backend=self.backend,
            context=self.dripzoid,
        )
        self.assertEqual(res.intent, "refund_request")
        self.assertGreaterEqual(res.confidence, 0.8)

    # 4. Structured order ID & multi-entity extraction
    def test_04_structured_order_id_extraction(self) -> None:
        entities = extract_structured_entities(
            "I want to check my order DZ1024 and know when my black hoodie will arrive.",
            backend=self.backend,
            context=self.dripzoid,
        )
        self.assertEqual(entities.order_id, "DZ1024")
        self.assertEqual(entities.product_name, "black hoodie")
        self.assertEqual(entities.requested_action, "check_delivery_status")
        self.assertIsNone(entities.customer_name)
        self.assertIsNone(entities.email)

    # 5. Product search tool
    def test_05_product_search_tool(self) -> None:
        res = self.registry.execute_tool(
            "product_search",
            {"query": "black hoodie", "max_price": 1000},
            self.dripzoid,
        )
        self.assertEqual(res.status, "success")
        self.assertEqual(res.data["count"], 2)
        names = [item["item"] for item in res.data["items"]]
        self.assertIn("Dripzoid Essential Black Hoodie", names)
        self.assertIn("Dripzoid Midnight Black Zip Hoodie", names)

    # 6. Business information tool
    def test_06_business_information_tool(self) -> None:
        res = self.registry.execute_tool(
            "business_info",
            {"topic": "hours"},
            self.dripzoid,
        )
        self.assertEqual(res.status, "success")
        self.assertIn("Monday to Saturday", res.data["support_hours"])
        self.assertEqual(res.data["contact"], "support@dripzoid.example.com")

    # 7. Order lookup for existing order
    def test_07_order_lookup_existing_order(self) -> None:
        res = self.registry.execute_tool(
            "order_lookup",
            {"order_id": "dz1024"},
            self.dripzoid,
        )
        self.assertEqual(res.status, "success")
        self.assertTrue(res.data["found"])
        self.assertTrue(res.data["is_mock_data"])
        self.assertEqual(res.data["order"]["order_id"], "DZ1024")

    # 8. Order lookup for unknown order
    def test_08_order_lookup_unknown_order(self) -> None:
        res = self.registry.execute_tool(
            "order_lookup",
            {"order_id": "DZ9999"},
            self.dripzoid,
        )
        self.assertEqual(res.status, "not_found")
        self.assertFalse(res.data["found"])

    # 9. Appointment availability (read-only preview)
    def test_09_appointment_availability(self) -> None:
        res = self.registry.execute_tool(
            "appointment_availability",
            {},
            self.dripzoid,
        )
        self.assertEqual(res.status, "success")
        self.assertFalse(res.data["booking_performed"])
        self.assertGreaterEqual(res.data["available_count"], 1)

    # 10. Correct tool selection inside end-to-end agent workflow
    def test_10_correct_tool_selection(self) -> None:
        agent = UniversalToolAgent(context=self.dripzoid, backend=self.backend)
        r_prod = agent.handle_query("Find black hoodies under 1000", stateless=True)
        r_order = agent.handle_query("Where is my order DZ1024?", stateless=True)
        r_kb = agent.handle_query("Is Cash on Delivery available?", stateless=True)
        self.assertEqual(r_prod.selected_tool, "product_search")
        self.assertEqual(r_order.selected_tool, "order_lookup")
        self.assertEqual(r_kb.selected_tool, "knowledge_base_search")

    # 11. Invalid tool arguments and unapproved tool rejection
    def test_11_invalid_tool_arguments_and_allowlist(self) -> None:
        bad_order = self.registry.execute_tool("order_lookup", {"order_id": "   "}, self.dripzoid)
        self.assertEqual(bad_order.status, "validation_error")

        bad_price = self.registry.execute_tool(
            "product_search",
            {"query": "hoodie", "max_price": -50},
            self.dripzoid,
        )
        self.assertEqual(bad_price.status, "validation_error")

        with self.assertRaises(ToolRegistryError):
            self.registry.execute_tool("run_shell", {}, self.dripzoid)

    # 12. Session memory retention & bounded window
    def test_12_session_memory_retention_and_window(self) -> None:
        memory = ConversationMemoryManager(window_size=4)
        agent = UniversalToolAgent(context=self.dripzoid, backend=self.backend, memory=memory)

        agent.handle_query("Show me black hoodies.", session_id="s1")
        follow = agent.handle_query("Which one is the cheapest?", session_id="s1")
        self.assertEqual(follow.selected_tool, "product_search")
        self.assertIn("899", follow.response_text)
        self.assertEqual(memory.inspect_session("s1")["message_count"], 4)

        # Sliding window evicts oldest turns beyond window_size=4
        agent.handle_query("What are your business hours?", session_id="s1")
        self.assertEqual(memory.inspect_session("s1")["message_count"], 4)

    # 13. Session isolation
    def test_13_session_isolation(self) -> None:
        memory = ConversationMemoryManager(window_size=4)
        agent = UniversalToolAgent(context=self.dripzoid, backend=self.backend, memory=memory)

        agent.handle_query("Show me black hoodies.", session_id="alice")
        bob_res = agent.handle_query("Which one is the cheapest?", session_id="bob")
        # Bob has no hoodie context from Alice
        self.assertNotIn("Dripzoid Essential Black Hoodie", bob_res.response_text)
        self.assertEqual(len(memory.get_history("alice")), 2)

    # 14. Memory clearing
    def test_14_memory_clearing(self) -> None:
        memory = ConversationMemoryManager(window_size=4)
        memory.add_message("temp-session", "user", "Hello")
        self.assertTrue(memory.clear_session("temp-session"))
        self.assertEqual(memory.get_history("temp-session"), [])
        self.assertFalse(memory.clear_session("temp-session"))

    # 15. Business profile switching
    def test_15_business_profile_switching(self) -> None:
        agent = UniversalToolAgent(context=self.dripzoid, backend=self.backend)
        drip_res = agent.handle_query("Where is my order DZ1024?", stateless=True)
        self.assertIn("Dripzoid Essential Black Hoodie", drip_res.response_text)

        agent.switch_business(self.edusmart)
        edu_res = agent.handle_query("Where is my order ES2041?", stateless=True)
        self.assertIn("Python & AI Agent Foundations Bootcamp", edu_res.response_text)

    # 16. Unknown query handling
    def test_16_unknown_query_handling(self) -> None:
        agent = UniversalToolAgent(context=self.dripzoid, backend=self.backend)
        res = agent.handle_query("xyz123???", stateless=True)
        self.assertEqual(res.intent.intent, "unknown")
        self.assertIsNone(res.selected_tool)
        self.assertEqual(res.execution_status, "clarification_needed")
        self.assertTrue(res.intent.requires_clarification)

    # 17. Demo mode without API key + log redaction + evaluation suite
    def test_17_demo_mode_logging_and_evaluation(self) -> None:
        exit_code = task.main(["--demo", "--profile", "dripzoid", "--query", "Where is my order DZ1024?"])
        self.assertEqual(exit_code, 0)

        agent = UniversalToolAgent(context=self.dripzoid, backend=self.backend)
        dummy_token = "AIza" + "DummyTestMasking123456"
        agent.handle_query(f"Check order DZ1024 with {dummy_token}", stateless=True)
        logs = agent.logger.get_logs()
        self.assertEqual(len(logs), 1)
        self.assertIn("[REDACTED]", logs[0].user_query)

        expected_path = DAY2_DIR / "examples" / "expected_intents.json"
        cases = json.loads(expected_path.read_text(encoding="utf-8"))
        summary = evaluate_agent_cases(cases, self.contexts, self.backend)
        self.assertEqual(summary["intent_accuracy"], 1.0)
        self.assertEqual(summary["tool_accuracy"], 1.0)
        self.assertEqual(summary["entity_accuracy"], 1.0)

        with self.assertRaises(ValidationError):
            ConversationMemoryManager(window_size=-1)


if __name__ == "__main__":
    unittest.main()
