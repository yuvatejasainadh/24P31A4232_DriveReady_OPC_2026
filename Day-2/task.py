"""Day 2: Structured Outputs, Business Tool Calling, Memory & Audit Logging.

Universal Business AI Agent | SAI Labs (Sainadh Advanced Intelligence Labs)
Drive Ready - AI Agents | One Person Company (OPC) Initiative (2026)

CLI entrypoint and interactive demonstrations extending Day 1's Gemini SDK
integration and business profiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent import UniversalToolAgent, evaluate_agent_cases
from business_config import (
    AgentError,
    BusinessContext,
    GeminiAgentBackend,
    ValidationError,
    load_business_contexts,
    load_environment,
)
from business_tools import create_default_tool_registry
from intent_classifier import classify_intent as _classify_intent_full
from memory_manager import ConversationMemoryManager
from structured_extraction import extract_structured_entities


DEFAULT_EXPECTED_INTENTS_PATH = Path(__file__).with_name("examples") / "expected_intents.json"


def classify_intent(message: str) -> dict[str, object]:
    """Compatibility wrapper returning structured intent classification JSON."""
    return _classify_intent_full(message).to_dict()


def _print_header(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def demonstrate_intent_classification(
    backend: GeminiAgentBackend,
    context: BusinessContext,
) -> None:
    """Task 1: Demonstrate structured intent classification and ambiguity handling."""
    _print_header(f"TASK 1: Business Intent Classification ({context.business_name})")

    queries = [
        "Where is my order DZ1024?"
        if context.business_id == "dripzoid"
        else "Where is my enrollment ES2041?",
        "Find black hoodies under 1000"
        if context.business_id == "dripzoid"
        else "Tell me about the Python & AI Agent Foundations Bootcamp duration",
        "I want a refund for my order",
        "xyz123???",
    ]
    for query in queries:
        res = _classify_intent_full(query, backend=backend, context=context)
        print(f"\nQuery: {query}")
        print(json.dumps(res.to_dict(), indent=2))


def demonstrate_structured_extraction(
    backend: GeminiAgentBackend,
    context: BusinessContext,
) -> None:
    """Task 2: Extract validated structured entities without inventing missing fields."""
    _print_header(f"TASK 2: Structured Information Extraction ({context.business_name})")

    samples = [
        "I want to check my order DZ1024 and know when my black hoodie will arrive.",
        "Hi, my name is Sainadh, email sainadh@example.com. Can I schedule a slot on 2026-11-28 at 4:00 PM?",
    ]
    for msg in samples:
        entities = extract_structured_entities(msg, backend=backend, context=context)
        print(f"\nInput: {msg}")
        print(json.dumps(entities.to_dict(omit_none=True), indent=2))


def demonstrate_business_tools(context: BusinessContext) -> None:
    """Task 3: Execute allowlisted tools directly with argument validation."""
    _print_header(f"TASK 3: Business Tool Registry & Direct Execution ({context.business_name})")

    registry = create_default_tool_registry()
    print("Registered Allowlisted Tools:", [t["name"] for t in registry.list_tools()])

    calls = [
        ("product_search", {"query": "black hoodie", "max_price": 1000} if context.business_id == "dripzoid" else {"query": "Python"}),
        ("business_info", {"topic": "hours"}),
        ("order_lookup", {"order_id": "DZ1024" if context.business_id == "dripzoid" else "ES2041"}),
        ("order_lookup", {"order_id": "UNKNOWN999"}),
        ("order_lookup", {"order_id": ""}),  # triggers validation_error cleanly
        ("appointment_availability", {}),
        ("knowledge_base_search", {"query": "refund"}),
    ]
    for tool_name, args in calls:
        result = registry.execute_tool(tool_name, args, context)
        print(f"\nTool: {tool_name} | Args: {args} | Status: {result.status} ({result.duration_ms} ms)")
        print(json.dumps(result.data, indent=2) if result.data else f"Message: {result.message}")


def demonstrate_agent_workflow(
    backend: GeminiAgentBackend,
    context: BusinessContext,
) -> None:
    """Task 4: End-to-end tool selection, execution, and grounded response."""
    _print_header(f"TASK 4: End-to-End Agent Tool Selection Workflow ({context.business_name})")

    agent = UniversalToolAgent(context=context, backend=backend)
    queries = (
        [
            "Find black hoodies under 1000",
            "Where is my order DZ1024?",
            "Can I check available styling appointment slots?",
        ]
        if context.business_id == "dripzoid"
        else [
            "Tell me about the Python & AI Agent Foundations Bootcamp",
            "Where is my enrollment order ES2041?",
            "What counseling appointment slots are open?",
        ]
    )

    for q in queries:
        res = agent.handle_query(q, session_id="workflow-demo")
        print(
            f"\nUser: {q}\n"
            f"-> Intent: {res.intent.intent} (conf={res.intent.confidence:.2f}) | "
            f"Tool: {res.selected_tool} | Status: {res.execution_status}\n"
            f"Agent:\n{res.response_text}"
        )


def demonstrate_conversational_memory(
    backend: GeminiAgentBackend,
    context: BusinessContext,
) -> None:
    """Task 5: Bounded session memory, session isolation, clearing, and stateless mode."""
    _print_header(f"TASK 5: Conversational Memory & Session Isolation ({context.business_name})")

    memory = ConversationMemoryManager(window_size=4)
    agent = UniversalToolAgent(context=context, backend=backend, memory=memory)

    q1 = "Show me black hoodies." if context.business_id == "dripzoid" else "Show me AI courses."
    q2 = "Which one is the cheapest?"

    r1 = agent.handle_query(q1, session_id="user_alpha")
    r2 = agent.handle_query(q2, session_id="user_alpha")
    print(f"[Session: user_alpha | Turn 1] {q1}\n{r1.response_text}\n")
    print(f"[Session: user_alpha | Turn 2] {q2}\n{r2.response_text}\n")

    # Separate session cannot see user_alpha's history
    r_other = agent.handle_query(q2, session_id="user_beta")
    print(f"[Session: user_beta (Isolated)] {q2}\n{r_other.response_text}\n")

    # Stateless override ignores stored memory
    r_stateless = agent.handle_query(q2, session_id="user_alpha", stateless=True)
    print(f"[Stateless Override on user_alpha] {q2}\n{r_stateless.response_text}\n")

    snapshot = memory.inspect_session("user_alpha")
    print(
        f"Memory Snapshot (user_alpha): mode={snapshot['mode']}, "
        f"window_size={snapshot['window_size']}, stored_messages={snapshot['message_count']}"
    )
    cleared = memory.clear_session("user_alpha")
    print(f"Cleared user_alpha memory: {cleared} | Remaining sessions: {memory.list_sessions()}")


def demonstrate_profile_switching(
    backend: GeminiAgentBackend,
    contexts: dict[str, BusinessContext],
) -> None:
    """Task 6: Switch business configurations on a single UniversalToolAgent."""
    _print_header("TASK 6: Multi-Business Runtime Switching (Same Agent Core)")

    first_id = next(iter(contexts))
    agent = UniversalToolAgent(context=contexts[first_id], backend=backend)

    for biz_id, ctx in contexts.items():
        agent.switch_business(ctx)
        res = agent.handle_query(
            "What are your business hours and refund policy?",
            session_id=f"switch-{biz_id}",
            stateless=True,
        )
        print(f"\n--- Business: {ctx.business_name} ({ctx.industry}) ---")
        print(f"Tool: {res.selected_tool} | Status: {res.execution_status}")
        print(res.response_text)


def demonstrate_logging_and_evaluation(
    backend: GeminiAgentBackend,
    contexts: dict[str, BusinessContext],
) -> None:
    """Task 7: Display structured execution logs and run evaluation suite."""
    _print_header("TASK 7: Structured Execution Logs & Agent Evaluation")

    agent = UniversalToolAgent(context=contexts["dripzoid"], backend=backend)
    agent.handle_query("Find black hoodies under 1000", session_id="log-demo")
    agent.handle_query("Where is my order DZ1024?", session_id="log-demo")

    print("Sample Structured Execution Logs:")
    for entry in agent.logger.get_logs():
        print(json.dumps(entry.to_dict(), indent=2))

    if DEFAULT_EXPECTED_INTENTS_PATH.exists():
        cases = json.loads(DEFAULT_EXPECTED_INTENTS_PATH.read_text(encoding="utf-8"))
        summary = evaluate_agent_cases(cases, contexts, backend)
        print("\nEvaluation Summary:")
        print(
            f"  Total Cases: {summary['total_cases']} | "
            f"Intent Accuracy: {summary['intent_accuracy']:.2f} | "
            f"Tool Selection Accuracy: {summary['tool_accuracy']:.2f} | "
            f"Entity Accuracy: {summary['entity_accuracy']:.2f}"
        )


def run_interactive_mode(
    backend: GeminiAgentBackend,
    contexts: dict[str, BusinessContext],
    initial_profile_id: str,
    session_id: str,
    window_size: int,
    stateless: bool,
) -> None:
    """Interactive CLI supporting tool calling, memory inspection, and business switching."""
    _print_header(f"Interactive Day 2 Agent ({backend.mode_reason})")
    memory = ConversationMemoryManager(window_size=0 if stateless else window_size)
    agent = UniversalToolAgent(
        context=contexts[initial_profile_id],
        backend=backend,
        memory=memory,
    )
    print(f"Active Business: {agent.context.business_name} ({agent.context.industry})")
    print(f"Session: {session_id} | Memory Window: {memory.window_size}")
    print("Commands: /switch <profile>, /profiles, /memory, /clear, /logs, /exit")

    while True:
        try:
            user_input = input(f"\n[{agent.context.business_id}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break

        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd in {"/exit", "exit", "quit"}:
            break
        if cmd == "/profiles":
            print("Available profiles:", ", ".join(sorted(contexts)))
            continue
        if cmd == "/memory":
            print(json.dumps(memory.inspect_session(session_id), indent=2))
            continue
        if cmd == "/clear":
            memory.clear_session(session_id)
            print(f"Cleared memory for session '{session_id}'.")
            continue
        if cmd == "/logs":
            print(json.dumps([e.to_dict() for e in agent.logger.get_logs()], indent=2))
            continue
        if cmd.startswith("/switch"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or parts[1].strip().lower() not in contexts:
                print(f"Usage: /switch <profile>. Available: {', '.join(sorted(contexts))}")
                continue
            target = parts[1].strip().lower()
            agent.switch_business(contexts[target])
            memory.clear_session(session_id)
            print(f"Switched to {agent.context.business_name}. Session memory cleared.")
            continue

        try:
            res = agent.handle_query(user_input, session_id=session_id, stateless=stateless)
            print(
                f"\n[Intent: {res.intent.intent} ({res.intent.confidence:.2f}) | "
                f"Tool: {res.selected_tool} | Status: {res.execution_status}]"
            )
            print(res.response_text)
        except AgentError as exc:
            print(f"[ERROR] {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 2: Universal Business AI Agent (Structured Outputs, Tools & Memory)"
    )
    parser.add_argument("--demo", action="store_true", help="Run in offline [MOCK DEMO] mode.")
    parser.add_argument("--live", action="store_true", help="Require live Gemini API mode.")
    parser.add_argument("--profile", default=None, help="Business profile ID (dripzoid, edusmart).")
    parser.add_argument("--query", default=None, help="Run a single query through the Day 2 agent.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat loop.")
    parser.add_argument("--session", default="cli-session", help="Session ID for conversation memory.")
    parser.add_argument(
        "--memory-window",
        type=int,
        default=6,
        help="Max messages kept per session (0 disables memory).",
    )
    parser.add_argument("--stateless", action="store_true", help="Disable session memory.")
    parser.add_argument(
        "--task",
        choices=["all", "1", "2", "3", "4", "5", "6", "7"],
        default="all",
        help="Run all tasks or a specific Day 2 task (1-7).",
    )
    parser.add_argument("--eval", action="store_true", help="Run evaluation suite and exit.")
    parser.add_argument("--show-logs", action="store_true", help="Print structured execution logs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_environment()
    args = parse_args(argv)

    try:
        contexts, default_id = load_business_contexts()
        selected_id = (args.profile or default_id).strip().lower()
        if selected_id not in contexts:
            raise ValidationError(
                f"Unknown profile '{selected_id}'. Available: {', '.join(sorted(contexts))}"
            )

        backend = GeminiAgentBackend(force_demo=args.demo, require_live=args.live)
        active_context = contexts[selected_id]

        print("Day 2 - Universal Business AI Agent (Structured Outputs, Tools & Memory)")
        print(f"Execution Mode: {backend.mode_reason}")
        print(f"Active Business: {active_context.business_name} ({active_context.industry})")

        if args.eval:
            cases = json.loads(DEFAULT_EXPECTED_INTENTS_PATH.read_text(encoding="utf-8"))
            summary = evaluate_agent_cases(cases, contexts, backend)
            print(json.dumps(summary, indent=2))
            return 0

        if args.query is not None:
            memory = ConversationMemoryManager(
                window_size=0 if args.stateless else args.memory_window
            )
            agent = UniversalToolAgent(context=active_context, backend=backend, memory=memory)
            res = agent.handle_query(
                args.query,
                session_id=args.session,
                stateless=args.stateless,
            )
            print(f"\nQuery: {args.query}")
            print(f"Intent: {res.intent.intent} (confidence={res.intent.confidence:.2f})")
            print(f"Entities: {json.dumps(res.entities.to_dict(omit_none=True))}")
            print(f"Tool: {res.selected_tool} | Status: {res.execution_status}")
            print(f"\nResponse:\n{res.response_text}")
            if args.show_logs:
                print("\nExecution Log:")
                print(json.dumps([e.to_dict() for e in agent.logger.get_logs()], indent=2))
            return 0

        if args.interactive:
            run_interactive_mode(
                backend=backend,
                contexts=contexts,
                initial_profile_id=selected_id,
                session_id=args.session,
                window_size=args.memory_window,
                stateless=args.stateless,
            )
            return 0

        runners = {
            "1": lambda: demonstrate_intent_classification(backend, active_context),
            "2": lambda: demonstrate_structured_extraction(backend, active_context),
            "3": lambda: demonstrate_business_tools(active_context),
            "4": lambda: demonstrate_agent_workflow(backend, active_context),
            "5": lambda: demonstrate_conversational_memory(backend, active_context),
            "6": lambda: demonstrate_profile_switching(backend, contexts),
            "7": lambda: demonstrate_logging_and_evaluation(backend, contexts),
        }
        if args.task == "all":
            for tid in ("1", "2", "3", "4", "5", "6", "7"):
                runners[tid]()
        else:
            runners[args.task]()
        return 0
    except AgentError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
