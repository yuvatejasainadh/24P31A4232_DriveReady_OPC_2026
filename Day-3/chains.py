"""Task 1: LangChain & LCEL Pipelines for Day 3.

Builds reusable LCEL chains using `langchain_core` (`RunnableSequence`,
`RunnableLambda`, `RunnablePassthrough`, `RunnableParallel`, `ChatPromptTemplate`,
`StrOutputParser`) and `langchain_google_genai` (`ChatGoogleGenerativeAI`) when
installed, with a transparent pure-Python LCEL-compatible fallback when those
optional packages are not installed in the offline environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Callable

DAY2_DIR = Path(__file__).resolve().parent.parent / "Day-2"
if str(DAY2_DIR) not in sys.path:
    sys.path.append(str(DAY2_DIR))

from business_config import (  # noqa: E402
    BusinessContext,
    GeminiAgentBackend,
    ValidationError,
    day1_task,
    validate_non_empty_text,
)
from grounding import KnowledgeRetriever  # noqa: E402
from intent_classifier import IntentResult, classify_intent  # noqa: E402
from memory_manager import ConversationMemoryManager  # noqa: E402
from structured_extraction import ExtractedEntities, extract_structured_entities  # noqa: E402

try:  # Prefer official langchain-core LCEL primitives when installed
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import (
        RunnableLambda,
        RunnableParallel,
        RunnablePassthrough,
        RunnableSequence,
    )

    LCEL_BACKEND = "langchain_core"
except ImportError:  # Offline-safe LCEL compatibility layer matching `.invoke()` and `|`
    LCEL_BACKEND = "builtin_lcel_compat"

    class _CompatRunnable:
        """Minimal LCEL runnable interface supporting invoke() and pipe (|) composition."""

        def invoke(self, input_data: Any, config: Any = None) -> Any:  # pragma: no cover
            raise NotImplementedError

        def __or__(self, other: Any) -> "RunnableSequence":
            right = other if isinstance(other, _CompatRunnable) else RunnableLambda(other)
            if isinstance(self, RunnableSequence):
                return RunnableSequence(*self.steps, right)
            return RunnableSequence(self, right)

        def __ror__(self, other: Any) -> "RunnableSequence":
            left = other if isinstance(other, _CompatRunnable) else RunnableLambda(other)
            if isinstance(self, RunnableSequence):
                return RunnableSequence(left, *self.steps)
            return RunnableSequence(left, self)

    class RunnableLambda(_CompatRunnable):
        def __init__(self, func: Callable[[Any], Any]) -> None:
            if not callable(func):
                raise TypeError("RunnableLambda requires a callable.")
            self.func = func

        def invoke(self, input_data: Any, config: Any = None) -> Any:
            return self.func(input_data)

    class RunnablePassthrough(_CompatRunnable):
        @staticmethod
        def assign(**kwargs: Any) -> "RunnableLambda":
            def _assign(data: dict[str, Any]) -> dict[str, Any]:
                out = dict(data)
                for key, runnable in kwargs.items():
                    r = runnable if isinstance(runnable, _CompatRunnable) else RunnableLambda(runnable)
                    out[key] = r.invoke(data)
                return out

            return RunnableLambda(_assign)

        def invoke(self, input_data: Any, config: Any = None) -> Any:
            return input_data

    class RunnableSequence(_CompatRunnable):
        def __init__(self, first: Any, *others: Any) -> None:
            raw_steps = [first, *others]
            self.steps = [
                s if isinstance(s, _CompatRunnable) else RunnableLambda(s)
                for s in raw_steps
            ]

        def invoke(self, input_data: Any, config: Any = None) -> Any:
            current = input_data
            for step in self.steps:
                current = step.invoke(current, config=config)
            return current

    class RunnableParallel(_CompatRunnable):
        def __init__(self, mapping: dict[str, Any] | None = None, **kwargs: Any) -> None:
            combined = dict(mapping or {})
            combined.update(kwargs)
            self.branches = {
                k: (v if isinstance(v, _CompatRunnable) else RunnableLambda(v))
                for k, v in combined.items()
            }

        def invoke(self, input_data: Any, config: Any = None) -> dict[str, Any]:
            return {
                key: branch.invoke(input_data, config=config)
                for key, branch in self.branches.items()
            }

    @dataclass
    class _CompatPromptValue:
        messages: list[tuple[str, str]]

        def to_string(self) -> str:
            return "\n".join(f"[{role.upper()}] {content}" for role, content in self.messages)

    class ChatPromptTemplate(_CompatRunnable):
        def __init__(self, messages: list[tuple[str, str]]) -> None:
            self.messages = messages

        @classmethod
        def from_messages(cls, messages: list[tuple[str, str]]) -> "ChatPromptTemplate":
            return cls(list(messages))

        def invoke(self, input_data: dict[str, Any], config: Any = None) -> _CompatPromptValue:
            rendered = [
                (role, template.format(**input_data))
                for role, template in self.messages
            ]
            return _CompatPromptValue(messages=rendered)

    class StrOutputParser(_CompatRunnable):
        def invoke(self, input_data: Any, config: Any = None) -> str:
            if isinstance(input_data, str):
                return input_data
            if hasattr(input_data, "content"):
                return str(input_data.content)
            if hasattr(input_data, "to_string"):
                return str(input_data.to_string())
            return str(input_data)


try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore[assignment,misc]


def build_context_preparation_chain(
    contexts: dict[str, BusinessContext],
) -> Any:
    """LCEL Chain 1: Validate query and attach active BusinessContext."""

    def _prepare(payload: dict[str, Any]) -> dict[str, Any]:
        query = validate_non_empty_text(payload.get("query"), "query")
        profile_id = str(payload.get("business_id", "")).strip().lower()
        if profile_id not in contexts:
            raise ValidationError(
                f"Unknown business profile '{profile_id}'. Available: {', '.join(sorted(contexts))}"
            )
        session_id = validate_non_empty_text(payload.get("session_id", "default"), "session_id")
        return {
            **payload,
            "query": query,
            "business_id": profile_id,
            "session_id": session_id,
            "context": contexts[profile_id],
        }

    return RunnablePassthrough() | RunnableLambda(_prepare)


def build_intent_and_entity_chain(
    backend: GeminiAgentBackend,
    memory_manager: ConversationMemoryManager,
) -> Any:
    """LCEL Chain 2: Parallel memory lookup + single-pass intent/entity extraction.

    Avoids duplicate live LLM calls by calling `classify_intent(...)` once in
    live mode (which already invokes `extract_structured_entities` internally)
    and reconstructing `ExtractedEntities` from `intent_result.entities`.
    """

    def _load_memory(payload: dict[str, Any]) -> dict[str, Any]:
        stateless = bool(payload.get("stateless", False)) or memory_manager.is_stateless
        recent_texts = (
            []
            if stateless
            else memory_manager.get_recent_texts(payload["session_id"])
        )
        return {"stateless": stateless, "recent_history": recent_texts}

    def _analyze_query(payload: dict[str, Any]) -> dict[str, Any]:
        stateless = bool(payload.get("stateless", False)) or memory_manager.is_stateless
        recent_texts = (
            []
            if stateless
            else memory_manager.get_recent_texts(payload["session_id"])
        )
        context: BusinessContext = payload["context"]
        query: str = payload["query"]

        intent_res: IntentResult = classify_intent(
            query,
            backend=backend,
            context=context,
            recent_history=recent_texts,
        )
        if backend.is_mock:
            entities = extract_structured_entities(
                query,
                backend=backend,
                context=context,
                recent_history=recent_texts,
            )
            llm_calls = 0
        else:
            # Reuse entities extracted during live classify_intent to prevent extra API calls
            from structured_extraction import normalize_extracted_entities

            entities = normalize_extracted_entities(intent_res.entities)
            llm_calls = 2  # classify_intent + internal entity extraction

        # Route explicit ID lookup queries (e.g., "Check enrollment ES2041.") to order_status
        if (
            entities.order_id
            and intent_res.intent in {"course_information", "general_inquiry", "unknown"}
            and any(kw in query.lower() for kw in ("check", "status", "track", "lookup", "enrollment"))
        ):
            intent_res = IntentResult(
                intent="order_status",
                confidence=max(intent_res.confidence, 0.93),
                entities=entities.to_dict(omit_none=True),
                requires_clarification=False,
                reason="Detected explicit order/enrollment ID with lookup verb.",
                is_mock=intent_res.is_mock,
            )
        elif intent_res.intent == "unknown":
            # Promote to general_inquiry when high-relevance FAQ/policy knowledge exists
            ret = KnowledgeRetriever().retrieve(query, context, top_k=2)
            if any(s.source_type in {"faq", "policy"} and s.score >= 3 for s in ret.snippets):
                intent_res = IntentResult(
                    intent="general_inquiry",
                    confidence=0.82,
                    entities=entities.to_dict(omit_none=True),
                    requires_clarification=False,
                    reason="Matched verified business FAQ/policy snippet.",
                    is_mock=intent_res.is_mock,
                )

        return {
            "intent_result": intent_res,
            "entities": entities,
            "analysis_llm_calls": llm_calls,
        }

    return RunnableParallel(
        {
            "input": RunnablePassthrough(),
            "memory_state": RunnableLambda(_load_memory),
            "analysis": RunnableLambda(_analyze_query),
        }
    ) | RunnableLambda(
        lambda out: {
            **out["input"],
            **out["memory_state"],
            **out["analysis"],
        }
    )


def build_knowledge_retrieval_chain(retriever: KnowledgeRetriever) -> Any:
    """LCEL Chain 3: Retrieve grounded business facts scoped to active BusinessContext."""

    def _retrieve(payload: dict[str, Any]) -> dict[str, Any]:
        retrieval = retriever.retrieve(payload["query"], payload["context"])
        return {
            **payload,
            "retrieval": retrieval,
            "knowledge_context": retrieval.to_context_text(),
        }

    return RunnableLambda(_retrieve)


def _extract_system_and_human_from_prompt(prompt_value: Any) -> tuple[str, str]:
    """Convert ChatPromptTemplate output into (system_instruction, user_prompt)."""
    if hasattr(prompt_value, "to_messages"):
        msgs = prompt_value.to_messages()
        system_parts = [str(m.content) for m in msgs if getattr(m, "type", "") == "system"]
        human_parts = [str(m.content) for m in msgs if getattr(m, "type", "") != "system"]
        return "\n".join(system_parts), "\n".join(human_parts)
    if hasattr(prompt_value, "messages"):
        msgs = prompt_value.messages
        system_parts = [c for role, c in msgs if role == "system"]
        human_parts = [c for role, c in msgs if role != "system"]
        return "\n".join(system_parts), "\n".join(human_parts)
    return "", str(prompt_value)


def build_response_generation_chain(backend: GeminiAgentBackend) -> Any:
    """LCEL Chain 4: ChatPromptTemplate -> Gemini / Demo Responder -> StrOutputParser."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the official AI assistant for {business_name} ({industry}).\n"
                "Tone: {tone}\n"
                "STRICT GROUNDING RULES:\n"
                "1. Answer ONLY using the Retrieved Knowledge and Verified Tool Result below.\n"
                "2. Never invent prices, policies, orders, or bookings.\n"
                "3. If information is missing, state clearly that you do not have enough information.\n\n"
                "Retrieved Knowledge:\n{knowledge_context}\n\n"
                "Verified Tool Result:\n{tool_result_text}",
            ),
            ("human", "Customer Query: {query}"),
        ]
    )

    def _invoke_model(prompt_value: Any) -> str:
        system_instruction, user_prompt = _extract_system_and_human_from_prompt(prompt_value)
        if not backend.is_mock:
            # Prefer ChatGoogleGenerativeAI if installed + API key configured, else Day 1 backend
            api_key = day1_task.get_configured_api_key()
            if ChatGoogleGenerativeAI is not None and api_key:
                model = ChatGoogleGenerativeAI(
                    model=backend.model,
                    google_api_key=api_key,
                    temperature=0.2,
                )
                chain_msg = model.invoke(prompt_value)
                return str(getattr(chain_msg, "content", chain_msg))
            live_res = backend.generate(
                user_prompt=user_prompt,
                system_instruction=system_instruction,
                temperature=0.2,
            )
            return live_res.text

        # Deterministic demo response fallback when no API key is active
        return user_prompt

    return prompt | RunnableLambda(_invoke_model) | StrOutputParser()


def build_final_formatting_chain() -> Any:
    """LCEL Chain 5: Normalize final response text and attach execution summary."""

    def _format(payload: dict[str, Any]) -> dict[str, Any]:
        raw_response = str(payload.get("raw_response", "")).strip()
        is_mock = bool(payload.get("is_mock", True))
        context: BusinessContext = payload["context"]
        prefix = f"[MOCK DEMO | {context.business_name}]"
        if is_mock and not raw_response.startswith("[MOCK DEMO") and not raw_response.startswith("[SAFE GUARD"):
            raw_response = f"{prefix} {raw_response}"
        return {
            **payload,
            "final_response": raw_response,
        }

    return RunnableSequence(RunnablePassthrough(), RunnableLambda(_format))


def build_master_lcel_pipeline(
    contexts: dict[str, BusinessContext],
    backend: GeminiAgentBackend,
    memory_manager: ConversationMemoryManager,
    retriever: KnowledgeRetriever,
) -> Any:
    """Compose Chains 1 -> 2 -> 3 into a reusable LCEL pre-decision RunnableSequence."""
    c1 = build_context_preparation_chain(contexts)
    c2 = build_intent_and_entity_chain(backend, memory_manager)
    c3 = build_knowledge_retrieval_chain(retriever)
    return RunnableSequence(c1, c2, c3)


def summarize_policy_via_lcel(
    backend: GeminiAgentBackend,
    context: BusinessContext,
    policy_topic: str = "returns_and_exchanges",
) -> str:
    """Task 3 helper: Demonstrate standalone LLM LCEL summarization without tools."""
    policy_text = context.policies.get(
        policy_topic,
        next(iter(context.policies.values()), "No policy available."),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the following verified {business_name} policy in 1-2 clear sentences.",
            ),
            ("human", "Policy ({topic}): {policy_text}"),
        ]
    )

    def _llm_or_mock(prompt_value: Any) -> str:
        sys_inst, usr_msg = _extract_system_and_human_from_prompt(prompt_value)
        if not backend.is_mock:
            api_key = day1_task.get_configured_api_key()
            if ChatGoogleGenerativeAI is not None and api_key:
                llm = ChatGoogleGenerativeAI(
                    model=backend.model,
                    google_api_key=api_key,
                    temperature=0.2,
                )
                return str(llm.invoke(prompt_value).content)
            return backend.generate(
                user_prompt=usr_msg,
                system_instruction=sys_inst,
                temperature=0.2,
            ).text

        return f"[MOCK DEMO | LLM-Only Summary] {context.business_name}: {policy_text}"

    chain = prompt | RunnableLambda(_llm_or_mock) | StrOutputParser()
    return str(
        chain.invoke(
            {
                "business_name": context.business_name,
                "topic": policy_topic.replace("_", " "),
                "policy_text": policy_text,
            }
        )
    )
