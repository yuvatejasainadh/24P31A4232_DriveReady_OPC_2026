"""Day 1: LLM Foundations, Prompt Engineering, Grounding, Stateless APIs, Token Usage,
and Configurable Business Profiles.

Universal Business AI Agent | SAI Labs (Sainadh Advanced Intelligence Labs)
Drive Ready - AI Agents | One Person Company (OPC) Initiative (2026)

This module implements all six Day 1 exercises using the current official
Google GenAI Python SDK (`google-genai`) when a valid `GEMINI_API_KEY` is
configured, and includes a clearly labeled `[MOCK DEMO]` fallback mode so
every feature can be executed and verified offline without an API key.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional if environment vars are already exported
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_PROFILES_PATH = Path(__file__).with_name("business_profiles.json")
PLACEHOLDER_KEYS = {"", "your_api_key_here", "replace_me", "changeme"}


class AgentError(Exception):
    """Base exception for Day 1 Universal Business AI Agent errors."""


class ConfigurationError(AgentError):
    """Raised when environment, SDK, or business profile configuration is invalid."""


class ValidationError(AgentError):
    """Raised when user inputs or model parameters fail validation."""


class LLMExecutionError(AgentError):
    """Raised when a live Gemini API call fails."""


@dataclass(frozen=True)
class BusinessProfile:
    """Validated configuration and local knowledge base for a single business."""

    business_id: str
    business_name: str
    industry: str
    description: str
    tone: str
    services: list[str]
    support_topics: list[str]
    knowledge_base: dict[str, Any]


@dataclass(frozen=True)
class RenderedPrompt:
    """Reusable prompt bundle produced by Task 2's prompt template."""

    system_instruction: str
    user_prompt: str


@dataclass(frozen=True)
class ConversationTurn:
    """Single message turn in a multi-turn conversation history."""

    role: str  # "user" or "model"
    text: str


@dataclass(frozen=True)
class TokenMetrics:
    """Separates local character/word estimates from live API usage_metadata."""

    prompt_chars: int
    prompt_words: int
    estimated_prompt_tokens: int
    response_chars: int
    response_words: int
    estimated_response_tokens: int
    api_prompt_tokens: int | None = None
    api_candidates_tokens: int | None = None
    api_total_tokens: int | None = None
    api_thoughts_tokens: int | None = None


@dataclass(frozen=True)
class LLMResult:
    """Normalized output returned by both live Gemini and mock demo backends."""

    text: str
    model: str
    temperature: float
    is_mock: bool
    system_instruction: str
    token_metrics: TokenMetrics
    backend_note: str = ""


# ---------------------------------------------------------------------------
# Input validation & environment helpers
# ---------------------------------------------------------------------------


def validate_non_empty_text(value: Any, field_name: str) -> str:
    """Ensure a string input is a non-empty string after stripping whitespace."""
    if not isinstance(value, str):
        raise ValidationError(f"'{field_name}' must be a string, got {type(value).__name__}.")
    cleaned = value.strip()
    if not cleaned:
        raise ValidationError(f"'{field_name}' cannot be empty or whitespace.")
    return cleaned


def validate_temperature(temperature: Any) -> float:
    """Validate that temperature is numeric and within Gemini's [0.0, 2.0] range."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValidationError(f"'temperature' must be a number, got {type(temperature).__name__}.")
    value = float(temperature)
    if not 0.0 <= value <= 2.0:
        raise ValidationError(f"'temperature' must be between 0.0 and 2.0, got {value}.")
    return value


def load_environment() -> None:
    """Load environment variables from `.env` if python-dotenv is available."""
    if load_dotenv is None:
        return
    day_env = Path(__file__).with_name(".env")
    root_env = Path(__file__).resolve().parent.parent / ".env"
    if day_env.exists():
        load_dotenv(dotenv_path=day_env, override=False)
    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=False)


def get_configured_api_key() -> str | None:
    """Return a valid-looking GEMINI_API_KEY, ignoring empty or placeholder strings."""
    raw = os.getenv("GEMINI_API_KEY", "").strip()
    if raw.lower() in PLACEHOLDER_KEYS:
        return None
    return raw


def get_configured_model() -> str:
    """Return the configured Gemini model name or the default model."""
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()
    return model or DEFAULT_MODEL


def estimate_tokens(text: str) -> int:
    """Approximate English token count (~4 characters per token) for local budgeting.

    NOTE: This is strictly an estimate. Actual billing and context limits depend on
    the model's tokenizer and are reported by the Gemini API in `usage_metadata`.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _build_token_metrics(
    prompt_text: str,
    response_text: str,
    api_prompt_tokens: int | None = None,
    api_candidates_tokens: int | None = None,
    api_total_tokens: int | None = None,
    api_thoughts_tokens: int | None = None,
) -> TokenMetrics:
    """Compute local text length metrics and attach API usage metadata if present."""
    return TokenMetrics(
        prompt_chars=len(prompt_text),
        prompt_words=len(prompt_text.split()),
        estimated_prompt_tokens=estimate_tokens(prompt_text),
        response_chars=len(response_text),
        response_words=len(response_text.split()),
        estimated_response_tokens=estimate_tokens(response_text),
        api_prompt_tokens=api_prompt_tokens,
        api_candidates_tokens=api_candidates_tokens,
        api_total_tokens=api_total_tokens,
        api_thoughts_tokens=api_thoughts_tokens,
    )


# ---------------------------------------------------------------------------
# Task 6 & Task 3: Business profile loading & knowledge base formatting
# ---------------------------------------------------------------------------


def load_business_profiles(
    profiles_path: Path = DEFAULT_PROFILES_PATH,
) -> tuple[dict[str, BusinessProfile], str]:
    """Load and validate business profiles from a JSON configuration file."""
    if not isinstance(profiles_path, Path):
        profiles_path = Path(profiles_path)

    if not profiles_path.exists():
        raise ConfigurationError(f"Business profiles file not found: {profiles_path}")

    try:
        raw_data = json.loads(profiles_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Malformed JSON in business profiles file '{profiles_path}': {exc}"
        ) from exc

    if not isinstance(raw_data, dict) or "profiles" not in raw_data:
        raise ConfigurationError("Business profiles JSON must be an object with a 'profiles' key.")

    raw_profiles = raw_data["profiles"]
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ConfigurationError("'profiles' must be a non-empty mapping of profile IDs to configs.")

    required_fields = (
        "business_name",
        "industry",
        "description",
        "tone",
        "services",
        "support_topics",
        "knowledge_base",
    )

    parsed: dict[str, BusinessProfile] = {}
    for profile_key, entry in raw_profiles.items():
        if not isinstance(entry, dict):
            raise ConfigurationError(f"Profile '{profile_key}' must be a JSON object.")

        missing = [field_name for field_name in required_fields if field_name not in entry]
        if missing:
            raise ConfigurationError(
                f"Profile '{profile_key}' is missing required fields: {', '.join(missing)}"
            )

        business_id = str(entry.get("business_id") or profile_key).strip().lower()
        business_name = validate_non_empty_text(entry["business_name"], f"{profile_key}.business_name")
        industry = validate_non_empty_text(entry["industry"], f"{profile_key}.industry")
        description = validate_non_empty_text(entry["description"], f"{profile_key}.description")
        tone = validate_non_empty_text(entry["tone"], f"{profile_key}.tone")

        services = entry["services"]
        support_topics = entry["support_topics"]
        knowledge_base = entry["knowledge_base"]

        if not isinstance(services, list) or not all(isinstance(s, str) and s.strip() for s in services):
            raise ConfigurationError(f"Profile '{profile_key}.services' must be a list of strings.")
        if not isinstance(support_topics, list) or not all(
            isinstance(t, str) and t.strip() for t in support_topics
        ):
            raise ConfigurationError(
                f"Profile '{profile_key}.support_topics' must be a list of strings."
            )
        if not isinstance(knowledge_base, dict) or not knowledge_base:
            raise ConfigurationError(
                f"Profile '{profile_key}.knowledge_base' must be a non-empty JSON object."
            )

        parsed[business_id] = BusinessProfile(
            business_id=business_id,
            business_name=business_name,
            industry=industry,
            description=description,
            tone=tone,
            services=[s.strip() for s in services],
            support_topics=[t.strip() for t in support_topics],
            knowledge_base=knowledge_base,
        )

    default_profile = str(raw_data.get("default_profile") or next(iter(parsed))).strip().lower()
    if default_profile not in parsed:
        raise ConfigurationError(
            f"default_profile '{default_profile}' does not exist in profiles: {sorted(parsed)}"
        )

    return parsed, default_profile


def format_knowledge_base(profile: BusinessProfile) -> str:
    """Serialize a BusinessProfile's local knowledge base into readable grounded context."""
    lines = [
        f"Business Name: {profile.business_name}",
        f"Industry: {profile.industry}",
        f"Description: {profile.description}",
        f"Services: {', '.join(profile.services)}",
        f"Support Topics: {', '.join(profile.support_topics)}",
        "",
        "Verified Knowledge Base:",
    ]

    kb = profile.knowledge_base
    catalog = kb.get("catalog", [])
    if isinstance(catalog, list) and catalog:
        lines.append("Catalog / Offerings:")
        for item in catalog:
            if isinstance(item, dict):
                details = ", ".join(
                    f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                    for k, v in item.items()
                )
                lines.append(f"  - {details}")

    policies = kb.get("policies", {})
    if isinstance(policies, dict) and policies:
        lines.append("Official Policies:")
        for policy_name, policy_text in policies.items():
            label = policy_name.replace("_", " ").title()
            lines.append(f"  - {label}: {policy_text}")

    for extra_key, extra_val in kb.items():
        if extra_key in {"catalog", "policies"}:
            continue
        label = extra_key.replace("_", " ").title()
        if isinstance(extra_val, (dict, list)):
            lines.append(f"{label}: {json.dumps(extra_val, ensure_ascii=False)}")
        else:
            lines.append(f"{label}: {extra_val}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 2 & Task 3: Reusable Prompt Template and Grounding Instructions
# ---------------------------------------------------------------------------


def build_business_prompt_template(
    business_name: str,
    business_industry: str,
    business_description: str,
    business_tone: str,
    user_query: str,
) -> RenderedPrompt:
    """Task 2: Build a reusable, validated prompt template for any business profile."""
    name = validate_non_empty_text(business_name, "business_name")
    industry = validate_non_empty_text(business_industry, "business_industry")
    description = validate_non_empty_text(business_description, "business_description")
    tone = validate_non_empty_text(business_tone, "business_tone")
    query = validate_non_empty_text(user_query, "user_query")

    system_instruction = (
        f"You are the official AI customer assistant for {name}.\n"
        f"Industry: {industry}\n"
        f"Business Overview: {description}\n"
        f"Required Tone: {tone}\n\n"
        "Behavioral Rules:\n"
        "1. Stay strictly in character as the assistant for this business.\n"
        "2. Answer accurately using only verified business context provided in instructions.\n"
        "3. Never fabricate prices, policies, timelines, or product guarantees.\n"
        "4. If required facts are missing, state clearly that you do not have enough information."
    )

    user_prompt = f"Customer Query for {name}: {query}"
    return RenderedPrompt(system_instruction=system_instruction, user_prompt=user_prompt)


def build_grounded_system_instruction(profile: BusinessProfile) -> str:
    """Task 3: Combine business identity prompt with strict grounded knowledge rules."""
    base_prompt = build_business_prompt_template(
        business_name=profile.business_name,
        business_industry=profile.industry,
        business_description=profile.description,
        business_tone=profile.tone,
        user_query="placeholder",
    ).system_instruction

    kb_block = format_knowledge_base(profile)
    return (
        f"{base_prompt}\n\n"
        "STRICT GROUNDING CONSTRAINT:\n"
        "Prioritize the Verified Knowledge Base below above all general assumptions. "
        "Do NOT invent unsupported discounts, policies, warranties, prices, delivery speeds, "
        "schedules, or contact channels. If the answer is not supported by the Verified "
        "Knowledge Base, respond clearly that you do not have enough information.\n\n"
        f"--- BEGIN BUSINESS KNOWLEDGE BASE ---\n{kb_block}\n--- END BUSINESS KNOWLEDGE BASE ---"
    )


# ---------------------------------------------------------------------------
# LLM Client Abstraction (Live Gemini `google-genai` SDK + Deterministic Mock)
# ---------------------------------------------------------------------------


class GeminiAgentBackend:
    """Executes single-turn and multi-turn requests via `google-genai` or demo fallback."""

    def __init__(
        self,
        force_demo: bool = False,
        require_live: bool = False,
        model: str | None = None,
    ) -> None:
        if force_demo and require_live:
            raise ValidationError("Cannot enable both force_demo and require_live.")

        self.model = model or get_configured_model()
        api_key = get_configured_api_key()

        if force_demo:
            self.is_mock = True
            self._client = None
            self.mode_reason = "Demo mode requested via --demo flag."
            return

        if api_key is None:
            if require_live:
                raise ConfigurationError(
                    "Live mode requested, but GEMINI_API_KEY is missing or still set to a placeholder."
                )
            self.is_mock = True
            self._client = None
            self.mode_reason = (
                "GEMINI_API_KEY is not configured; automatically running in [MOCK DEMO] mode."
            )
            return

        if genai is None or types is None:
            if require_live:
                raise ConfigurationError(
                    "The 'google-genai' package is not installed. Run `pip install -r requirements.txt`."
                )
            self.is_mock = True
            self._client = None
            self.mode_reason = (
                "'google-genai' SDK is not installed; automatically running in [MOCK DEMO] mode."
            )
            return

        self.is_mock = False
        self._client = genai.Client(api_key=api_key)
        self.mode_reason = f"Live Gemini API mode active (model: {self.model})."

    def generate(
        self,
        user_prompt: str,
        system_instruction: str = "",
        temperature: float = 0.2,
        history: list[ConversationTurn] | None = None,
        profile: BusinessProfile | None = None,
        max_output_tokens: int = 512,
    ) -> LLMResult:
        """Generate a response from Gemini or the deterministic mock demo backend."""
        cleaned_prompt = validate_non_empty_text(user_prompt, "user_prompt")
        valid_temp = validate_temperature(temperature)
        sys_inst = system_instruction.strip()
        turns = list(history or [])

        if self.is_mock:
            return self._generate_mock(
                user_prompt=cleaned_prompt,
                system_instruction=sys_inst,
                temperature=valid_temp,
                history=turns,
                profile=profile,
            )

        return self._generate_live(
            user_prompt=cleaned_prompt,
            system_instruction=sys_inst,
            temperature=valid_temp,
            history=turns,
            max_output_tokens=max_output_tokens,
        )

    def _generate_live(
        self,
        user_prompt: str,
        system_instruction: str,
        temperature: float,
        history: list[ConversationTurn],
        max_output_tokens: int,
    ) -> LLMResult:
        """Call the current `google-genai` SDK (`client.models.generate_content`)."""
        assert self._client is not None and types is not None

        contents: list[Any] = []
        for turn in history:
            role = "model" if turn.role == "model" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=turn.text)],
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            )
        )

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise LLMExecutionError(
                f"Gemini API call failed ({type(exc).__name__}): {exc}"
            ) from exc

        response_text = (getattr(response, "text", None) or "").strip()
        if not response_text:
            response_text = "[Empty response returned by Gemini API]"

        usage = getattr(response, "usage_metadata", None)
        api_prompt = getattr(usage, "prompt_token_count", None) if usage else None
        api_candidates = getattr(usage, "candidates_token_count", None) if usage else None
        api_total = getattr(usage, "total_token_count", None) if usage else None
        api_thoughts = getattr(usage, "thoughts_token_count", None) if usage else None

        combined_prompt_for_estimate = "\n".join(
            part for part in [system_instruction, *(t.text for t in history), user_prompt] if part
        )
        metrics = _build_token_metrics(
            prompt_text=combined_prompt_for_estimate,
            response_text=response_text,
            api_prompt_tokens=api_prompt,
            api_candidates_tokens=api_candidates,
            api_total_tokens=api_total,
            api_thoughts_tokens=api_thoughts,
        )

        return LLMResult(
            text=response_text,
            model=self.model,
            temperature=temperature,
            is_mock=False,
            system_instruction=system_instruction,
            token_metrics=metrics,
            backend_note="Live Gemini API response.",
        )

    def _generate_mock(
        self,
        user_prompt: str,
        system_instruction: str,
        temperature: float,
        history: list[ConversationTurn],
        profile: BusinessProfile | None,
    ) -> LLMResult:
        """Produce clearly labeled, realistic mock responses without network/API access."""
        reply = self._route_mock_response(
            user_prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            history=history,
            profile=profile,
        )

        combined_prompt = "\n".join(
            part for part in [system_instruction, *(t.text for t in history), user_prompt] if part
        )
        metrics = _build_token_metrics(
            prompt_text=combined_prompt,
            response_text=reply,
            api_prompt_tokens=None,
            api_candidates_tokens=None,
            api_total_tokens=None,
            api_thoughts_tokens=None,
        )

        return LLMResult(
            text=reply,
            model=f"{self.model} (MOCK DEMO)",
            temperature=temperature,
            is_mock=True,
            system_instruction=system_instruction,
            token_metrics=metrics,
            backend_note="Mock demo response (no live Gemini API call executed).",
        )

    def _route_mock_response(
        self,
        user_prompt: str,
        system_instruction: str,
        temperature: float,
        history: list[ConversationTurn],
        profile: BusinessProfile | None,
    ) -> str:
        """Simulate prompt sensitivity, temperature, grounding, and memory behavior."""
        lower_prompt = user_prompt.lower()

        # Task 1: Prompt influence & temperature demos (when no business profile is bound)
        if profile is None and "tagline" in lower_prompt and "streetwear" in lower_prompt:
            if temperature <= 0.3:
                return (
                    "[MOCK DEMO | temp=low] 1. Everyday Streetwear, Built to Last.\n"
                    "2. Clean Fits for Daily Wear."
                )
            return (
                "[MOCK DEMO | temp=high] 1. Drop Loud, Walk Proud — Urban Threads Reimagined!\n"
                "2. Midnight Concrete Meets Heavyweight Cotton Energy."
            )

        if profile is None and "how does a large language model work" in lower_prompt:
            if "bullet" in lower_prompt or "2 sentences" in lower_prompt or "3 concise" in lower_prompt:
                return (
                    "[MOCK DEMO] 1. An LLM predicts the next most likely token based on your prompt.\n"
                    "2. System instructions set global rules, while user prompts pass the immediate task.\n"
                    "3. It relies on patterns learned during training plus whatever context you supply."
                )
            return (
                "[MOCK DEMO] A large language model is a neural network trained on vast text datasets "
                "to predict the next token in a sequence given prior context."
            )

        if profile is None and "what is your return policy" in lower_prompt:
            if "dripzoid" in system_instruction.lower():
                return (
                    "[MOCK DEMO | system=Dripzoid] At Dripzoid, unworn items with original tags "
                    "can be returned or exchanged within 7 days of delivery."
                )
            return (
                "[MOCK DEMO | no system role] Return policies vary by merchant. Please provide the "
                "specific store name or policy document you would like me to reference."
            )

        # Business-profile-driven mock response (Tasks 2, 3, 4, 5, 6)
        if profile is not None:
            return _mock_grounded_business_reply(
                profile=profile,
                user_prompt=user_prompt,
                history=history,
                temperature=temperature,
            )

        # Fallback generic mock response
        tone_hint = "concise" if temperature < 0.4 else "expressive"
        return (
            f"[MOCK DEMO | {tone_hint}, temp={temperature:.1f}] "
            f"Received prompt ({len(user_prompt)} chars): \"{user_prompt}\""
        )


def _mock_grounded_business_reply(
    profile: BusinessProfile,
    user_prompt: str,
    history: list[ConversationTurn],
    temperature: float,
) -> str:
    """Deterministic grounded mock responder over a BusinessProfile's JSON knowledge base."""
    kb = profile.knowledge_base
    query = user_prompt.lower()

    # Check if the query is an ambiguous follow-up relying on prior history (Task 4)
    pronoun_followup = bool(
        re.search(r"\b(it|its|that|this|they|them|those|same one|first one|what did i)\b", query)
    )
    has_explicit_item = any(
        token in query
        for token in (
            "hoodie",
            "tee",
            "cargo",
            "bootcamp",
            "python",
            "full-stack",
            "data analytics",
        )
    )

    resolved_subject_text = ""
    if pronoun_followup and not has_explicit_item:
        if not history:
            return (
                f"[MOCK DEMO | {profile.business_name} | Stateless] "
                "I do not have previous conversation context, so I cannot tell what item or topic "
                "you are referring to. Could you please specify the exact product, course, or policy?"
            )
        # Inspect prior user/model turns to resolve context
        prior_text = " ".join(turn.text for turn in history).lower()
        resolved_subject_text = prior_text

    search_text = f"{query} {resolved_subject_text}".strip()

    if "what did i" in query and "ask" in query:
        prior_users = [t.text for t in history if t.role == "user"]
        if not prior_users:
            return (
                f"[MOCK DEMO | {profile.business_name} | Stateless] "
                "I have no record of any earlier question because each API call is stateless."
            )
        return (
            f"[MOCK DEMO | {profile.business_name} | With History] "
            f"Earlier in this conversation, you asked: \"{prior_users[-1]}\""
        )

    # Explicit guard against unsupported features in user queries (Task 3 hallucination test)
    unsupported_terms = (
        "drone",
        "lifetime warranty",
        "crypto",
        "bitcoin",
        "hostel",
        "free laptop",
        "international express",
        "custom tailoring",
        "job guarantee",
        "100% placement",
    )
    if any(term in query for term in unsupported_terms):
        contact = kb.get("contact", "our official support channel")
        return (
            f"[MOCK DEMO | {profile.business_name}] I do not have enough information in the "
            f"provided {profile.business_name} knowledge base to confirm that. I cannot guess or "
            f"invent unsupported policies, pricing, or timelines. Please contact {contact}."
        )

    # Search catalog items (ignore generic brand tokens so queries match specific products/courses)
    brand_tokens = set(re.split(r"[^a-z0-9]+", profile.business_name.lower()))
    catalog = kb.get("catalog", [])
    matched_items: list[dict[str, Any]] = []
    for item in catalog:
        if not isinstance(item, dict):
            continue
        item_name = str(item.get("item", "")).lower()
        item_tokens = [
            tok
            for tok in re.split(r"[^a-z0-9]+", item_name)
            if len(tok) > 2 and tok not in brand_tokens
        ]
        if any(tok in search_text for tok in item_tokens):
            matched_items.append(item)

    policies = kb.get("policies", {})
    policy_keywords = {
        "shipping_and_delivery": ("ship", "shipping", "delivery", "deliver", "express", "free"),
        "returns_and_exchanges": ("return", "exchange", "unworn", "tags"),
        "order_tracking": ("track", "tracking", "order id", "status"),
        "payment_methods": ("pay", "payment", "upi", "cod", "cash on delivery", "card"),
        "enrollment": ("enroll", "admission", "join", "seat", "register"),
        "refund_policy": ("refund", "money back", "cancel"),
        "certification": ("certificate", "certification", "capstone", "attendance"),
        "class_recordings": ("recording", "recorded", "missed class", "portal"),
    }

    stop_tokens = {"and", "for", "the", "with"}
    matched_policies: list[str] = []
    for policy_key, policy_text in policies.items():
        key_tokens = [
            tok for tok in policy_key.lower().split("_") if len(tok) > 2 and tok not in stop_tokens
        ]
        known_keywords = policy_keywords.get(policy_key, ())
        if any(kw in query for kw in (*known_keywords, *key_tokens)):
            matched_policies.append(f"{policy_key.replace('_', ' ').title()}: {policy_text}")

    if matched_items:
        lines = [f"[MOCK DEMO | {profile.business_name}] Here are the verified details:"]
        for item in matched_items:
            details = "; ".join(
                f"{k.replace('_', ' ')}: {', '.join(v) if isinstance(v, list) else v}"
                for k, v in item.items()
            )
            lines.append(f"- {details}")
        for policy_line in matched_policies:
            lines.append(f"- {policy_line}")
        return "\n".join(lines)

    if matched_policies:
        intro = (
            f"[MOCK DEMO | {profile.business_name}] Verified business information:"
            if temperature < 0.5
            else f"[MOCK DEMO | {profile.business_name} | temp={temperature:.1f}] Happy to help! Here is what our official knowledge base states:"
        )
        sections: list[str] = []
        if any(kw in search_text for kw in ("product", "catalog", "course", "offer", "service")):
            item_names = [str(item.get("item", "")) for item in catalog if isinstance(item, dict)]
            sections.append(f"Featured Offerings: {', '.join(item_names)}")
        sections.extend(matched_policies)
        return intro + "\n- " + "\n- ".join(sections)

    if any(kw in search_text for kw in ("product", "catalog", "course", "offer", "service", "sell", "teach")):
        item_names = [str(item.get("item", "")) for item in catalog if isinstance(item, dict)]
        return (
            f"[MOCK DEMO | {profile.business_name}] {profile.description} "
            f"Services: {', '.join(profile.services)}. "
            f"Featured offerings: {', '.join(item_names)}."
        )

    if any(kw in search_text for kw in ("hour", "time", "open", "contact", "email", "support")):
        return (
            f"[MOCK DEMO | {profile.business_name}] Support hours: {kb.get('support_hours', 'N/A')}. "
            f"Contact: {kb.get('contact', 'N/A')}."
        )

    contact = kb.get("contact", "our support team")
    return (
        f"[MOCK DEMO | {profile.business_name}] I do not have enough information in the provided "
        f"{profile.business_name} knowledge base to answer \"{user_prompt}\" reliably. "
        f"Please reach out to {contact} for verified assistance."
    )


# ---------------------------------------------------------------------------
# Task 6: Reusable Universal Business AI Agent Core
# ---------------------------------------------------------------------------


@dataclass
class UniversalBusinessAgent:
    """Single reusable AI agent core configured dynamically via BusinessProfile."""

    profile: BusinessProfile
    backend: GeminiAgentBackend
    system_instruction: str = field(init=False)

    def __post_init__(self) -> None:
        self.system_instruction = build_grounded_system_instruction(self.profile)

    def switch_profile(self, new_profile: BusinessProfile) -> None:
        """Switch business configuration at runtime without changing agent code."""
        self.profile = new_profile
        self.system_instruction = build_grounded_system_instruction(new_profile)

    def respond(
        self,
        user_query: str,
        history: list[ConversationTurn] | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Generate a grounded response for the active business profile."""
        return self.backend.generate(
            user_prompt=user_query,
            system_instruction=self.system_instruction,
            temperature=temperature,
            history=history,
            profile=self.profile,
        )


# ---------------------------------------------------------------------------
# Task 4: Explicit Stateless vs Conversation-Aware Interaction Functions
# ---------------------------------------------------------------------------


def query_stateless(
    agent: UniversalBusinessAgent,
    user_query: str,
    temperature: float = 0.2,
) -> LLMResult:
    """Task 4a: Process a query completely independently without any prior messages."""
    cleaned_query = validate_non_empty_text(user_query, "user_query")
    return agent.respond(user_query=cleaned_query, history=None, temperature=temperature)


def query_with_history(
    agent: UniversalBusinessAgent,
    user_query: str,
    history: list[ConversationTurn],
    temperature: float = 0.2,
) -> tuple[LLMResult, list[ConversationTurn]]:
    """Task 4b: Process a query with prior conversation turns and return updated history."""
    cleaned_query = validate_non_empty_text(user_query, "user_query")
    if not isinstance(history, list):
        raise ValidationError("'history' must be a list of ConversationTurn items.")

    result = agent.respond(
        user_query=cleaned_query,
        history=history,
        temperature=temperature,
    )
    updated_history = [
        *history,
        ConversationTurn(role="user", text=cleaned_query),
        ConversationTurn(role="model", text=result.text),
    ]
    return result, updated_history


# ---------------------------------------------------------------------------
# Task 5: Token Usage Formatting & Reporting
# ---------------------------------------------------------------------------


def format_token_report(result: LLMResult, label: str = "Interaction") -> str:
    """Format local estimates separately from API-reported usage_metadata."""
    m = result.token_metrics
    api_status = (
        f"prompt={m.api_prompt_tokens}, candidates={m.api_candidates_tokens}, total={m.api_total_tokens}"
        if not result.is_mock and m.api_total_tokens is not None
        else "Unavailable in mock/demo mode (no live API call made; not fabricated)"
    )
    if not result.is_mock and m.api_thoughts_tokens is not None:
        api_status += f", thoughts={m.api_thoughts_tokens}"

    return (
        f"[{label} | Model: {result.model}]\n"
        f"  - Input length (ESTIMATED): {m.prompt_chars} chars, {m.prompt_words} words, "
        f"~{m.estimated_prompt_tokens} tokens\n"
        f"  - Response length (ESTIMATED): {m.response_chars} chars, {m.response_words} words, "
        f"~{m.estimated_response_tokens} tokens\n"
        f"  - API usage_metadata (ACTUAL): {api_status}"
    )


# ---------------------------------------------------------------------------
# Task 1 - Task 6 Executable Demonstrations
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def demonstrate_llm_fundamentals(backend: GeminiAgentBackend) -> None:
    """Task 1: Demonstrate LLM behavior, prompt specificity, temperature, and roles."""
    _print_header("TASK 1: LLM Fundamentals (Prompts, Temperature, System vs User)")

    print("Concept: An LLM predicts the next most likely token conditioned on context.")
    print("\n1A. How Prompt Specificity Shapes Output:")
    vague = backend.generate(
        user_prompt="How does a large language model work?",
        temperature=0.2,
    )
    structured = backend.generate(
        user_prompt="How does a large language model work? Explain in 3 concise bullet points.",
        temperature=0.2,
    )
    print(f"Vague Prompt Response:\n{vague.text}\n")
    print(f"Structured Prompt Response:\n{structured.text}\n")

    print("1B. Effect of Temperature (0.0 Deterministic vs 0.9 Creative):")
    creative_prompt = "Write 2 short taglines for a streetwear hoodie brand."
    low_temp = backend.generate(user_prompt=creative_prompt, temperature=0.0)
    high_temp = backend.generate(user_prompt=creative_prompt, temperature=0.9)
    print(f"Temperature = 0.0:\n{low_temp.text}\n")
    print(f"Temperature = 0.9:\n{high_temp.text}\n")

    print("1C. System Instructions vs User Prompt:")
    same_user_query = "What is your return policy?"
    no_system = backend.generate(user_prompt=same_user_query, system_instruction="", temperature=0.2)
    with_system = backend.generate(
        user_prompt=same_user_query,
        system_instruction=(
            "You are the Dripzoid support assistant. Only state verified facts: "
            "Unworn items with original tags can be returned or exchanged within 7 days of delivery."
        ),
        temperature=0.2,
    )
    print(f"Without System Instruction:\n{no_system.text}\n")
    print(f"With Dripzoid System Instruction:\n{with_system.text}")


def demonstrate_prompt_engineering(
    backend: GeminiAgentBackend,
    profile: BusinessProfile,
) -> None:
    """Task 2: Demonstrate the reusable parameterized business prompt template."""
    _print_header("TASK 2: Reusable Business Prompt Template")

    sample_query = "What services do you provide and how can you help me today?"
    rendered = build_business_prompt_template(
        business_name=profile.business_name,
        business_industry=profile.industry,
        business_description=profile.description,
        business_tone=profile.tone,
        user_query=sample_query,
    )

    print("Rendered System Instruction:")
    print(rendered.system_instruction)
    print("\nRendered User Prompt:")
    print(rendered.user_prompt)

    result = backend.generate(
        user_prompt=rendered.user_prompt,
        system_instruction=rendered.system_instruction,
        temperature=0.3,
        profile=profile,
    )
    print(f"\nGenerated Response:\n{result.text}")


def demonstrate_grounding(
    backend: GeminiAgentBackend,
    profile: BusinessProfile,
) -> None:
    """Task 3: Demonstrate grounded answering and explicit refusal on missing facts."""
    _print_header(f"TASK 3: Grounding & Hallucination Prevention ({profile.business_name})")

    agent = UniversalBusinessAgent(profile=profile, backend=backend)

    supported_query = (
        "How long does standard delivery take, and what is your return policy?"
        if profile.business_id == "dripzoid"
        else "How much is the Python & AI Agent Foundations Bootcamp, and what is the refund policy?"
    )
    unsupported_query = (
        "Do you offer 30-minute drone delivery and a lifetime warranty on hoodies?"
        if profile.business_id == "dripzoid"
        else "Do you provide free laptops and a 100% job guarantee with hostel accommodation?"
    )

    print(f"[Supported Query] {supported_query}")
    supported_result = query_stateless(agent, supported_query, temperature=0.1)
    print(f"Response:\n{supported_result.text}\n")

    print(f"[Unsupported / Out-of-Scope Query] {unsupported_query}")
    unsupported_result = query_stateless(agent, unsupported_query, temperature=0.1)
    print(f"Response:\n{unsupported_result.text}")


def demonstrate_stateless_vs_conversation(
    backend: GeminiAgentBackend,
    profile: BusinessProfile,
) -> None:
    """Task 4: Compare independent stateless calls against history-aware multi-turn calls."""
    _print_header(f"TASK 4: Stateless API vs Conversation History ({profile.business_name})")

    agent = UniversalBusinessAgent(profile=profile, backend=backend)

    turn_1 = (
        "Tell me about the Dripzoid Oversized Graphic Hoodie."
        if profile.business_id == "dripzoid"
        else "Tell me about the Python & AI Agent Foundations Bootcamp."
    )
    turn_2 = (
        "Does it qualify for free standard shipping?"
        if profile.business_id == "dripzoid"
        else "What are its prerequisites and duration?"
    )

    print("--- 4A. Stateless Interaction (no shared history) ---")
    s1 = query_stateless(agent, turn_1)
    print(f"Turn 1 User: {turn_1}\nTurn 1 Agent: {s1.text}\n")
    s2 = query_stateless(agent, turn_2)
    print(f"Turn 2 User: {turn_2}\nTurn 2 Agent (no memory): {s2.text}\n")

    print("--- 4B. Conversation-Aware Interaction (passing prior turns) ---")
    history: list[ConversationTurn] = []
    c1, history = query_with_history(agent, turn_1, history)
    print(f"Turn 1 User: {turn_1}\nTurn 1 Agent: {c1.text}\n")
    c2, history = query_with_history(agent, turn_2, history)
    print(f"Turn 2 User: {turn_2}\nTurn 2 Agent (with history): {c2.text}")


def demonstrate_token_usage(
    backend: GeminiAgentBackend,
    profile: BusinessProfile,
) -> None:
    """Task 5: Measure input/output lengths, report usage_metadata, and explain impact."""
    _print_header("TASK 5: Token Usage, Cost & Context Window Awareness")

    agent = UniversalBusinessAgent(profile=profile, backend=backend)

    short_query = "What are your support hours?"
    short_result = query_stateless(agent, short_query, temperature=0.2)

    # Simulate a multi-turn history to show why conversation length grows token usage
    _, history = query_with_history(
        agent,
        "List all items in your catalog with prices and details.",
        [],
    )
    longer_result, _ = query_with_history(
        agent,
        "Summarize your catalog, payment or refund rules, and support hours in detail.",
        history,
    )

    print(format_token_report(short_result, label="Short Stateless Query"))
    print()
    print(format_token_report(longer_result, label="Multi-Turn Grounded Query"))
    print(
        "\nWhy Token Usage Matters:\n"
        "1. Cost: LLM APIs bill per input token (system prompt + knowledge base + chat history + "
        "user query) and per output token generated.\n"
        "2. Context Limits: Every model has a finite context window. Passing huge unpruned histories "
        "or massive knowledge bases increases latency, cost, and risk of hitting token limits."
    )


def demonstrate_profile_switching(
    backend: GeminiAgentBackend,
    profiles: dict[str, BusinessProfile],
) -> None:
    """Task 6: Run the exact same UniversalBusinessAgent core across multiple businesses."""
    _print_header("TASK 6: Configurable Business Profiles (One Core, Multiple Industries)")

    shared_question = "What offerings do you have and what are your return or refund rules?"

    first_id = next(iter(profiles))
    agent = UniversalBusinessAgent(profile=profiles[first_id], backend=backend)

    for profile_id, profile in profiles.items():
        agent.switch_profile(profile)
        result = query_stateless(agent, shared_question, temperature=0.2)
        print(f"\n--- Active Profile: {profile.business_name} ({profile.industry}) ---")
        print(f"Tone: {profile.tone}")
        print(f"Query: {shared_question}")
        print(f"Response:\n{result.text}")


def run_interactive_mode(
    backend: GeminiAgentBackend,
    profiles: dict[str, BusinessProfile],
    initial_profile_id: str,
) -> None:
    """Interactive terminal chat loop demonstrating profile switching and memory."""
    _print_header(f"Interactive Universal Business AI Agent ({backend.mode_reason})")
    agent = UniversalBusinessAgent(profile=profiles[initial_profile_id], backend=backend)
    history: list[ConversationTurn] = []

    print(f"Active profile: {agent.profile.business_name} ({agent.profile.industry})")
    print("Commands: /switch <profile_id>, /profiles, /reset, /exit")

    while True:
        try:
            user_input = input(f"\n[{agent.profile.business_id}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive session.")
            break

        if not user_input:
            print("Please enter a message or command (/exit to quit).")
            continue
        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "/profiles":
            print("Available profiles:", ", ".join(sorted(profiles)))
            continue
        if user_input.lower() == "/reset":
            history.clear()
            print("Conversation history cleared.")
            continue
        if user_input.lower().startswith("/switch"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"Usage: /switch <profile_id>. Available: {', '.join(sorted(profiles))}")
                continue
            target = parts[1].strip().lower()
            if target not in profiles:
                print(f"Unknown profile '{target}'. Available: {', '.join(sorted(profiles))}")
                continue
            agent.switch_profile(profiles[target])
            history.clear()
            print(f"Switched to {agent.profile.business_name} ({agent.profile.industry}). History reset.")
            continue

        try:
            result, history = query_with_history(agent, user_input, history)
            print(f"\nAgent: {result.text}")
            print(format_token_report(result, label="Turn Metrics"))
        except AgentError as exc:
            print(f"Error: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for running Day 1 tasks, single queries, or interactive chat."""
    parser = argparse.ArgumentParser(
        description="Day 1: Universal Business AI Agent (SAI Labs)"
    )
    parser.add_argument(
        "--task",
        choices=["all", "1", "2", "3", "4", "5", "6"],
        default="all",
        help="Run all tasks or a single task (1-6). Default: all.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Business profile ID to use (e.g., dripzoid, edusmart).",
    )
    parser.add_argument(
        "--profiles-file",
        type=Path,
        default=DEFAULT_PROFILES_PATH,
        help="Path to business_profiles.json.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a single grounded query against the selected profile and exit.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature in [0.0, 2.0] for --query mode.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force deterministic [MOCK DEMO] mode without calling the Gemini API.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Require live Gemini API mode (fail if GEMINI_API_KEY or SDK is missing).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive conversation-aware CLI session.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for Day 1 execution."""
    load_environment()
    args = parse_args(argv)

    try:
        profiles, default_profile_id = load_business_profiles(args.profiles_file)
        selected_id = (args.profile or default_profile_id).strip().lower()
        if selected_id not in profiles:
            raise ValidationError(
                f"Unknown profile '{selected_id}'. Available profiles: {', '.join(sorted(profiles))}"
            )

        backend = GeminiAgentBackend(force_demo=args.demo, require_live=args.live)
        active_profile = profiles[selected_id]

        print("Day 1 - Universal Business AI Agent | SAI Labs")
        print(f"Execution Mode: {backend.mode_reason}")
        print(f"Loaded Profiles: {', '.join(sorted(profiles))} (active: {selected_id})")

        if args.query is not None:
            agent = UniversalBusinessAgent(profile=active_profile, backend=backend)
            result = query_stateless(agent, args.query, temperature=args.temperature)
            print(f"\n[{active_profile.business_name}] Query: {args.query}")
            print(f"Response:\n{result.text}\n")
            print(format_token_report(result, label="Single Query"))
            return 0

        if args.interactive:
            run_interactive_mode(backend, profiles, selected_id)
            return 0

        task_runners = {
            "1": lambda: demonstrate_llm_fundamentals(backend),
            "2": lambda: demonstrate_prompt_engineering(backend, active_profile),
            "3": lambda: demonstrate_grounding(backend, active_profile),
            "4": lambda: demonstrate_stateless_vs_conversation(backend, active_profile),
            "5": lambda: demonstrate_token_usage(backend, active_profile),
            "6": lambda: demonstrate_profile_switching(backend, profiles),
        }

        if args.task == "all":
            for task_id in ("1", "2", "3", "4", "5", "6"):
                task_runners[task_id]()
        else:
            task_runners[args.task]()

        return 0
    except AgentError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
