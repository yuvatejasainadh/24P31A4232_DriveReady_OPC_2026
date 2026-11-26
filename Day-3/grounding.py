"""Tasks 4 & 8: Lightweight Knowledge Retrieval, Grounding, and Safety Controls.

Reuses Day 1 + Day 2 BusinessContext objects with strict business isolation,
keyword-based retrieval, prompt-injection / cross-customer data protection, and
deterministic grounding verification checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys
from typing import Any

DAY2_DIR = Path(__file__).resolve().parent.parent / "Day-2"
if str(DAY2_DIR) not in sys.path:
    sys.path.append(str(DAY2_DIR))

from business_config import BusinessContext, validate_non_empty_text  # noqa: E402


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "tell",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "you",
    "your",
}

_PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|business|system)\s+instructions",
    r"bypass\s+(?:all\s+)?(?:rules|guardrails|safety|constraints)",
    r"reveal\s+(?:your\s+)?system\s+prompt",
    r"override\s+(?:business|system)\s+rules",
    r"pretend\s+to\s+be\s+an?\s+unrestricted",
)

_CROSS_CUSTOMER_PATTERNS = (
    r"(?:another|other|someone\s+else'?s|different)\s+(?:customer|user|student|person)'?s?\s+(?:order|account|history|session|data|email)",
    r"show\s+me\s+all\s+customers",
    r"access\s+(?:another|other)\s+session",
)

_UNSUPPORTED_CLAIM_TERMS = (
    "drone delivery",
    "30-minute",
    "lifetime warranty",
    "free laptop",
    "100% job guarantee",
    "hostel accommodation",
    "bitcoin",
    "cryptocurrency",
    "international express",
)


@dataclass(frozen=True)
class KnowledgeSnippet:
    """Single retrieved fact or policy snippet from the active BusinessContext."""

    source_type: str  # "faq", "policy", "catalog", "profile"
    title: str
    content: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    """Grounded knowledge retrieval output scoped strictly to one business."""

    business_id: str
    business_name: str
    query: str
    snippets: list[KnowledgeSnippet]
    has_grounding_facts: bool

    def to_context_text(self) -> str:
        if not self.snippets:
            return (
                f"No direct FAQ/policy/catalog match found in {self.business_name} knowledge base. "
                "Do not invent facts."
            )
        lines = [f"Retrieved {self.business_name} Knowledge:"]
        for item in self.snippets:
            lines.append(f"- [{item.source_type}] {item.title}: {item.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "business_id": self.business_id,
            "business_name": self.business_name,
            "query": self.query,
            "has_grounding_facts": self.has_grounding_facts,
            "snippets": [s.to_dict() for s in self.snippets],
        }


@dataclass(frozen=True)
class SafetyCheckResult:
    """Pre-execution or post-execution safety gate outcome."""

    allowed: bool
    reason_code: str  # "ok", "prompt_injection", "cross_customer_access", "unsupported_claim"
    safe_message: str = ""


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.split(r"[^a-z0-9]+", text.lower()):
        if len(raw) < 2 or raw in _STOP_WORDS:
            continue
        if raw.endswith("ies") and len(raw) > 4:
            tokens.append(raw[:-3] + "y")
        elif raw.endswith("s") and len(raw) > 3 and not raw.endswith("ss"):
            tokens.append(raw[:-1])
        else:
            tokens.append(raw)
    return tokens


class KnowledgeRetriever:
    """Transparent keyword-based retriever scoped strictly to one BusinessContext."""

    def retrieve(
        self,
        query: str,
        context: BusinessContext,
        top_k: int = 5,
    ) -> RetrievalResult:
        cleaned = validate_non_empty_text(query, "query")
        query_tokens = set(_tokenize(cleaned))
        candidates: list[KnowledgeSnippet] = []

        # 1. Search FAQs
        for faq in context.faqs:
            q_text = str(faq.get("question", ""))
            a_text = str(faq.get("answer", ""))
            kws = " ".join(str(k) for k in faq.get("keywords", []))
            doc_tokens = set(_tokenize(f"{q_text} {a_text} {kws}"))
            overlap = len(query_tokens & doc_tokens)
            if overlap > 0:
                candidates.append(
                    KnowledgeSnippet(
                        source_type="faq",
                        title=q_text,
                        content=a_text,
                        score=overlap * 3,
                    )
                )

        # 2. Search official policies
        for policy_key, policy_text in context.policies.items():
            title = policy_key.replace("_", " ").title()
            doc_tokens = set(_tokenize(f"{title} {policy_text}"))
            overlap = len(query_tokens & doc_tokens)
            # Treat "refund" as matching "return/exchange" policy in e-commerce
            if "refund" in query_tokens and "return" in doc_tokens:
                overlap += 1
            if overlap > 0:
                candidates.append(
                    KnowledgeSnippet(
                        source_type="policy",
                        title=title,
                        content=policy_text,
                        score=overlap * 3,
                    )
                )

        # 3. Search catalog items / courses
        brand_tokens = set(_tokenize(context.business_name))
        for item in context.catalog:
            item_title = str(item.get("item", ""))
            details = "; ".join(
                f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                for k, v in item.items()
                if k != "numeric_price"
            )
            doc_tokens = set(_tokenize(details)) - brand_tokens
            overlap = len(query_tokens & doc_tokens)
            if overlap > 0:
                candidates.append(
                    KnowledgeSnippet(
                        source_type="catalog",
                        title=item_title,
                        content=details,
                        score=overlap * 2,
                    )
                )

        # 4. Support hours & contact metadata
        if query_tokens & {"hour", "open", "timing", "support", "contact", "email", "reach"}:
            candidates.append(
                KnowledgeSnippet(
                    source_type="profile",
                    title="Support Hours & Contact",
                    content=f"Hours: {context.support_hours}; Contact: {context.contact}",
                    score=4,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        top_snippets = candidates[:top_k]

        return RetrievalResult(
            business_id=context.business_id,
            business_name=context.business_name,
            query=cleaned,
            snippets=top_snippets,
            has_grounding_facts=len(top_snippets) > 0,
        )


class SafetyGuard:
    """Enforces grounding boundaries, session isolation, and injection resistance."""

    def check_query(self, query: str, context: BusinessContext) -> SafetyCheckResult:
        """Screen incoming queries for instruction override, cross-user access, or unsupported claims."""
        lower = query.lower().strip()

        for pattern in _PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lower):
                return SafetyCheckResult(
                    allowed=False,
                    reason_code="prompt_injection",
                    safe_message=(
                        f"[SAFE GUARD | {context.business_name}] I cannot ignore or override "
                        f"verified {context.business_name} business rules. I can only answer "
                        "questions grounded in approved business data and tools."
                    ),
                )

        for pattern in _CROSS_CUSTOMER_PATTERNS:
            if re.search(pattern, lower):
                return SafetyCheckResult(
                    allowed=False,
                    reason_code="cross_customer_access",
                    safe_message=(
                        f"[SAFE GUARD | {context.business_name}] For privacy and session isolation, "
                        "I cannot access another customer's order, account, or conversation history. "
                        "Please provide only your own order ID."
                    ),
                )

        if any(term in lower for term in _UNSUPPORTED_CLAIM_TERMS):
            return SafetyCheckResult(
                allowed=False,
                reason_code="unsupported_claim",
                safe_message=(
                    f"[SAFE GUARD | {context.business_name}] I do not have information confirming "
                    f"that feature or policy in the verified {context.business_name} knowledge base. "
                    f"Please contact {context.contact} for official confirmation."
                ),
            )

        return SafetyCheckResult(allowed=True, reason_code="ok")

    def verify_grounding_compliance(
        self,
        response_text: str,
        context: BusinessContext,
        expected_response_type: str = "",
    ) -> bool:
        """Deterministic grounding compliance check for automated evaluation."""
        lower = response_text.lower()
        # Never claim real booking or refund execution occurred in this mock-safe version
        unsafe_claims = (
            "your appointment is booked",
            "refund has been processed",
            "money has been transferred",
        )
        if any(claim in lower for claim in unsafe_claims):
            return False

        if expected_response_type in {"refusal", "clarification"}:
            markers = (
                "do not have",
                "cannot",
                "not sure",
                "not found",
                "no order record",
                "unable",
                "please share",
                "contact",
                "privacy",
                "could not find",
            )
            return any(marker in lower for marker in markers)

        if expected_response_type == "not_found":
            return any(m in lower for m in ("not found", "no order record", "verify your order"))

        # Ensure business isolation when business name appears
        other_name = "edusmart" if context.business_id == "dripzoid" else "dripzoid"
        if other_name in lower and context.business_id not in lower:
            return False

        return bool(response_text.strip())
