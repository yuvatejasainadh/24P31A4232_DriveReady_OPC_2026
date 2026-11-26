"""Day 1 scaffold: LLM foundations and business context.

Set GEMINI_API_KEY in your environment before implementing API exercises.
"""
import os


def check_configuration() -> bool:
    """Check whether the Gemini API key is available without printing it."""
    configured = bool(os.getenv("GEMINI_API_KEY"))
    print(f"Gemini API key configured: {'yes' if configured else 'no'}")
    return configured


def main() -> None:
    print("Day 1 - LLM Foundations & Business Context")
    check_configuration()
    print("Complete the Day-1 exercises and record your observations in README.md.")


if __name__ == "__main__":
    main()
