"""Day 3 scaffold: agent architecture and evaluation.

The evaluation harness should be implemented after the agent interface is defined.
"""
from pathlib import Path
import csv


def load_test_cases(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    test_file = Path(__file__).with_name("test.csv")
    cases = load_test_cases(test_file)
    print(f"Loaded {len(cases)} starter test cases.")
    print("Implement the agent and evaluation assertions described in README.md.")


if __name__ == "__main__":
    main()
