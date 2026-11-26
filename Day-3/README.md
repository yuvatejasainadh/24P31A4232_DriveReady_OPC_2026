# Day 3: Agent Architecture, LCEL & Evaluation

> **Universal Business AI Agent** | SAI Labs | Roll No. 24P31A4232

## Objective

Combine the earlier exercises into a reusable agent workflow and evaluate its behavior using repeatable test cases.

## Planned Tasks

1. Build a prompt/model/output-parser chain using LCEL.
2. Design an agent workflow that selects from a controlled set of tools.
3. Separate deterministic business operations from LLM reasoning.
4. Create a CSV-driven evaluation harness.
5. Test grounding, fallback behavior, intent routing, and tool selection.
6. Compare plain code, a single LLM call, and an AI agent for different tasks.

## Evaluation

The starter `test.csv` contains example test cases. Extend it as features are implemented. Record actual pass counts and model settings; do not claim results before running the tests.

## Run

```bash
python task.py
```
