# Triage reasoning

`reasoning.py` sends prompts to the configured local Ollama service, parses
agent actions, applies scenario checks, and produces a deterministic fallback
when the service cannot be reached.

The output is a software-simulation result only. It is not a validated
clinical decision, and the nurse dashboard that would receive routed cases is
not implemented.
