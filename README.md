# Agentic AI Loop

A production-grade agentic retry loop built on [smolagents](https://github.com/huggingface/smolagents). Runs a local LLM (Qwen3 via llama.cpp) through a structured pipeline of perception, planning, memory, multi-agent collaboration, and self-evaluation — retrying intelligently until the output meets quality requirements or the budget is exhausted.

---

## How it works

Every task passes through 13 stages, split across a pre-loop setup and a retry loop:

```
Task input
    │
    ├─ 1. Perception      — classify task type, complexity, intent, keywords
    ├─ 2. Memory          — retrieve semantically similar past interactions
    ├─ 3. Planning        — build a structured plan using perception + memory
    ├─ 4. Compaction      — summarise old conversation turns if context is growing large
    ├─ 5. Checkpoint      — resume from a prior crash, or start fresh
    │
    └─► RETRY LOOP (up to max_retries+1 attempts, scaled by complexity)
            │
            ├─ Build enriched prompt (plan + stage guidance + stall guidance + history)
            ├─ [if near token budget] Compact agent.memory.steps
            ├─ _execute_attempt()
            │       ├─ A2A path: executor → critic → (result, score, issues)
            │       └─ Direct path: agent.run() → result
            │       StallError caught here → hard failure + recovery guidance
            │       Unexpected exception → emergency memory save → exit
            ├─ Merge stall guidance (hard-stall issues + soft-detector warnings, both)
            ├─ 7. Verification   — structural check (did agent actually do the task?)
            ├─ 8. Evaluation     — quality score (is the output good?)
            ├─ Track best result seen across all attempts
            ├─ 9. Checkpoint save
            ├─ 10. Stage advance
            │
            ├─ PASS → break
            └─ FAIL →
                    hard failure (stall/budget) → keep plan, inject recovery guidance
                    soft failure (low score)    → fresh memory retrieval + replan
    │
    ├─ 11. Synthesize     — polish best result via A2A synthesizer
    ├─ 12. Memory store   — save interaction for future retrieval
    └─ 13. Output         — render result with score, timing, memory stats
```

### Hard vs soft failure

The loop distinguishes two failure modes and handles them differently:

| | Hard failure | Soft failure |
|---|---|---|
| **Trigger** | `StallError` (token budget / infinite loop) | Critic FAIL or low evaluation score |
| **Recovery** | Keep existing plan, inject guidance about *why* it got stuck | Discard plan, re-query memory, replan from scratch |
| **Rationale** | Agent got stuck doing something — guide it out | Agent was trying the wrong approach — change strategy |

---

## Project structure

```
smolagents/
├── agent.py              # Entry point — full pipeline + retry loop
├── config.json           # All tuneable parameters
├── knowledge_graph.json  # Tool/concept/pattern registry for semantic lookup
├── skills-lock.json
│
├── layers/               # Pipeline stages (each is a standalone class)
│   ├── perception.py     # Task classification via LLM + regex patterns
│   ├── planning.py       # Structured plan generation
│   ├── memory.py         # Semantic long-term + short-term memory (MiniLM embeddings)
│   ├── evaluation.py     # Quality scoring + retry prompt generation
│   ├── verification.py   # Structural output verification
│   ├── compaction.py     # Conversation history summarisation
│   ├── checkpoint.py     # Crash recovery (save/load/resume)
│   ├── token_budget.py   # Token counter + circuit breaker (StallError at limit)
│   ├── guards.py         # Stall detection (error streaks, parse loops, write thrash)
│   ├── stages.py         # Stage pipeline (draft → refine → finalise)
│   └── output.py         # Result rendering
│
└── a2a/                  # Agent-to-Agent collaboration layer
    ├── protocol.py       # AgentHub message routing
    └── agents.py         # Executor, critic (N-sample self-consistency), synthesizer
```

---

## Setup

**Requirements:** Python 3.9+, a llama.cpp server running locally.

```bash
pip install smolagents litellm sentence-transformers numpy pyyaml
```

The memory and knowledge graph layers use `all-MiniLM-L6-v2` for semantic embeddings. It downloads automatically on first run via `sentence-transformers`.

---

## Configuration

All tuneable parameters live in `config.json`:

```jsonc
{
  "model": {
    "provider": "llmacpp",        // LiteLLM provider alias
    "name": "qwen3",
    "api_base": "http://...:8000/v1",
    "api_key": "not-needed"
  },
  "agent": {
    "max_steps": 10,              // Max smolagents steps per attempt
    "stream_outputs": true
  },
  "evaluation": {
    "min_pass_score": 0.5,        // Score threshold to accept a result
    "max_retries": 2              // Fallback if complexity not detected
  },
  "evaluation_complexity": {
    "max_retries_simple": 1,      // Simple tasks: 2 total attempts
    "max_retries_moderate": 2,    // Moderate: 3 total
    "max_retries_complex": 4      // Complex: 5 total
  },
  "critic": {
    "n_samples": 2,               // Number of critic samples per attempt
    "temperature": 0.7,           // Higher = more variance between samples
    "pass_avg": 3.0               // Minimum average score to pass
  },
  "token_budget": {
    "max_tokens": 32000,          // Hard token limit per attempt
    "warn_threshold": 0.8         // Log warning at 80%, compact at limit
  },
  "memory": {
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k_retrieval": 3,
    "min_score_to_store": 0.6
  }
}
```

---

## Running

```bash
python agent.py
```

```
Agentic AI ready. Type your task, 'quit' to exit.
Commands: /new (reset session), /history (show turns), /help
```

**Session commands:**

| Command | Effect |
|---|---|
| `/new` | Clear conversation history, memory, checkpoints — start fresh |
| `/history` | Show recent conversation turns |
| `/help` | Show command list |
| `quit` / `exit` | Exit |

---

## Key design properties

**Checkpoint/resume** — every attempt is saved to `.agent/progress.md`. If the process crashes or is interrupted mid-task, the next run resumes from where it left off, restoring the best result seen so far, the stage, and any stall guidance.

**Token budget circuit breaker** — `TokenBudgetLayer` accumulates token usage as a smolagents step callback. At 80% of budget it logs a warning; at 100% it raises `StallError`, which is caught by `_execute_attempt` and triggers a hard-failure recovery path rather than crashing.

**Step compaction** — when the agent's internal step memory approaches the token limit, the last 3 steps are summarised and the step list is cleared before the next attempt, preventing context overflow across retries.

**Complexity-adaptive retry budget** — `PerceptionLayer` classifies each task as `simple`, `moderate`, or `complex`. The retry budget scales accordingly: complex tasks get up to 4 retries (5 total attempts) while simple tasks get 1 retry, avoiding wasteful LLM calls on easy questions.

**Semantic knowledge graph** — `knowledge_graph.json` maps tools, concepts, and task patterns to usage guidance. The orchestrator agent can call `query_knowledge_graph(query)` to look up which sub-agent or tool to use. Uses the same MiniLM model as memory retrieval for zero-overhead embedding reuse.

**A2A multi-agent collaboration** — three roles registered on an `AgentHub`:
- **Executor** — wraps the main `CodeAgent`, handles tool calls and sub-agent delegation
- **Critic** — scores results on a rubric using N-sample self-consistency at `temperature=0.7` to generate genuine variance between samples
- **Synthesizer** — polishes the best result using the critique tied to that specific attempt (not the last attempt's critique)

---

## Extending

**Add a new layer:** implement a class with the relevant `process()` / `verify()` / `evaluate()` method, add it to `layers/__init__.py`, wire it into `run()` in `agent.py`.

**Add a new sub-agent:** register it with `hub` in `agent.py` and add it to `knowledge_graph.json` under `"tools"` so the orchestrator can discover it semantically.

**Disable A2A:** set `"a2a": { "enabled": false }` in `config.json`. The loop falls back to direct `agent.run()` with `evaluation_layer` providing quality scoring.
