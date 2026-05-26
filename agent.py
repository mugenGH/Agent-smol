import json
import os
import time
import warnings
import yaml
from pathlib import Path
from typing import Optional

# Prevent LiteLLM from fetching remote model cost map on startup (avoids network
# timeout delay and the "Failed to fetch" warning when offline).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# Prevent HuggingFace Hub from making network calls — the embedding model is
# already cached locally so this has no downside.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Suppress botocore/sagemaker noise — those packages aren't installed and AWS
# services are not used. LiteLLM emits these via logging, not warnings.
warnings.filterwarnings("ignore", message=".*could not pre-load.*")
import logging as _logging
_logging.getLogger("LiteLLM").setLevel(_logging.ERROR)

import smolagents as _smolagents
import inspect as _inspect
from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, LiteLLMModel

import numpy as np

from layers import (
    EvaluationLayer, Interaction, MemoryLayer,
    OutputLayer, PerceptionLayer, PlanningLayer,
    StallDetector, StallError,
    CheckpointLayer, Checkpoint,
    StagePipeline,
    CompactionLayer,
    VerificationLayer,
    TokenBudgetLayer,
)
from a2a import AgentHub, A2AMessage, register_executor, register_critic, register_synthesizer


# ── Config ────────────────────────────────────────────────────────────────────

CFG_PATH = Path(__file__).parent / "config.json"
KG_PATH  = Path(__file__).parent / "knowledge_graph.json"

cfg = json.loads(CFG_PATH.read_text())
kg  = json.loads(KG_PATH.read_text())

model_cfg = cfg["model"]
agent_cfg  = cfg["agent"]

_LITELLM_PROVIDER_MAP = {"llmacpp": "openai", "llamacpp": "openai"}
_provider = _LITELLM_PROVIDER_MAP.get(model_cfg["provider"], model_cfg["provider"])
MODEL_ID  = f"{_provider}/{model_cfg['name']}"
API_KEY   = model_cfg.get("api_key", "not-needed")

_raw_base = model_cfg["api_base"].rstrip("/")
for _suffix in ("/chat/completions", "/v1"):
    if _raw_base.endswith(_suffix):
        _raw_base = _raw_base[: -len(_suffix)]
        break
API_BASE  = _raw_base

MAX_STEPS   = agent_cfg["max_steps"]
STREAM      = agent_cfg["stream_outputs"]
MAX_RETRIES = cfg["evaluation"]["max_retries"]
A2A_ENABLED = cfg.get("a2a", {}).get("enabled", True)

# Dynamic retry budget by complexity (Fix 6)
_RETRY_BY_COMPLEXITY = cfg.get("evaluation_complexity", {
    "max_retries_simple": 1,
    "max_retries_moderate": 2,
    "max_retries_complex": 4,
})


def _retries_for(complexity: str) -> int:
    """Pick retry budget by perception complexity, defaulting to config value."""
    key = f"max_retries_{complexity}"
    return _RETRY_BY_COMPLEXITY.get(key, MAX_RETRIES)

# ── Layers ────────────────────────────────────────────────────────────────────

perception_layer  = PerceptionLayer(cfg["perception"])
perception_layer.set_model(MODEL_ID, API_BASE, API_KEY)

planning_layer    = PlanningLayer()
memory_layer      = MemoryLayer(cfg["memory"])
evaluation_layer  = EvaluationLayer(cfg["evaluation"])
output_layer      = OutputLayer()
stall_detector    = StallDetector(cfg["guards"])
token_budget      = TokenBudgetLayer(cfg["token_budget"])
checkpoint_layer  = CheckpointLayer()
stage_pipeline    = StagePipeline(cfg["stages"])
compaction_layer  = CompactionLayer(cfg["compaction"])
compaction_layer.set_model(MODEL_ID, API_BASE, API_KEY)
verification_layer = VerificationLayer(cfg["verification"])

# Conversation history (multi-turn)
conversation_history: list[dict] = []


# ── KG semantic index (built once at startup) ──────────────────────────────────

_kg_entries: list[tuple] = []  # (section, key, value, text)
for _sec in ("tools", "concepts", "task_patterns"):
    for _k, _v in kg[_sec].items():
        _kg_entries.append((_sec, _k, _v, f"{_k} {json.dumps(_v)}"))

# Reuse the memory layer's already-loaded MiniLM model to avoid double-loading
_kg_vecs: Optional[np.ndarray] = None
_kg_embedder = memory_layer._embedder
if _kg_embedder is not None:
    try:
        _kg_vecs = _kg_embedder.encode([e[3] for e in _kg_entries])
    except Exception:
        _kg_vecs = None


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def query_knowledge_graph(query: str) -> str:
    """Look up tools, concepts, or task patterns using semantic similarity.

    Args:
        query: A natural-language description of what you need (e.g. 'search the web').
    """
    if _kg_embedder is not None and _kg_vecs is not None:
        q_vec = _kg_embedder.encode(query)
        norms = np.linalg.norm(_kg_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        sims = (_kg_vecs / norms) @ q_vec / (np.linalg.norm(q_vec) + 1e-9)
        top_idxs = np.argsort(sims)[::-1][:5]
        results = [
            f"[{_kg_entries[i][0]}] {_kg_entries[i][1]}: {json.dumps(_kg_entries[i][2], indent=2)}"
            for i in top_idxs if sims[i] > 0.2
        ]
        return "\n\n".join(results) if results else f"No results found for '{query}'."

    # Keyword fallback when embedder unavailable
    query_lower = query.lower()
    results = []
    for section, key, value, _ in _kg_entries:
        if query_lower in key.lower() or query_lower in json.dumps(value).lower():
            results.append(f"[{section}] {key}: {json.dumps(value, indent=2)}")
    return "\n\n".join(results) if results else f"No results found for '{query}'."


@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path.

    Args:
        path: The file path to read from.
    """
    return Path(path).read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path.

    Args:
        path: The file path to write to.
        content: The text content to write.
    """
    Path(path).write_text(content)
    return f"Written to {path}"


# ── System prompt ─────────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(_inspect.getfile(_smolagents)).parent / "prompts"
_DEFAULT_TEMPLATES = yaml.safe_load((_PROMPTS_DIR / "code_agent.yaml").read_text())


def build_system_prompt() -> str:
    tool_lines = [
        f"  - {name}: {info['description']} (use when: {'; '.join(info['use_when'][:2])})"
        for name, info in kg["tools"].items()
    ]
    pattern_lines = [
        f"  - {name}: {' → '.join(info['steps'])}"
        for name, info in kg["task_patterns"].items()
    ]
    custom = (
        "\n\n---\n"
        "You are an orchestrator AI agent with structured planning and memory.\n"
        "You coordinate specialized sub-agents to complete tasks.\n\n"
        "## Sub-Agents (call these like functions in your code)\n"
        "  - search_agent(task): searches the web — use for any information lookup\n"
        "  - file_agent(task): reads/writes files — use for any file I/O\n\n"
        "## Meta Tools\n" + "\n".join(tool_lines) + "\n\n"
        "## Task Patterns\n" + "\n".join(pattern_lines) + "\n\n"
        "Always decompose the task and delegate to the right sub-agent. "
        "Use query_knowledge_graph when unsure which agent or tool to use. "
        "Be concise but complete."
    )
    return _DEFAULT_TEMPLATES["system_prompt"] + custom


# ── Model & agents ────────────────────────────────────────────────────────────

model = LiteLLMModel(model_id=MODEL_ID, api_base=API_BASE, api_key=API_KEY)

_prompt_templates = {**_DEFAULT_TEMPLATES, "system_prompt": build_system_prompt()}

# Sub-agent: web search
_search_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    name="search_agent",
    description=(
        "Searches the web using DuckDuckGo. "
        "Call with a plain English search query. "
        "Returns a summary of the top results."
    ),
    max_steps=4,
    stream_outputs=STREAM,
)

# Sub-agent: file I/O
_file_agent = CodeAgent(
    tools=[read_file, write_file],
    model=model,
    name="file_agent",
    description=(
        "Reads and writes files on disk. "
        "Call with an instruction like 'read /path/to/file' or "
        "'write <content> to /path/to/file'."
    ),
    max_steps=4,
    stream_outputs=STREAM,
)

# Orchestrator: delegates to sub-agents, retains KG lookup for meta-reasoning
agent = CodeAgent(
    tools=[query_knowledge_graph],
    model=model,
    managed_agents=[_search_agent, _file_agent],
    max_steps=MAX_STEPS,
    stream_outputs=STREAM,
    prompt_templates=_prompt_templates,
    step_callbacks=[stall_detector, token_budget],
)


# ── A2A Hub ───────────────────────────────────────────────────────────────────

hub = AgentHub(reraise=(StallError,))

if A2A_ENABLED:
    register_executor(hub, "executor", agent)
    register_critic(hub, model, api_base=API_BASE, api_key=API_KEY, cfg=cfg.get("critic", {}))
    register_synthesizer(hub, model)


# ── Chitchat short-circuit ────────────────────────────────────────────────────

_CHITCHAT_REPLIES = {
    "hi": "Hi! What can I help you with?",
    "hello": "Hello! What can I help you with?",
    "hey": "Hey! What can I help you with?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "bye": "Goodbye!",
    "goodbye": "Goodbye!",
}


# ── A2A collaborative pipeline ────────────────────────────────────────────────

def _collaborative_run(original_task: str, enriched_task: str) -> tuple[str, dict]:
    """Run executor → critic via A2A and return (raw_result, verdict_dict).

    verdict_dict keys: passed (bool), avg_score (0-1), rubric (dict), issues (str).
    StallError from the executor propagates through the hub (reraise is set)
    and must be caught by the caller exactly as before.
    """
    exec_msg = A2AMessage(sender="loop", recipient="executor", task=enriched_task)
    exec_resp = hub.send(exec_msg)

    critic_msg = A2AMessage(
        sender="loop",
        recipient="critic",
        task=original_task,
        payload={"result": exec_resp.result},
        reply_to=exec_msg.message_id,
    )
    critic_resp = hub.send(critic_msg)

    try:
        verdict: dict = json.loads(critic_resp.result)
    except (json.JSONDecodeError, ValueError, TypeError):
        passed = "VERDICT: PASS" in critic_resp.result
        verdict = {
            "passed": passed,
            "avg_score": 1.0 if passed else 0.0,
            "rubric": {},
            "issues": critic_resp.result[:200],
        }
    return exec_resp.result, verdict


# ── Core pipeline ─────────────────────────────────────────────────────────────

def _execute_attempt(
    enriched_task: str,
    task: str,
    best_result: str,
    attempt: int,
) -> tuple[str, dict, str, str]:
    """Run one executor pass and return (result, critique, failure_type, stall_str).

    StallError (expected budget/loop failure) is caught here and returned as a
    hard failure so the outer loop can retry with stall_guidance injected.
    Unexpected Exception propagates to the caller for emergency cleanup and exit.
    Critic issues are always captured when the critic fails; the outer loop decides
    whether to use them based on whether a retry will actually occur.
    """
    try:
        if A2A_ENABLED:
            raw_result, critique = _collaborative_run(task, enriched_task)
            critic_passed = critique.get("passed", True)
            verdict_str = "PASS" if critic_passed else "FAIL"
            print(f"\n  [A2A critic] {verdict_str} | budget: {token_budget.summary()}")
            stall_str = critique.get("issues", "")[:300] if not critic_passed else ""
            return raw_result, critique, "soft", stall_str
        else:
            result = str(agent.run(enriched_task))
            return result, {}, "soft", ""
    except StallError as e:
        checkpoint_layer.append_progress(f"  STALL: {str(e)[:120]}")
        print(f"\n  [Stall guard] {str(e)[:100]}...\n")
        return best_result or "", {}, "hard", str(e)


def _build_task_prompt(
    task: str,
    plan_context: str,
    retry_context: str = "",
    attempt: int = 1,
    stage_guidance: str = "",
    stall_guidance: str = "",
) -> str:
    parts = [task]

    if len(conversation_history) > 1:
        recent = conversation_history[-3:]
        history_lines = "\n".join(
            f"  [{m['role']}]: {m['content'][:100]}" for m in recent[:-1]
        )
        parts.append(f"\nConversation so far:\n{history_lines}")

    parts.append(f"\n{plan_context}")

    if stage_guidance:
        parts.append(f"\n{stage_guidance}")

    if stall_guidance:
        parts.append(f"\nRECOVERY GUIDANCE: {stall_guidance}")

    if retry_context:
        parts.append(f"\nRetry #{attempt}: {retry_context}")

    return "\n".join(parts)


def run(task: str) -> None:
    task_start = time.time()
    conversation_history.append({"role": "user", "content": task})

    # ── 1. Perception ────────────────────────────────────────────
    perception = perception_layer.process(task)

    # Short-circuit for chitchat
    if perception.task_type == "chitchat":
        reply = _CHITCHAT_REPLIES.get(
            task.strip().lower().rstrip("!.,?"),
            "How can I help you?"
        )
        print(reply)
        conversation_history.append({"role": "assistant", "content": reply})
        return

    # ── 2. Memory ────────────────────────────────────────────────
    memory_context = memory_layer.retrieve_relevant(task)
    recent = memory_layer.recent_context()
    combined_context = "\n".join(filter(None, [memory_context, recent]))

    # ── 3. Planning ──────────────────────────────────────────────
    plan = planning_layer.create_plan(perception, memory_context=combined_context)

    # ── 4. Compaction ────────────────────────────────────────────
    if compaction_layer.needs_compaction(conversation_history):
        conversation_history[:] = compaction_layer.compact(conversation_history)

    # ── 5. Checkpoint — resume or start fresh ────────────────────
    checkpoint = checkpoint_layer.load(task)
    if checkpoint:
        print(f"  [Resume] Attempt {checkpoint.attempt}, stage={checkpoint.stage}, best_score={checkpoint.best_score:.2f}")
        stage_pipeline.state.current = checkpoint.stage
        best_result = checkpoint.best_result
        best_evaluation = None
        start_attempt = checkpoint.attempt
        stall_guidance = checkpoint.stall_guidance
    else:
        checkpoint_layer.init_progress(task)
        best_result = ""
        best_evaluation = None
        start_attempt = 1
        stall_guidance = ""
        stage_pipeline.reset()

    # ── 6. Retry loop ────────────────────────────────────────────
    retry_context = ""
    best_critique: dict = {}
    max_retries = _retries_for(perception.complexity)
    print(f"  [Retry budget] complexity={perception.complexity} → max_retries={max_retries}")

    # max_retries+1 total attempts (1 initial + max_retries retries); range end is exclusive → +2
    for attempt in range(start_attempt, max_retries + 2):
        attempt_start = time.time()
        stage_guidance = stage_pipeline.get_guidance()
        enriched_task = _build_task_prompt(
            task,
            plan.as_prompt_context(),
            retry_context,
            attempt,
            stage_guidance,
            stall_guidance,
        )

        # Compact agent.memory.steps when the previous attempt approached the token budget.
        # The orchestrator's internal step list — not conversation_history — is the
        # actual context the LLM sees on the next agent.run() call.
        if attempt > start_attempt and token_budget.is_near_limit():
            steps = getattr(agent.memory, "steps", None)
            if steps:
                n_steps = len(steps)
                summary = "\n".join(
                    f"  Step {getattr(s, 'step_number', '?')}: "
                    f"{str(getattr(s, 'observations', '') or '')[:200]}"
                    for s in steps[-3:]
                )
                conversation_history.append({
                    "role": "system",
                    "content": f"[Step memory compacted — {n_steps} steps cleared]\n{summary[:1000]}",
                })
                try:
                    steps.clear()
                except Exception:
                    pass
                print(f"\n  [Step compaction] Cleared {n_steps} steps ({token_budget.summary()})\n")

        stall_detector.reset()
        token_budget.reset()
        stall_guidance = ""

        checkpoint_layer.append_progress(
            f"\n## Attempt {attempt} | Stage: {stage_pipeline.current_stage}"
        )

        try:
            result, critique, failure_type, stall_str = _execute_attempt(
                enriched_task, task, best_result, attempt
            )
        except Exception as e:
            print(output_layer.render_error(str(e), perception))
            checkpoint_layer.append_progress(f"  ERROR: {e}")
            if best_result and best_evaluation is not None:
                memory_layer.store(Interaction(
                    task=task, intent=perception.intent, plan_pattern=plan.pattern,
                    result_summary=best_result[:200], evaluation_score=best_evaluation.score,
                    tags=perception.keywords[:5],
                ))
            return
        if stall_str:
            stall_guidance = (stall_str + ("\n" + stall_guidance if stall_guidance else "")).strip()

        # Merge soft guidance — both signals carry useful information and should not suppress each other
        if stall_detector.state.soft_guidance:
            stall_guidance = (stall_guidance + "\n" + stall_detector.state.soft_guidance).strip() if stall_guidance else stall_detector.state.soft_guidance

        # ── 7. Verification ──────────────────────────────────────
        agent_steps = getattr(agent.memory, "steps", None)
        vr = verification_layer.verify(result, perception.task_type, agent_steps)
        if not vr.passed:
            if vr.requires_smoke_test:
                stall_guidance = (stall_guidance + "\n" if stall_guidance else "") + \
                    verification_layer.build_smoke_test_guidance()
            checkpoint_layer.append_progress(f"  VERIFY FAIL: {vr.reason}")

        # ── 8. Evaluation ────────────────────────────────────────
        evaluation = evaluation_layer.evaluate(result, perception, plan)

        if best_evaluation is None or evaluation.score > best_evaluation.score:
            best_result = result
            best_evaluation = evaluation
            best_critique = critique if isinstance(critique, dict) else {}

        # ── 9. Checkpoint save ───────────────────────────────────
        cp = Checkpoint(
            task=task,
            attempt=attempt + 1,
            best_result=best_result,
            best_score=best_evaluation.score,
            stage=stage_pipeline.current_stage,
            conversation_history=list(conversation_history),
            stall_guidance=stall_guidance,
        )
        checkpoint_layer.save(cp)

        # ── 10. Stage advance ────────────────────────────────────
        stage_changed = stage_pipeline.tick(evaluation.score)
        if stage_changed:
            checkpoint_layer.append_progress(
                f"  Stage → {stage_pipeline.current_stage}"
            )

        if evaluation.passed and vr.passed:
            break

        if attempt <= max_retries:
            if failure_type == "hard":
                # Hard failure (stall/crash): inject recovery guidance, keep plan
                retry_context = evaluation.retry_prompt
                if stall_guidance:
                    retry_context = f"{stall_guidance}\n{retry_context}"
                mode_str = "recovery"
            else:
                # Soft failure: replan with fresh memory context to break out of the quality rut
                memory_context = memory_layer.retrieve_relevant(task)
                recent = memory_layer.recent_context()
                combined_context = "\n".join(filter(None, [memory_context, recent]))
                plan = planning_layer.create_plan(perception, memory_context=combined_context)
                retry_context = f"[REPLANNING]\n{plan.as_prompt_context()}\n{evaluation.retry_prompt}"
                mode_str = "replanning"
            score_str = f"{evaluation.score:.2f}"
            stage_str = stage_pipeline.summary()
            elapsed = time.time() - attempt_start
            print(f"\n  [Retry {attempt}/{max_retries}] Score {score_str} | {stage_str} | {mode_str} | {elapsed:.1f}s — retrying...\n")

    # ── 11. Synthesize final answer ──────────────────────────────
    if A2A_ENABLED and best_result:
        synth_msg = A2AMessage(
            sender="loop",
            recipient="synthesizer",
            task=task,
            payload={"result": best_result, "critique": best_critique.get("issues", "")},
        )
        synth_resp = hub.send(synth_msg)
        if synth_resp.status == "success" and synth_resp.result.strip():
            best_result = synth_resp.result

    # Guard: if the loop body never executed (e.g. start_attempt > max_retries+1 on resume),
    # best_evaluation is still None. Re-derive from best_result if possible.
    if best_evaluation is None:
        if best_result:
            best_evaluation = evaluation_layer.evaluate(best_result, perception, plan)
        else:
            return

    # ── 12. Memory storage ───────────────────────────────────────
    memory_layer.store(Interaction(
        task=task,
        intent=perception.intent,
        plan_pattern=plan.pattern,
        result_summary=best_result[:200],
        evaluation_score=best_evaluation.score,
        tags=perception.keywords[:5],
    ))

    conversation_history.append({"role": "assistant", "content": best_result[:300]})

    # Clear checkpoint on successful completion
    if best_evaluation.passed:
        checkpoint_layer.clear()

    # ── 13. Output ───────────────────────────────────────────────
    print(f"  [Task complete] {time.time() - task_start:.1f}s total\n")
    print(output_layer.render(
        result=best_result,
        perception=perception,
        plan=plan,
        evaluation=best_evaluation,
        memory_stats=memory_layer.stats(),
    ))


# ── Main loop ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal

    def _handle_shutdown(sig, frame):
        print("\nGoodbye.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    print("Agentic AI ready. Type your task, 'quit' to exit.")
    print("Commands: /new (reset session), /history (show turns), /help\n")
    while True:
        try:
            task = input("Task: ").strip()
            if not task:
                continue
            if task.lower() in ("quit", "exit"):
                print("Goodbye.")
                break
            if task == "/new":
                conversation_history.clear()
                memory_layer._short_term.clear()
                checkpoint_layer.clear()
                stage_pipeline.reset()
                print("Session reset.\n")
                continue
            if task == "/history":
                if not conversation_history:
                    print("(no history)\n")
                else:
                    for m in conversation_history:
                        print(f"  [{m['role']}]: {m['content'][:120]}")
                    print()
                continue
            if task == "/help":
                print("  /new      — clear conversation & memory, start fresh")
                print("  /history  — show recent turns")
                print("  /help     — show this help")
                print("  quit/exit — exit the agent\n")
                continue
            run(task)
        except (KeyboardInterrupt, SystemExit):
            print("\nGoodbye.")
            break
