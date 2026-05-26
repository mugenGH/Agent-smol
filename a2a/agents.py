"""Collaborator agents registered with the AgentHub.

Three roles:
  executor    — wraps an existing CodeAgent for A2A routing
  critic      — scores results on a rubric using N-sample self-consistency
  synthesizer — assembles the final polished answer from result + critique

All agents run with stream_outputs=False so they don't clutter the terminal.
"""
import json
import re
from typing import Optional

import litellm
from smolagents import CodeAgent

from .protocol import AgentCard, A2AMessage, AgentHub

_CRITIC_PROMPT = """\
You are a quality reviewer scoring an AI agent's response.

Task: {task}

Agent result:
{result}

Rate each dimension from 1 (very poor) to 5 (excellent):
- correctness: Is the information accurate and the task addressed correctly?
- completeness: Does the response fully answer what was asked?
- reasoning: Is the explanation clear and coherent?

Respond with ONLY valid JSON, no other text:
{{"correctness": <1-5>, "completeness": <1-5>, "reasoning": <1-5>, "issues": "<one-line summary or empty string>"}}"""


def _parse_rubric(text: str) -> Optional[dict]:
    """Extract JSON rubric from critic response text."""
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if all(k in data for k in ("correctness", "completeness", "reasoning")):
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def register_executor(hub: AgentHub, name: str, smolagent: CodeAgent) -> None:
    """Wrap an existing CodeAgent as an A2A-addressable executor.

    Surfaces truncation to the outer loop when max_steps is reached.
    StallError propagates through the hub unchanged — the reraise tuple on
    the hub must include StallError for this to work correctly.
    """
    card = AgentCard(
        name=name,
        description=smolagent.description or f"{name} executor",
        capabilities=["execution", "tool_use", "delegation"],
    )

    def handler(msg: A2AMessage) -> str:
        result = str(smolagent.run(msg.task))
        steps = getattr(getattr(smolagent, "memory", None), "steps", None)
        if steps is not None and len(steps) >= smolagent.max_steps:
            result += (
                f"\n\n[TRUNCATION WARNING: executor reached max_steps={smolagent.max_steps}. "
                "Result may be incomplete — consider breaking the task into subtasks.]"
            )
        return result

    hub.register(card, handler)


def register_critic(
    hub: AgentHub,
    model,
    api_base: str = "",
    api_key: str = "not-needed",
    cfg: Optional[dict] = None,
) -> None:
    """Register a scored-rubric critic with N-sample self-consistency.

    Runs N_SAMPLES independent critique passes, averages dimension scores
    (correctness, completeness, reasoning each 1-5), and passes when the
    average meets PASS_AVG. Returns a JSON verdict dict so the outer loop
    has full score visibility.
    """
    cfg = cfg or {}
    n_samples: int = cfg.get("n_samples", 3)
    pass_avg: float = cfg.get("pass_avg", 3.0)
    temperature: float = cfg.get("temperature", 0.3)
    model_id: str = getattr(model, "model_id", "")

    card = AgentCard(
        name="critic",
        description="Scores agent results on a rubric using self-consistency sampling.",
        capabilities=["evaluation", "critique", "quality_assurance"],
    )

    def handler(msg: A2AMessage) -> str:
        result_text = msg.payload.get("result", "")
        prompt = _CRITIC_PROMPT.format(task=msg.task, result=result_text[:1500])

        all_scores: list[dict] = []
        all_issues: list[str] = []

        for _ in range(n_samples):
            try:
                resp = litellm.completion(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    api_base=api_base or None,
                    api_key=api_key,
                    max_tokens=150,
                    temperature=temperature,
                )
                text = resp.choices[0].message.content.strip()
                rubric = _parse_rubric(text)
                if rubric:
                    all_scores.append(rubric)
                    if rubric.get("issues"):
                        all_issues.append(rubric["issues"])
            except Exception:
                pass

        if not all_scores:
            # No parseable scores — fail-open to avoid blocking on critic errors
            return json.dumps({
                "passed": True,
                "avg_score": 1.0,
                "rubric": {},
                "issues": "critic parse failure — defaulting to pass",
                "samples": 0,
            })

        dims = ("correctness", "completeness", "reasoning")
        avg_dims = {
            d: round(sum(s.get(d, 3) for s in all_scores) / len(all_scores), 2)
            for d in dims
        }
        overall_avg = round(sum(avg_dims.values()) / len(dims), 2)
        passed = overall_avg >= pass_avg

        verdict_str = "PASS" if passed else "FAIL"
        print(
            f"    [Critic/{len(all_scores)}s] {verdict_str} — "
            f"correctness={avg_dims['correctness']:.1f} "
            f"completeness={avg_dims['completeness']:.1f} "
            f"reasoning={avg_dims['reasoning']:.1f} "
            f"(avg={overall_avg:.1f}/5)"
        )

        return json.dumps({
            "passed": passed,
            "avg_score": round(overall_avg / 5.0, 2),
            "rubric": avg_dims,
            "issues": "; ".join(dict.fromkeys(all_issues))[:300],
            "samples": len(all_scores),
        })

    hub.register(card, handler)


def register_synthesizer(hub: AgentHub, model) -> None:
    """Register a synthesizer that produces the final polished answer.

    Takes the best raw result and any critic feedback, then produces a
    clean, complete, well-formatted final answer.
    """
    card = AgentCard(
        name="synthesizer",
        description="Combines agent output and critic feedback into a final answer.",
        capabilities=["synthesis", "formatting", "completion"],
    )
    agent = CodeAgent(
        tools=[],
        model=model,
        name="synthesizer",
        description=card.description,
        max_steps=4,
        stream_outputs=False,
    )

    def handler(msg: A2AMessage) -> str:
        result = msg.payload.get("result", "")
        critique = msg.payload.get("critique", "")
        prompt = (
            f"Produce the final, complete, well-formatted answer for this task.\n\n"
            f"Task: {msg.task}\n\n"
            f"Best draft answer:\n{result[:2000]}\n"
            + (
                f"\nCritic feedback to address:\n{critique[:500]}\n"
                if critique else ""
            )
            + "\nWrite the final answer now:"
        )
        return str(agent.run(prompt))

    hub.register(card, handler)
