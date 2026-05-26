from dataclasses import dataclass, field
from typing import List, Optional

from .perception import Perception
from .planning import Plan


FAILURE_SIGNALS = [
    "i don't know", "i cannot", "unable to", "error", "failed",
    "sorry", "no result", "not found", "none", "n/a",
]

DEFAULT_MIN_PASS_SCORE = 0.5


@dataclass
class Evaluation:
    score: float
    passed: bool
    feedback: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    retry_prompt: str = ""

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        fb = " | ".join(f for f in self.feedback if f) or "No issues"
        return f"[{status}] Score: {self.score:.2f} — {fb}"

    def as_retry_context(self) -> str:
        issues = "; ".join(f for f in self.feedback if f) or "output was incomplete"
        return (
            f"Your previous attempt scored {self.score:.2f}/1.00. "
            f"Issues: {issues}. "
            f"Please try again, addressing these issues: {'; '.join(self.suggestions) or 'be more thorough and complete'}."
        )


class EvaluationLayer:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.min_pass_score: float = cfg.get("min_pass_score", DEFAULT_MIN_PASS_SCORE)

    def evaluate(self, result: str, perception: Perception, plan: Plan) -> Evaluation:
        checks = [
            self._check_non_empty(result),
            self._check_no_failure_signals(result),
            self._check_entity_coverage(result, perception),
            self._check_length(result, perception),
            self._check_plan_coverage(result, plan),
        ]

        scores = [c[0] for c in checks]
        feedback = [c[1] for c in checks if c[1]]
        score = round(sum(scores) / len(scores), 2)
        passed = score >= self.min_pass_score
        suggestions = self._suggest_improvements(score, perception, result)

        evaluation = Evaluation(
            score=score,
            passed=passed,
            feedback=feedback,
            suggestions=suggestions,
        )
        if not passed:
            evaluation.retry_prompt = evaluation.as_retry_context()

        return evaluation

    def _check_non_empty(self, result: str) -> tuple:
        if not result or len(result.strip()) < 10:
            return 0.0, "Result is empty or too short"
        return 1.0, ""

    def _check_no_failure_signals(self, result: str) -> tuple:
        lower = result.lower()
        hits = [s for s in FAILURE_SIGNALS if s in lower]
        if len(hits) >= 2:
            return 0.2, f"Result contains failure signals: {hits[:2]}"
        if hits:
            return 0.6, f"Result may be incomplete: '{hits[0]}' detected"
        return 1.0, ""

    def _check_entity_coverage(self, result: str, perception: Perception) -> tuple:
        if not perception.entities:
            return 1.0, ""
        lower = result.lower()
        covered = [e for e in perception.entities if e.lower() in lower]
        ratio = len(covered) / len(perception.entities)
        if ratio < 0.3:
            missing = [e for e in perception.entities if e.lower() not in lower]
            return ratio, f"Missing entities in result: {missing[:3]}"
        return min(1.0, ratio + 0.2), ""

    def _check_length(self, result: str, perception: Perception) -> tuple:
        length = len(result.split())
        if perception.complexity == "complex" and length < 50:
            return 0.5, "Result too brief for a complex task"
        if perception.complexity == "simple" and length > 500:
            return 0.8, "Result verbose for a simple task"
        return 1.0, ""

    def _check_plan_coverage(self, result: str, plan: Plan) -> tuple:
        if plan.pattern == "research_and_report" and len(result.split()) < 30:
            return 0.5, "Research result seems too short"
        if plan.pattern == "code_generation" and "def " not in result and "```" not in result:
            return 0.6, "Code generation result may not contain actual code"
        return 1.0, ""

    def _suggest_improvements(self, score: float, perception: Perception, result: str) -> List[str]:
        suggestions = []
        if score < 0.5:
            suggestions.append("Be more specific and thorough in your response")
        if perception.complexity == "complex" and len(result.split()) < 100:
            suggestions.append("Elaborate more — the task is complex and requires a detailed answer")
        if perception.intent == "code_gen" and "error" in result.lower():
            suggestions.append("Specify the language and expected input/output more clearly")
        return suggestions
