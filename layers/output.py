from .evaluation import Evaluation
from .perception import Perception
from .planning import Plan


DIVIDER = "─" * 60
SECTION = "═" * 60


class OutputLayer:
    def render(
        self,
        result: str,
        perception: Perception,
        plan: Plan,
        evaluation: Evaluation,
        memory_stats: dict,
    ) -> str:
        blocks = [
            self._header(perception),
            self._plan_block(plan),
            self._result_block(result, perception),
            self._evaluation_block(evaluation),
            self._memory_block(memory_stats),
        ]
        return "\n".join(blocks)

    def render_error(self, error: str, perception: Perception) -> str:
        return (
            f"\n{SECTION}\n"
            f"  AGENT ERROR\n"
            f"{DIVIDER}\n"
            f"  Task   : {perception.raw_input[:80]}\n"
            f"  Intent : {perception.intent}\n"
            f"{DIVIDER}\n"
            f"  {error}\n"
            f"{SECTION}\n"
        )

    def _header(self, perception: Perception) -> str:
        return (
            f"\n{SECTION}\n"
            f"  TASK    : {perception.raw_input[:70]}\n"
            f"  Intent  : {perception.intent:<15} Complexity: {perception.complexity}\n"
            f"  Entities: {', '.join(perception.entities) or 'none'}\n"
            f"{DIVIDER}"
        )

    def _plan_block(self, plan: Plan) -> str:
        lines = [f"  PLAN  [{plan.pattern}]"]
        for step in plan.steps:
            lines.append(f"    {step.index}. {step.action}")
            lines.append(f"       Tool: {step.tool} — {step.rationale[:60]}")
        if plan.memory_context:
            lines.append(f"  Memory: {plan.memory_context[:80]}")
        return "\n".join(lines) + f"\n{DIVIDER}"

    def _result_block(self, result: str, perception: Perception) -> str:
        label = self._result_label(perception.intent)
        lines = [f"  {label}"]
        for line in result.splitlines():
            lines.append(f"    {line}")
        return "\n".join(lines) + f"\n{DIVIDER}"

    def _evaluation_block(self, evaluation: Evaluation) -> str:
        status = "✓ PASS" if evaluation.passed else "✗ FAIL"
        bar = self._score_bar(evaluation.score)
        lines = [
            f"  EVALUATION  {status}",
            f"    Score : {bar} {evaluation.score:.2f}/1.00",
        ]
        if evaluation.feedback:
            for fb in evaluation.feedback:
                if fb:
                    lines.append(f"    Note  : {fb}")
        if evaluation.suggestions:
            for sg in evaluation.suggestions:
                lines.append(f"    Tip   : {sg}")
        return "\n".join(lines) + f"\n{DIVIDER}"

    def _memory_block(self, stats: dict) -> str:
        return (
            f"  MEMORY  "
            f"Short-term: {stats['short_term_count']}  "
            f"Long-term: {stats['long_term_count']}  "
            f"Avg score: {stats['avg_score']}\n"
            f"{SECTION}\n"
        )

    def _result_label(self, intent: str) -> str:
        labels = {
            "research":    "RESEARCH RESULT",
            "compute":     "COMPUTATION RESULT",
            "file_op":     "FILE RESULT",
            "code_gen":    "GENERATED CODE",
            "fact_check":  "FACT CHECK",
            "data_process":"PROCESSED DATA",
            "general":     "RESULT",
        }
        return labels.get(intent, "RESULT")

    def _score_bar(self, score: float, width: int = 10) -> str:
        filled = round(score * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"
