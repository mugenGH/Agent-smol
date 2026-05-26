"""Token budget tracking via smolagents step callbacks.

Accumulates token counts per attempt and acts as a circuit breaker:
warns at warn_threshold (default 80%), aborts the agent run by raising
StallError at the hard budget. Works with ActionStep.token_usage
(smolagents ≥0.3); falls back to length-based estimation otherwise.
"""
from dataclasses import dataclass

from .guards import StallError


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenBudgetLayer:
    """Tracks cumulative token usage for a task run via step_callback.

    Register on CodeAgent alongside StallDetector:
      CodeAgent(..., step_callbacks=[stall_detector, token_budget])

    Call reset() before each attempt. is_near_limit() exposes budget
    pressure to the outer loop; summary() produces a status line.
    """

    def __init__(self, cfg: dict):
        self.budget: int = cfg.get("max_tokens", 32_000)
        self.warn_threshold: float = cfg.get("warn_threshold", 0.8)
        self._usage = _Usage()
        self._warned = False

    def reset(self) -> None:
        self._usage = _Usage()
        self._warned = False

    @property
    def used(self) -> int:
        return self._usage.total

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def fraction_used(self) -> float:
        return self._usage.total / self.budget if self.budget > 0 else 0.0

    def is_near_limit(self) -> bool:
        return self.fraction_used >= self.warn_threshold

    def is_exhausted(self) -> bool:
        return self._usage.total >= self.budget

    def __call__(self, step, agent=None) -> None:
        """Called by smolagents after each ActionStep."""
        try:
            from smolagents.memory import ActionStep
        except ImportError:
            return

        if not isinstance(step, ActionStep):
            return

        usage = getattr(step, "token_usage", None)
        if usage is not None:
            self._usage.input_tokens += (
                getattr(usage, "input_tokens", 0)
                or getattr(usage, "prompt_tokens", 0)
            )
            self._usage.output_tokens += (
                getattr(usage, "output_tokens", 0)
                or getattr(usage, "completion_tokens", 0)
            )
        else:
            # Estimate: ~4 chars per token
            content = (step.code_action or "") + str(
                getattr(step, "observations", "") or ""
            )
            self._usage.output_tokens += max(1, len(content) // 4)

        if self.is_near_limit() and not self._warned:
            self._warned = True
            print(
                f"\n  [Token budget] {self.used:,}/{self.budget:,} tokens used "
                f"({self.fraction_used:.0%}) — approaching limit\n"
            )

        # Circuit breaker: hard-stop the agent when over budget
        if self.is_exhausted():
            raise StallError(
                f"TOKEN BUDGET EXHAUSTED: {self.used:,}/{self.budget:,} tokens. "
                "Be more concise — summarize tool outputs and avoid re-reading."
            )

    def summary(self) -> str:
        return (
            f"tokens={self.used:,}/{self.budget:,} "
            f"({self.fraction_used:.0%} used, {self.remaining:,} remaining)"
        )
