"""4-stage outer-loop pipeline.

Drives the retry loop through structured stages instead of blind retries:
  execution    → first attempt, standard task prompt
  verification → result exists but quality is low; ask agent to verify/fix
  refinement   → still failing; inject specific feedback and targeted guidance
  finalization → final pass; ask agent to produce clean, complete output

Stages advance based on evaluation score and attempt count.
"""
from dataclasses import dataclass
from typing import Optional


STAGE_ORDER = ["execution", "verification", "refinement", "finalization"]

STAGE_GUIDANCE: dict[str, str] = {
    "execution": "",  # no extra guidance on first attempt
    "verification": (
        "VERIFICATION STAGE: Review your previous response carefully. "
        "Check that it fully answers the task, contains no errors, and is complete. "
        "If you wrote code, test it mentally. Fix any gaps before responding."
    ),
    "refinement": (
        "REFINEMENT STAGE: Your previous attempts were insufficient. "
        "Re-read the task requirements, identify exactly what is missing or wrong, "
        "and produce a corrected, complete response. Be specific and thorough."
    ),
    "finalization": (
        "FINALIZATION STAGE: This is your final attempt. "
        "Produce the best possible, complete, and well-formatted response. "
        "Include all required information and ensure correctness."
    ),
}

# Minimum score to advance past a stage without forcing it
STAGE_PASS_SCORE: dict[str, float] = {
    "execution": 0.7,
    "verification": 0.6,
    "refinement": 0.5,
    "finalization": 0.0,  # always accept on final stage
}

# How many attempts to spend in each stage before advancing regardless
STAGE_MAX_ATTEMPTS: dict[str, int] = {
    "execution": 1,
    "verification": 1,
    "refinement": 1,
    "finalization": 1,
}


@dataclass
class StageState:
    current: str = "execution"
    attempts_in_stage: int = 0
    total_attempts: int = 0


class StagePipeline:
    """Manages stage progression for the outer retry loop."""

    def __init__(self, cfg: dict):
        self.enabled: bool = cfg.get("enabled", True)
        self.state = StageState()

    def reset(self) -> None:
        self.state = StageState()

    @property
    def current_stage(self) -> str:
        return self.state.current

    def get_guidance(self) -> str:
        """Return the guidance string to inject for the current stage."""
        if not self.enabled:
            return ""
        return STAGE_GUIDANCE.get(self.state.current, "")

    def tick(self, score: float) -> bool:
        """
        Record an attempt result and advance the stage if appropriate.
        Returns True if the stage changed.
        """
        if not self.enabled:
            return False

        self.state.attempts_in_stage += 1
        self.state.total_attempts += 1

        current = self.state.current
        pass_score = STAGE_PASS_SCORE[current]
        max_attempts = STAGE_MAX_ATTEMPTS[current]
        idx = STAGE_ORDER.index(current)

        should_advance = (
            score < pass_score or
            self.state.attempts_in_stage >= max_attempts
        ) and idx < len(STAGE_ORDER) - 1

        if should_advance:
            self.state.current = STAGE_ORDER[idx + 1]
            self.state.attempts_in_stage = 0
            return True

        return False

    def is_final_stage(self) -> bool:
        return self.state.current == STAGE_ORDER[-1]

    def summary(self) -> str:
        return (
            f"Stage: {self.state.current} | "
            f"Stage attempts: {self.state.attempts_in_stage} | "
            f"Total: {self.state.total_attempts}"
        )
