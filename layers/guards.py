"""Stall detection via smolagents step_callback.

Registers on the CodeAgent and raises StallError when the inner loop
gets stuck — parse error bursts, consecutive failures, identical action
streaks, write-thrash, or sustained high failure rates.
"""
import re
from dataclasses import dataclass, field
from typing import List


class StallError(RuntimeError):
    """Raised from the step callback to abort agent.run() early."""
    pass


@dataclass
class StallState:
    consecutive_errors: int = 0
    parse_error_streak: int = 0
    last_action_sig: str = ""
    identical_streak: int = 0
    file_write_counts: dict = field(default_factory=dict)
    recent_errors: List[bool] = field(default_factory=list)
    soft_guidance: str = ""
    hard_stopped: bool = False


class StallDetector:
    """
    smolagents step_callback that aborts the inner loop on hard stalls
    and sets soft_guidance for the outer retry loop to inject.

    Register as:  agent = CodeAgent(..., step_callbacks=[stall_detector])
    Check after:  if stall_detector.state.soft_guidance: ...
    Reset before: stall_detector.reset()
    """

    def __init__(self, cfg: dict):
        self.max_consecutive_errors: int = cfg.get("max_consecutive_errors", 5)
        self.max_identical_streak: int = cfg.get("max_identical_streak", 4)
        self.max_parse_streak: int = cfg.get("max_parse_streak", 3)
        self.write_thrash_threshold: int = cfg.get("write_thrash_threshold", 4)
        self.failure_rate_window: int = cfg.get("failure_rate_window", 10)
        self.failure_rate_threshold: float = cfg.get("failure_rate_threshold", 0.6)
        self.state = StallState()

    def reset(self) -> None:
        self.state = StallState()

    def __call__(self, step, agent=None) -> None:
        """Called by smolagents after each ActionStep."""
        try:
            from smolagents.memory import ActionStep
            from smolagents.utils import AgentParsingError
        except ImportError:
            return

        if not isinstance(step, ActionStep):
            return

        has_error = step.error is not None
        is_parse_error = has_error and isinstance(step.error, AgentParsingError)

        # ── Error tracking ────────────────────────────────────────
        if has_error:
            self.state.consecutive_errors += 1
        else:
            self.state.consecutive_errors = 0

        if is_parse_error:
            self.state.parse_error_streak += 1
        else:
            self.state.parse_error_streak = 0

        # ── Identical action streak ───────────────────────────────
        sig = (step.code_action or "")[:300]
        if sig and sig == self.state.last_action_sig:
            self.state.identical_streak += 1
        else:
            self.state.identical_streak = 0
            self.state.last_action_sig = sig

        # ── Sliding failure rate window ───────────────────────────
        self.state.recent_errors.append(has_error)
        if len(self.state.recent_errors) > self.failure_rate_window:
            self.state.recent_errors.pop(0)

        # ── Write-thrash tracking ─────────────────────────────────
        if step.code_action and not has_error:
            for path in re.findall(r'write_file\(\s*["\']([^"\']+)["\']', step.code_action):
                self.state.file_write_counts[path] = (
                    self.state.file_write_counts.get(path, 0) + 1
                )

        # ── Hard stops (raise to abort agent.run()) ───────────────
        if self.state.parse_error_streak >= self.max_parse_streak:
            msg = (
                f"PARSE RECOVERY: {self.state.parse_error_streak} consecutive parse failures. "
                "Your response MUST be valid Python inside <code>...</code> tags. "
                "No prose, no markdown fences — just code."
            )
            self.state.soft_guidance = msg
            self.state.hard_stopped = True
            raise StallError(msg)

        if self.state.consecutive_errors >= self.max_consecutive_errors:
            msg = (
                f"STALL: {self.state.consecutive_errors} consecutive tool errors. "
                "Change approach: read a different file, simplify the task, or use a different tool."
            )
            self.state.soft_guidance = msg
            self.state.hard_stopped = True
            raise StallError(msg)

        if self.state.identical_streak >= self.max_identical_streak:
            msg = (
                f"LOOP: same action repeated {self.state.identical_streak} times identically. "
                "This is a loop — you must take a completely different action."
            )
            self.state.soft_guidance = msg
            self.state.hard_stopped = True
            raise StallError(msg)

        # ── Soft guards (set guidance, don't abort) ───────────────
        for path, count in self.state.file_write_counts.items():
            if count >= self.write_thrash_threshold:
                self.state.soft_guidance = (
                    f"WRITE THRASH: '{path}' rewritten {count} times. "
                    "Stop rewriting it — read the file first, make ONE targeted fix."
                )
                break

        if not self.state.soft_guidance and len(self.state.recent_errors) >= self.failure_rate_window:
            rate = sum(self.state.recent_errors) / len(self.state.recent_errors)
            if rate >= self.failure_rate_threshold:
                self.state.soft_guidance = (
                    f"HIGH FAILURE RATE ({int(rate * 100)}% of recent steps failed). "
                    "Re-read key files, verify paths, or try a fundamentally different approach."
                )
