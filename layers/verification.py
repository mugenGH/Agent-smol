"""Completion verification layer.

Inspects the agent's result for evidence of actual work done.
For code tasks: requires smoke test evidence.
For all tasks: checks for completion signals and artifact evidence.

Mirrors Agent-loop's smoke test gate and sandbox verification concept,
adapted for the smolagents wrapper architecture.
"""
import re
from dataclasses import dataclass


SMOKE_OK_PATTERN = re.compile(r'\bSMOKE\s+OK\b', re.IGNORECASE)
FILE_WRITE_PATTERN = re.compile(r'write_file\s*\(|Written to |saved to |created file', re.IGNORECASE)
CODE_PATTERN = re.compile(r'```|def |class |import |function |const |let |var ')
ERROR_PATTERN = re.compile(r'\bTraceback\b|\bError:\b|\bException:\b|\bSyntaxError\b')

COMPLETION_SIGNALS = [
    "completed", "done", "finished", "successfully", "result:", "answer:",
    "output:", "solution:", "here is", "here's", "i have",
]

FAILURE_SIGNALS = [
    "i cannot", "i'm unable", "i don't know", "not possible",
    "no information", "couldn't find", "failed to",
]


@dataclass
class VerificationResult:
    passed: bool
    reason: str
    requires_smoke_test: bool = False
    smoke_test_passed: bool = False


class VerificationLayer:
    """Checks that the agent's result represents genuine task completion."""

    def __init__(self, cfg: dict):
        self.require_smoke_test: bool = cfg.get("require_smoke_test", False)
        self.min_result_words: int = cfg.get("min_result_words", 5)
        self.enabled: bool = cfg.get("enabled", True)

    def verify(self, result: str, task_type: str, agent_steps=None) -> VerificationResult:
        """
        Verify result quality. agent_steps is agent.memory.steps (ActionStep list)
        from the last run, used to check for file write evidence.
        """
        if not self.enabled:
            return VerificationResult(passed=True, reason="verification disabled")

        result_lower = result.lower()

        # Insufficient word count — char threshold was gaming the check
        word_count = len(result.strip().split())
        if word_count < self.min_result_words:
            return VerificationResult(
                passed=False,
                reason=f"Result too short ({word_count} words, need {self.min_result_words})"
            )

        # Outright failure signals
        failure_hits = [s for s in FAILURE_SIGNALS if s in result_lower]
        if len(failure_hits) >= 2:
            return VerificationResult(
                passed=False,
                reason=f"Result contains failure signals: {failure_hits[:2]}"
            )

        # Code task verification
        is_code_task = task_type in ("code_generation", "data_processing", "file_inspection")
        if is_code_task and self.require_smoke_test:
            smoke_passed = self._check_smoke_test(result, agent_steps)
            if not smoke_passed:
                return VerificationResult(
                    passed=False,
                    reason=(
                        "Code task requires a smoke test. "
                        "Write a _smoke_test.py that prints 'SMOKE OK' and run it. "
                        "Do not call finish until SMOKE OK appears in the output."
                    ),
                    requires_smoke_test=True,
                    smoke_test_passed=False,
                )
            return VerificationResult(
                passed=True,
                reason="Smoke test passed",
                requires_smoke_test=True,
                smoke_test_passed=True,
            )

        # General completion check
        has_completion_signal = any(s in result_lower for s in COMPLETION_SIGNALS)
        has_content = len(result.split()) >= 10
        has_no_unresolved_errors = not ERROR_PATTERN.search(result)

        if has_content and (has_completion_signal or has_no_unresolved_errors):
            return VerificationResult(passed=True, reason="result passes quality checks")

        return VerificationResult(
            passed=False,
            reason="Result lacks completion signals or contains unresolved errors"
        )

    def _check_smoke_test(self, result: str, agent_steps) -> bool:
        """Check for SMOKE OK in result text or in agent step observations."""
        if SMOKE_OK_PATTERN.search(result):
            return True

        if agent_steps:
            for step in agent_steps:
                obs = getattr(step, "observations", "") or ""
                if SMOKE_OK_PATTERN.search(str(obs)):
                    return True

        return False

    def build_smoke_test_guidance(self) -> str:
        return (
            "VERIFICATION REQUIRED: Write a _smoke_test.py that:\n"
            "  1. Imports the key functions/classes\n"
            "  2. Calls them with known inputs\n"
            "  3. Uses assert statements (not console.assert)\n"
            "  4. Prints 'SMOKE OK' only after ALL checks pass\n"
            "Run it with python3 _smoke_test.py. It must exit 0 and print 'SMOKE OK'."
        )
