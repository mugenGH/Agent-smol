"""Crash recovery via per-task checkpoint files.

Saves full outer-loop state after each attempt so a session can
be resumed if the process is killed or crashes mid-run.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


CHECKPOINT_DIR = Path(".agent")
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json"
PROGRESS_FILE = CHECKPOINT_DIR / "progress.md"


@dataclass
class Checkpoint:
    task: str
    attempt: int = 0
    best_result: str = ""
    best_score: float = 0.0
    stage: str = "execution"
    conversation_history: List[dict] = field(default_factory=list)
    stall_guidance: str = ""
    total_steps: int = 0


class CheckpointLayer:
    """Saves and loads per-task checkpoints for crash recovery."""

    def __init__(self):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def save(self, cp: Checkpoint) -> None:
        try:
            CHECKPOINT_FILE.write_text(json.dumps(asdict(cp), indent=2))
        except Exception:
            pass

    def load(self, task: str) -> Optional[Checkpoint]:
        """Return a saved checkpoint only if it matches the current task."""
        try:
            if CHECKPOINT_FILE.exists():
                data = json.loads(CHECKPOINT_FILE.read_text())
                if data.get("task") == task:
                    return Checkpoint(**{
                        k: v for k, v in data.items()
                        if k in Checkpoint.__dataclass_fields__
                    })
        except Exception:
            pass
        return None

    def clear(self) -> None:
        try:
            CHECKPOINT_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    def append_progress(self, text: str) -> None:
        """Append a line to the persistent step-by-step progress log."""
        try:
            with PROGRESS_FILE.open("a") as f:
                f.write(text + "\n")
        except Exception:
            pass

    def init_progress(self, task: str) -> None:
        try:
            from datetime import datetime
            PROGRESS_FILE.write_text(
                f"# Task: {task}\nStarted: {datetime.now().isoformat()}\n\n"
            )
        except Exception:
            pass
