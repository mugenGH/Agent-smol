import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .perception import Perception


KG_PATH = Path(__file__).parent.parent / "knowledge_graph.json"


@dataclass
class Step:
    index: int
    action: str
    tool: str
    rationale: str
    depends_on: Optional[int] = None


@dataclass
class Plan:
    goal: str
    pattern: str
    steps: List[Step] = field(default_factory=list)
    memory_context: str = ""
    complexity: str = "simple"

    def summary(self) -> str:
        lines = [f"Pattern: {self.pattern} | Steps: {len(self.steps)}"]
        for s in self.steps:
            dep = f" (after step {s.depends_on})" if s.depends_on else ""
            lines.append(f"  {s.index}. [{s.tool}] {s.action}{dep}")
        return "\n".join(lines)

    def as_prompt_context(self) -> str:
        lines = [f"Goal: {self.goal}", f"Plan ({self.pattern}):"]
        for s in self.steps:
            lines.append(f"  Step {s.index}: {s.action} using {s.tool}")
        if self.memory_context:
            lines.append(f"Relevant past context: {self.memory_context}")
        return "\n".join(lines)


TOOL_FOR_ACTION = {
    "web_search":     "DuckDuckGoSearch",
    "code_execution": "Python interpreter",
    "read_file":      "read_file",
    "write_file":     "write_file",
    "query_kg":       "query_knowledge_graph",
}

PATTERN_STEPS = {
    "research_and_report": [
        ("Search for information about the topic", "web_search"),
        ("Summarize and organize the findings", "code_execution"),
        ("Save the report to a file if requested", "write_file"),
    ],
    "data_processing": [
        ("Read the source data file", "read_file"),
        ("Process and transform the data", "code_execution"),
        ("Save the processed output", "write_file"),
    ],
    "fact_check": [
        ("Search for evidence about the claim", "web_search"),
        ("Compare evidence and reason about truth", "code_execution"),
    ],
    "code_generation": [
        ("Write the code logic", "code_execution"),
        ("Test and verify the code runs correctly", "code_execution"),
        ("Save the code to a file", "write_file"),
    ],
    "file_inspection": [
        ("Read the target file", "read_file"),
        ("Parse and analyze the contents", "code_execution"),
    ],
}


class PlanningLayer:
    def __init__(self):
        self.kg = json.loads(KG_PATH.read_text())

    def create_plan(self, perception: Perception, memory_context: str = "") -> Plan:
        pattern = self._select_pattern(perception)
        raw_steps = self._get_steps(pattern, perception)

        if perception.complexity == "complex":
            raw_steps = self._expand_for_complexity(raw_steps, perception)

        steps = [
            Step(
                index=i + 1,
                action=action,
                tool=tool,
                rationale=self._rationale(tool, perception),
                depends_on=i if i > 0 else None,
            )
            for i, (action, tool) in enumerate(raw_steps)
        ]

        return Plan(
            goal=perception.raw_input,
            pattern=pattern,
            steps=steps,
            memory_context=memory_context,
            complexity=perception.complexity,
        )

    def _select_pattern(self, perception: Perception) -> str:
        task = perception.task_type
        if task in PATTERN_STEPS:
            return task
        # fallback: find closest match in KG
        for key in self.kg["task_patterns"]:
            if perception.intent in key or key in perception.intent:
                return key
        return "research_and_report"

    def _get_steps(self, pattern: str, perception: Perception) -> List[tuple]:
        base = list(PATTERN_STEPS.get(pattern, PATTERN_STEPS["research_and_report"]))
        if not perception.file_paths and pattern in ("data_processing", "file_inspection"):
            base = [s for s in base if s[1] != "read_file"]
        return base

    def _expand_for_complexity(self, steps: list, perception: Perception) -> list:
        expanded = []
        for action, tool in steps:
            expanded.append((action, tool))
            if tool == "web_search" and len(perception.keywords) > 3:
                expanded.append(("Search for additional context on sub-topics", "web_search"))
        return expanded

    def _rationale(self, tool: str, perception: Perception) -> str:
        info = self.kg["tools"].get(tool, {})
        hints = info.get("use_when", [])
        for hint in hints:
            for kw in perception.keywords:
                if kw in hint.lower():
                    return hint
        return hints[0] if hints else f"Best tool for {perception.intent} tasks"
