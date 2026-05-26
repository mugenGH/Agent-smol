import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import litellm


INTENT_PATTERNS = {
    "chitchat":     ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "how are you", "good morning", "good evening", "good night", "sup", "yo"],
    "research":     ["research", "find out", "look up", "what is", "who is", "explain", "tell me about", "search"],
    "compute":      ["calculate", "compute", "how many", "sum", "average", "count", "convert", "math"],
    "file_op":      ["read", "write", "open", "save", "load", "file", "folder", "directory", "path"],
    "code_gen":     ["write code", "generate code", "create a script", "build a function", "implement"],
    "fact_check":   ["is it true", "verify", "confirm", "check if", "fact", "correct"],
    "data_process": ["parse", "process", "extract", "transform", "format", "csv", "json", "xml"],
    "general":      [],
}

TASK_TYPE_MAP = {
    "chitchat":     "chitchat",
    "research":     "research_and_report",
    "compute":      "data_processing",
    "file_op":      "file_inspection",
    "code_gen":     "code_generation",
    "fact_check":   "fact_check",
    "data_process": "data_processing",
    "general":      "research_and_report",
}

COMPLEXITY_SIGNALS = {
    "complex":  ["then", "and also", "after that", "multiple", "compare", "summarize and", "step by step"],
    "moderate": ["and", "then save", "and write", "then show"],
}

LLM_PERCEPTION_PROMPT = """\
Classify the following user task for an AI agent. Respond with ONLY a JSON object — no explanation.

Task: "{task}"

JSON schema:
{{
  "intent": one of [chitchat, research, compute, file_op, code_gen, fact_check, data_process, general],
  "task_type": one of [chitchat, research_and_report, data_processing, file_inspection, code_generation, fact_check, general],
  "entities": [list of key nouns or proper names, max 5],
  "complexity": one of [simple, moderate, complex]
}}"""


@dataclass
class Perception:
    raw_input: str
    intent: str
    task_type: str
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    complexity: str = "simple"
    file_paths: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    source: str = "heuristic"

    def summary(self) -> str:
        return (
            f"Intent: {self.intent} | Task: {self.task_type} | "
            f"Complexity: {self.complexity} | Entities: {self.entities} | Source: {self.source}"
        )


class PerceptionLayer:
    def __init__(self, cfg: dict):
        self.use_llm = cfg.get("use_llm", True)
        self._model_id = None

    def set_model(self, model_id: str, api_base: str, api_key: str = "not-needed") -> None:
        self._model_id = model_id
        self._api_base = api_base
        self._api_key = api_key

    def process(self, user_input: str) -> Perception:
        llm_result = None
        if self.use_llm and self._model_id:
            llm_result = self._llm_classify(user_input)

        if llm_result:
            intent = llm_result.get("intent", "general")
            task_type = llm_result.get("task_type", TASK_TYPE_MAP.get(intent, "research_and_report"))
            entities = llm_result.get("entities", [])
            complexity = llm_result.get("complexity", "simple")
            source = "llm"
        else:
            text = user_input.lower()
            intent = self._classify_intent(text)
            task_type = TASK_TYPE_MAP[intent]
            entities = self._extract_entities(user_input)
            complexity = self._assess_complexity(text)
            source = "heuristic"

        text = user_input.lower()
        return Perception(
            raw_input=user_input,
            intent=intent,
            task_type=task_type,
            entities=entities,
            keywords=self._extract_keywords(text),
            complexity=complexity,
            file_paths=self._extract_file_paths(user_input),
            questions=self._extract_questions(user_input),
            source=source,
        )

    def _llm_classify(self, user_input: str) -> Optional[dict]:
        try:
            prompt = LLM_PERCEPTION_PROMPT.format(task=user_input)
            response = litellm.completion(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
                api_base=self._api_base,
                api_key=self._api_key,
                max_tokens=200,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            # Extract JSON even if wrapped in markdown
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return None

    def _classify_intent(self, text: str) -> str:
        stripped = text.strip().rstrip("!.,?")
        for signal in INTENT_PATTERNS["chitchat"]:
            if stripped == signal or text.startswith(signal + " ") or text.endswith(" " + signal):
                return "chitchat"
        scores = {intent: 0 for intent in INTENT_PATTERNS if intent != "chitchat"}
        for intent, signals in INTENT_PATTERNS.items():
            if intent == "chitchat":
                continue
            for signal in signals:
                if signal in text:
                    scores[intent] += 1
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "general"

    def _extract_entities(self, text: str) -> List[str]:
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        entities = [q[0] or q[1] for q in quoted]
        caps = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
        entities += [c for c in caps if c not in ("The", "This", "That", "What", "How")]
        return list(dict.fromkeys(entities))[:6]

    def _extract_keywords(self, text: str) -> List[str]:
        stopwords = {"a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
                     "of", "and", "or", "but", "i", "me", "my", "you", "can", "do"}
        words = re.findall(r'\b\w{3,}\b', text)
        return [w for w in words if w not in stopwords][:10]

    def _assess_complexity(self, text: str) -> str:
        for signal in COMPLEXITY_SIGNALS["complex"]:
            if signal in text:
                return "complex"
        for signal in COMPLEXITY_SIGNALS["moderate"]:
            if signal in text:
                return "moderate"
        return "simple"

    def _extract_file_paths(self, text: str) -> List[str]:
        return re.findall(r'[\w./\\-]+\.(?:txt|json|csv|py|md|log|yaml|yml)', text)

    def _extract_questions(self, text: str) -> List[str]:
        return [s.strip() + "?" for s in re.split(r'\?', text) if len(s.strip()) > 5]
