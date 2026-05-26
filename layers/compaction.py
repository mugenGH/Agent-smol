"""Conversation history compaction.

When the conversation_history list grows beyond a threshold, summarizes
old entries using the LLM so that future prompts don't balloon out of
control. Preserves the most recent N turns verbatim and replaces older
turns with a structured summary.
"""
import litellm


COMPACTION_PROMPT = """\
You are summarizing a conversation history for an AI agent. Compress the
exchanges below into a concise structured summary that preserves ALL
critical information the agent will need to continue the conversation.

Include:
1. Original task and intent
2. What was attempted (key tools/actions used)
3. What worked and what failed
4. Current state — what has been done, what is still missing
5. Any important facts, file paths, or values discovered

Be extremely concise. Preserve specific details (paths, values, errors).
Output only the summary text, no preamble.

Conversation to summarize:
{history}"""


class CompactionLayer:
    """Compacts conversation_history when it grows too large."""

    def __init__(self, cfg: dict):
        self.max_turns: int = cfg.get("max_turns", 20)
        self.keep_recent: int = cfg.get("keep_recent", 6)
        self._model_id: str = ""
        self._api_base: str = ""
        self._api_key: str = "not-needed"

    def set_model(self, model_id: str, api_base: str, api_key: str = "not-needed") -> None:
        self._model_id = model_id
        self._api_base = api_base
        self._api_key = api_key

    def needs_compaction(self, history: list) -> bool:
        return len(history) > self.max_turns

    def compact(self, history: list) -> list:
        """
        Return a new history list with old turns replaced by a summary entry.
        Falls back to simple truncation if the LLM call fails.
        """
        if len(history) <= self.keep_recent:
            return history

        old = history[:-self.keep_recent]
        recent = history[-self.keep_recent:]

        summary_text = self._llm_summarize(old)
        summary_entry = {
            "role": "system",
            "content": f"[CONTEXT SUMMARY — {len(old)} older turns compressed]\n{summary_text}",
        }
        return [summary_entry] + recent

    def _llm_summarize(self, history: list) -> str:
        """Call the LLM to produce a structured summary of the old turns."""
        if not self._model_id:
            return self._fallback_summarize(history)

        history_text = "\n".join(
            f"[{entry['role'].upper()}]: {str(entry['content'])[:300]}"
            for entry in history
        )
        try:
            response = litellm.completion(
                model=self._model_id,
                messages=[{
                    "role": "user",
                    "content": COMPACTION_PROMPT.format(history=history_text),
                }],
                api_base=self._api_base,
                api_key=self._api_key,
                max_tokens=600,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return self._fallback_summarize(history)

    def _fallback_summarize(self, history: list) -> str:
        """Deterministic fallback when LLM summarization fails."""
        lines = [f"Compressed {len(history)} turns:"]
        for entry in history:
            role = entry.get("role", "?").upper()
            content = str(entry.get("content", ""))[:120]
            lines.append(f"  [{role}]: {content}")
        return "\n".join(lines)
