"""Agent-to-Agent (A2A) protocol primitives.

Defines the message envelope, agent card, and hub that routes messages
between registered agents. Agents communicate through the hub using
A2AMessage objects and receive A2AResponse objects back.

The hub optionally re-raises specific exception types (e.g. StallError)
so callers can handle them without losing protocol semantics.
"""
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AgentCard:
    """Describes an agent's identity and capabilities for discovery."""
    name: str
    description: str
    capabilities: list[str]


@dataclass
class A2AMessage:
    """Message envelope sent between agents through the hub."""
    sender: str
    recipient: str
    task: str
    payload: dict = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None


@dataclass
class A2AResponse:
    """Response produced by a receiving agent."""
    message_id: str
    sender: str
    recipient: str
    result: str
    status: str  # "success" | "failed"
    artifacts: list[dict] = field(default_factory=list)


class AgentHub:
    """Routes A2A messages between registered agents.

    Agents register with an AgentCard (describing capabilities) and a handler
    callable that accepts A2AMessage and returns str.  All exchanges are logged.

    reraise: exception types that bubble through hub.send() unmodified.
    This is used to propagate StallError from executors without wrapping it
    in a failed A2AResponse.
    """

    def __init__(self, reraise: tuple = ()) -> None:
        self._reraise = reraise
        self._cards: dict[str, AgentCard] = {}
        self._handlers: dict[str, Callable[[A2AMessage], str]] = {}
        self._log: list[tuple[A2AMessage, A2AResponse]] = []

    def register(self, card: AgentCard, handler: Callable[[A2AMessage], str]) -> None:
        self._cards[card.name] = card
        self._handlers[card.name] = handler

    @property
    def agents(self) -> list[AgentCard]:
        return list(self._cards.values())

    def send(self, msg: A2AMessage) -> A2AResponse:
        """Dispatch msg to the target agent and return its response."""
        handler = self._handlers.get(msg.recipient)
        if handler is None:
            resp = A2AResponse(
                message_id=msg.message_id,
                sender="hub",
                recipient=msg.sender,
                result=f"No agent registered as '{msg.recipient}'.",
                status="failed",
            )
            self._log.append((msg, resp))
            return resp

        try:
            result = handler(msg)
            status = "success"
        except self._reraise:
            raise
        except Exception as exc:
            result = f"[{msg.recipient} error] {exc}"
            status = "failed"

        resp = A2AResponse(
            message_id=msg.message_id,
            sender=msg.recipient,
            recipient=msg.sender,
            result=result,
            status=status,
        )
        self._log.append((msg, resp))
        return resp

    def exchange_log(self) -> list[tuple[A2AMessage, A2AResponse]]:
        return list(self._log)

    def clear_log(self) -> None:
        self._log.clear()
