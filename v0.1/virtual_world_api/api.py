"""Public v0.1 contract. Python 3.9+, standard library only.

ResetRequest is runner-only. A policy receives a detached Frame, never an env.
The runner owns identity/step fields; the model prompt should contain only the
declared observation, action menu and past public feedback.
"""
from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

SCHEMA_VERSION = '0.1.0'


class WorldError(ValueError):
    """Rejected operation; environment state and RNG must remain unchanged."""
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)

    def to_dict(self) -> Dict[str, str]:
        return {'code': self.code, 'message': str(self)}


class JsonRecord:
    def to_dict(self) -> Dict[str, Any]:
        # asdict recursively copies all nested containers.
        try:
            value = asdict(self)
            validate_json_value(value)
            return value
        except (RecursionError, TypeError) as exc:
            raise WorldError('INVALID_REQUEST', 'record must contain only acyclic JSON values') from exc


def validate_json_value(value, seen=None):
    """Reject Python-only values and NaN/Infinity before JSON Schema validation."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) not in (dict, list):
        raise WorldError('INVALID_REQUEST', 'non-JSON or nonfinite value')
    seen = set() if seen is None else seen
    if id(value) in seen:
        raise WorldError('INVALID_REQUEST', 'cyclic JSON container')
    seen.add(id(value))
    try:
        if type(value) is dict:
            if any(type(k) is not str for k in value):
                raise WorldError('INVALID_REQUEST', 'JSON object keys must be strings')
            children = value.values()
        else:
            children = value
        for child in children:
            validate_json_value(child, seen)
    finally:
        seen.remove(id(value))


@dataclass(frozen=True)
class ResetRequest(JsonRecord):
    world_id: str
    world_version: str
    episode_id: str
    seed: int
    max_steps: int
    reveal_world_id: bool = True


@dataclass(frozen=True)
class ActionRequest(JsonRecord):
    episode_id: str
    expected_step: int
    request_id: str
    action_id: str


@dataclass(frozen=True)
class Observation(JsonRecord):
    text: str
    facts: Dict[str, Any]


@dataclass(frozen=True)
class Action(JsonRecord):
    id: str
    label: str


@dataclass(frozen=True)
class Feedback(JsonRecord):
    transition_id: str
    from_step: int
    action_id: str
    reward: Optional[float]
    events: List[str]


@dataclass(frozen=True)
class Frame(JsonRecord):
    schema_version: str
    world_id: Optional[str]
    world_version: Optional[str]
    episode_id: str
    step_index: int
    observation: Observation
    actions: List[Action]
    last_feedback: Optional[Feedback]
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@runtime_checkable
class WorldEnv(Protocol):
    """Single subject, discrete enumerated actions, synchronous feedback.

    reset creates a NEW environment episode, never changes model state.
    observe is a detached, side-effect-free read.
    step accepts one current observation's action; request_id is episode-scoped.
    close is idempotent; all other calls then raise CLOSED.
    """
    def reset(self, request: ResetRequest) -> Frame: ...
    def observe(self) -> Frame: ...
    def step(self, request: ActionRequest) -> Frame: ...
    def close(self) -> None: ...
