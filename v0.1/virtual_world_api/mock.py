"""Original synthetic contract fixture, NOT an implemented novel world.

Three names are cosmetic variants of the SAME hidden-choice mechanism.
They must not be counted as three independent research environments.
"""
import copy
import hashlib
import json
import random
from dataclasses import replace
from typing import Dict, Optional, Tuple

from .api import (
    SCHEMA_VERSION, Action, ActionRequest, Feedback, Frame, Observation,
    ResetRequest, WorldError,
)

ENGINE_VERSION = 'mock-0.1.0'
WORLD_VERSION = '1.0.0'
SCENES = {
    'demo.ledger': '账房里有左右两只箱子，一只装有目标凭据。',
    'demo.workshop': '工坊里有左右两只盒子，一只装有目标零件。',
    'demo.dream': '房间里有左右两扇门，一扇通向目标房间。',
}
RULES = {
    'target': 'one fixed hidden side, sampled at reset',
    'inspect': 'public signal is correct with probability 0.9; reward undisclosed',
    'wait': 'no new signal; observed reward is zero',
    'choose': 'left/right choice terminates; observed reward is +1 or -1',
    'budget': 'external step limit truncates unless a natural terminal occurred',
    'rendering': 'deterministic templates; no online language model',
}
DEFINITION_HASH = hashlib.sha256(json.dumps(
    {'rules': RULES, 'scenes': SCENES, 'version': WORLD_VERSION},
    ensure_ascii=False, sort_keys=True,
).encode('utf-8')).hexdigest()


def identifier(value, name):
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise WorldError('INVALID_REQUEST', name + ' must be a nonempty string of at most 128 characters')


class DemoWorld:
    def __init__(self):
        self._closed = False
        self._frame: Optional[Frame] = None
        self._config: Optional[ResetRequest] = None
        self._seen_episodes = set()
        self._cache: Dict[str, Tuple[ActionRequest, Frame]] = {}
        self._trace = []
        self._rng = None
        self._target = None
        self._signal = 'unknown'
        self._inspections = 0

    def _open(self):
        if self._closed:
            raise WorldError('CLOSED', 'environment is closed')

    def reset(self, request: ResetRequest) -> Frame:
        self._open()
        if not isinstance(request, ResetRequest):
            raise WorldError('INVALID_REQUEST', 'ResetRequest required')
        for name in ('world_id', 'world_version', 'episode_id'):
            identifier(getattr(request, name), name)
        if type(request.seed) is not int or not 0 <= request.seed <= 2**32 - 1:
            raise WorldError('INVALID_REQUEST', 'seed must be an integer in [0, 2**32-1]')
        if type(request.max_steps) is not int or not 1 <= request.max_steps <= 100000:
            raise WorldError('INVALID_REQUEST', 'max_steps must be an integer in [1, 100000]')
        if type(request.reveal_world_id) is not bool:
            raise WorldError('INVALID_REQUEST', 'reveal_world_id must be boolean')
        if request.world_id not in SCENES:
            raise WorldError('UNKNOWN_WORLD', 'unknown world_id')
        if request.world_version != WORLD_VERSION:
            raise WorldError('VERSION_MISMATCH', 'world version must be ' + WORLD_VERSION)
        if request.episode_id in self._seen_episodes:
            raise WorldError('EPISODE_REUSED', 'reset requires a fresh episode_id')
        # All validation precedes mutation. Seed stays on the runner side.
        rng = random.Random(request.seed)
        target = rng.randrange(2)
        frame = Frame(
            schema_version=SCHEMA_VERSION,
            world_id=request.world_id if request.reveal_world_id else None,
            world_version=WORLD_VERSION if request.reveal_world_id else None,
            episode_id=request.episode_id, step_index=0,
            observation=self._observation(request.world_id, 'unknown', 0, 'none'),
            actions=self._actions(), last_feedback=None,
            terminated=False, truncated=False,
        )
        self._config, self._rng, self._target = request, rng, target
        self._signal, self._inspections = 'unknown', 0
        self._frame = frame
        self._cache = {}
        self._seen_episodes.add(request.episode_id)
        self._trace = [{
            'kind': 'reset', 'engine_version': ENGINE_VERSION,
            'definition_sha256': DEFINITION_HASH,
            'runner_config': request.to_dict(), 'hidden_target': target,
            'public_frame': frame.to_dict(),
        }]
        return copy.deepcopy(frame)

    @staticmethod
    def _actions():
        # Same menu for both hidden targets: legal != guaranteed successful.
        return [Action('inspect', '观察线索'), Action('wait', '等待'),
                Action('choose_left', '选择左侧'), Action('choose_right', '选择右侧')]

    @staticmethod
    def _observation(world_id, signal, inspections, result):
        return Observation(
            text=SCENES[world_id] + ' 当前线索：' + signal + '。',
            facts={'signal': signal, 'inspection_count': inspections, 'result': result},
        )

    def observe(self) -> Frame:
        self._open()
        if self._frame is None:
            raise WorldError('NOT_READY', 'reset must be called first')
        return copy.deepcopy(self._frame)

    def step(self, request: ActionRequest) -> Frame:
        self._open()
        if self._frame is None:
            raise WorldError('NOT_READY', 'reset must be called first')
        if not isinstance(request, ActionRequest):
            raise WorldError('INVALID_REQUEST', 'ActionRequest required')
        for name in ('episode_id', 'request_id', 'action_id'):
            identifier(getattr(request, name), name)
        if type(request.expected_step) is not int or not 0 <= request.expected_step <= 100000:
            raise WorldError('INVALID_REQUEST', 'expected_step must be an integer in [0, 100000]')
        if request.episode_id != self._frame.episode_id:
            raise WorldError('EPISODE_MISMATCH', 'action belongs to another episode')
        # Check cached requests BEFORE done/stale checks, including final actions.
        if request.request_id in self._cache:
            previous, response = self._cache[request.request_id]
            if request != previous:
                raise WorldError('REQUEST_CONFLICT', 'request_id already has another payload')
            return copy.deepcopy(response)
        if request.expected_step != self._frame.step_index:
            raise WorldError('STALE_STEP', 'expected_step does not match current observation')
        if self._frame.done:
            raise WorldError('EPISODE_DONE', 'reset required after episode end')
        if request.action_id not in {a.id for a in self._frame.actions}:
            raise WorldError('INVALID_ACTION', 'action_id is not in the public action menu')

        # Work on local state/RNG; install only after construction succeeds.
        rng = random.Random(0)
        rng.setstate(self._rng.getstate())
        signal, inspections = self._signal, self._inspections
        terminated, result, reward = False, 'none', None
        if request.action_id == 'inspect':
            side = self._target if rng.random() < .9 else 1 - self._target
            signal = ('left', 'right')[side]
            inspections += 1
            events = ['观察到一条可能不可靠的线索：' + signal]
        elif request.action_id == 'wait':
            reward, events = 0.0, ['等待了一步，没有新线索。']
        else:
            chosen = int(request.action_id == 'choose_right')
            reward = 1.0 if chosen == self._target else -1.0
            terminated = True
            result = 'success' if chosen == self._target else 'failure'
            events = ['选择结果：' + result]
        next_step = self._frame.step_index + 1
        truncated = next_step >= self._config.max_steps and not terminated
        if truncated:
            events.append('外部步数预算已耗尽。')
        # Fixed-length deterministic ID avoids leaking the seed/private state.
        transition_id = hashlib.sha256(json.dumps(
            [request.episode_id, request.expected_step], ensure_ascii=False,
        ).encode('utf-8')).hexdigest()
        frame = replace(
            self._frame, step_index=next_step,
            observation=self._observation(self._config.world_id, signal, inspections, result),
            actions=[] if terminated or truncated else self._actions(),
            last_feedback=Feedback(transition_id, request.expected_step, request.action_id, reward, events),
            terminated=terminated, truncated=truncated,
        )
        record = {'kind': 'transition', 'request': request.to_dict(),
                  'hidden_target': self._target, 'public_frame': frame.to_dict()}
        self._rng, self._signal, self._inspections = rng, signal, inspections
        self._frame = frame
        self._cache[request.request_id] = (request, copy.deepcopy(frame))
        self._trace.append(record)
        return copy.deepcopy(frame)

    def private_trace(self):
        """Runner/evaluator only. NOT part of WorldEnv and NEVER sent to policy."""
        self._open()
        return copy.deepcopy(self._trace)

    def close(self) -> None:
        self._closed = True
