import copy
import json
import unittest
from dataclasses import replace

from virtual_world_api import ActionRequest, ResetRequest, WorldEnv, WorldError
from virtual_world_api.mock import DemoWorld, WORLD_VERSION


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.env = DemoWorld()
        self.config = ResetRequest('demo.ledger', WORLD_VERSION, 'episode-one', 7, 8)
        self.initial = self.env.reset(self.config)

    def req(self, action='inspect', step=0, rid='req-one', episode='episode-one'):
        return ActionRequest(episode, step, rid, action)

    def rejected(self, code, operation):
        before = self.env.observe()
        trace = self.env.private_trace()
        rng = self.env._rng.getstate()  # test harness only, never actor code
        with self.assertRaises(WorldError) as raised:
            operation()
        self.assertEqual(code, raised.exception.code)
        self.assertEqual(before, self.env.observe())
        self.assertEqual(trace, self.env.private_trace())
        self.assertEqual(rng, self.env._rng.getstate())

    def test_protocol_and_feedback_time(self):
        self.assertIsInstance(self.env, WorldEnv)
        self.assertEqual(0, self.initial.step_index)
        self.assertIsNone(self.initial.last_feedback)
        frame = self.env.step(self.req())
        self.assertEqual(1, frame.step_index)
        self.assertEqual(0, frame.last_feedback.from_step)
        self.assertEqual('inspect', frame.last_feedback.action_id)
        self.assertIsNone(frame.last_feedback.reward)
        self.assertEqual(frame, self.env.observe())
        zero = self.env.step(self.req('wait', 1, 'req-two'))
        self.assertEqual(0.0, zero.last_feedback.reward)
        self.assertIsNotNone(zero.last_feedback.reward)
        self.assertNotEqual(frame.last_feedback.transition_id, zero.last_feedback.transition_id)

    def test_observe_and_return_values_are_detached(self):
        rng = self.env._rng.getstate()
        observed = self.env.observe()
        observed.observation.facts['signal'] = 'tampered'
        observed.actions.clear()
        encoded = self.env.observe().to_dict()
        encoded['observation']['facts']['signal'] = 'tampered-too'
        self.assertEqual(self.initial, self.env.observe())
        self.assertEqual(rng, self.env._rng.getstate())
        frame = self.env.step(self.req())
        frame.last_feedback.events.append('injected')
        self.assertNotIn('injected', self.env.observe().last_feedback.events)
        self.assertNotIn('injected', self.env.step(self.req()).last_feedback.events)

    def test_rejections_do_not_mutate_any_state(self):
        self.rejected('INVALID_ACTION', lambda: self.env.step(self.req('unavailable')))
        self.rejected('STALE_STEP', lambda: self.env.step(self.req(step=1)))
        self.rejected('EPISODE_MISMATCH', lambda: self.env.step(self.req(episode='other')))
        self.rejected('INVALID_REQUEST', lambda: self.env.step(self.req(step=True)))
        self.rejected('INVALID_REQUEST', lambda: self.env.step(self.req(rid='')))
        self.rejected('INVALID_REQUEST', lambda: self.env.reset(replace(self.config, seed=True)))
        self.rejected('INVALID_REQUEST', lambda: self.env.reset(replace(self.config, max_steps=0)))
        self.rejected('INVALID_REQUEST', lambda: self.env.reset(replace(self.config, reveal_world_id=1)))
        self.rejected('UNKNOWN_WORLD', lambda: self.env.reset(replace(self.config, world_id='unknown')))
        self.rejected('VERSION_MISMATCH', lambda: self.env.reset(replace(self.config, world_version='2')))
        self.rejected('EPISODE_REUSED', lambda: self.env.reset(self.config))

    def test_idempotency_including_old_and_terminal_requests(self):
        first = self.env.step(self.req())
        final_request = self.req('choose_left', 1, 'terminal-request')
        final = self.env.step(final_request)
        trace = self.env.private_trace()
        self.assertEqual(final, self.env.step(final_request))
        self.assertEqual(first, self.env.step(self.req()))
        self.assertEqual(final, self.env.observe())
        self.assertEqual(trace, self.env.private_trace())
        self.rejected('REQUEST_CONFLICT', lambda: self.env.step(self.req('wait')))
        self.rejected('EPISODE_DONE', lambda: self.env.step(self.req('wait', 2, 'new')))
        self.rejected('STALE_STEP', lambda: self.env.step(self.req('wait', 0, 'new-stale')))

    def test_replay_invariant_to_reads_retries_rejections(self):
        clean = DemoWorld()
        clean.reset(self.config)
        for i, action in enumerate(('inspect', 'wait', 'inspect', 'choose_right')):
            request = self.req(action, i, 'r' + str(i))
            expected = clean.step(request)
            self.env.observe()
            self.rejected('STALE_STEP', lambda: self.env.step(self.req('inspect', i + 1, 'bad')))
            actual = self.env.step(request)
            self.assertEqual(expected, actual)
            self.assertEqual(actual, self.env.step(request))
        self.assertEqual(clean.private_trace(), self.env.private_trace())

    def test_hidden_truth_not_in_initial_public_frame_or_menu(self):
        frames = {}
        for seed in range(20):
            env = DemoWorld()
            frame = env.reset(replace(self.config, seed=seed))
            target = env.private_trace()[0]['hidden_target']
            frames[target] = frame
        self.assertEqual({0, 1}, set(frames))
        self.assertEqual(frames[0], frames[1])
        public = json.dumps(frames[0].to_dict())
        for forbidden in ('seed', 'hidden_target', 'oracle', 'runner_config', 'definition_sha256'):
            self.assertNotIn(forbidden, public)

    def test_natural_terminal_budget_and_terminal_precedence(self):
        for action in ('choose_left', 'choose_right'):
            env = DemoWorld()
            env.reset(replace(self.config, max_steps=1))
            frame = env.step(self.req(action))
            self.assertTrue(frame.terminated)
            self.assertFalse(frame.truncated)
            self.assertEqual([], frame.actions)
        env = DemoWorld()
        env.reset(replace(self.config, max_steps=1))
        frame = env.step(self.req('wait'))
        self.assertFalse(frame.terminated)
        self.assertTrue(frame.truncated)
        self.assertEqual([], frame.actions)
        self.assertEqual('none', frame.observation.facts['result'])

    def test_reset_new_episode_and_hidden_world_metadata(self):
        self.env.step(self.req())
        frame = self.env.reset(replace(self.config, episode_id='episode-two', reveal_world_id=False))
        self.assertIsNone(frame.world_id)
        self.assertIsNone(frame.world_version)
        self.assertEqual(0, frame.step_index)
        self.assertIsNone(frame.last_feedback)
        self.assertEqual('unknown', frame.observation.facts['signal'])
        self.rejected('EPISODE_MISMATCH', lambda: self.env.step(self.req()))
        self.rejected('EPISODE_REUSED', lambda: self.env.reset(self.config))

    def test_not_ready_and_close(self):
        env = DemoWorld()
        for operation in (env.observe, lambda: env.step(self.req())):
            with self.assertRaises(WorldError) as raised:
                operation()
            self.assertEqual('NOT_READY', raised.exception.code)
        env.close()
        env.close()
        for operation in (env.observe, lambda: env.step(self.req()), lambda: env.reset(self.config)):
            with self.assertRaises(WorldError) as raised:
                operation()
            self.assertEqual('CLOSED', raised.exception.code)


if __name__ == '__main__':
    unittest.main()
