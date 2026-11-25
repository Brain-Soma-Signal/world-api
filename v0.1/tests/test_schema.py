import copy
import json
import unittest
from pathlib import Path

from virtual_world_api import ActionRequest, Observation, ResetRequest, WorldError
from virtual_world_api.api import validate_json_value
from virtual_world_api.mock import DemoWorld, WORLD_VERSION

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None

ROOT = Path(__file__).resolve().parents[1]


class JsonValueTests(unittest.TestCase):
    def test_python_only_and_nonfinite_values_rejected(self):
        cyclic = []
        cyclic.append(cyclic)
        for bad in (float('nan'), float('inf'), -float('inf'), (1, 2), {1: 'not-string'}, {1, 2}, cyclic):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(WorldError):
                    validate_json_value(bad)
        for bad in ({'x': float('nan')}, {'x': {1, 2}}, {1: 'value'}):
            with self.assertRaises(WorldError):
                Observation('test', bad).to_dict()
        shared = {'a': 1}
        validate_json_value({'x': shared, 'y': shared, 'nested': [True, None, 0.5]})


@unittest.skipIf(Draft202012Validator is None, 'optional jsonschema dependency not installed')
class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / 'world.schema.json').read_text())
        Draft202012Validator.check_schema(cls.schema)

    def validator(self, definition='Frame'):
        schema = dict(self.schema, **{'$ref': '#/$defs/' + definition})
        return Draft202012Validator(schema)

    def test_real_public_trace_and_requests_validate(self):
        config = ResetRequest('demo.ledger', WORLD_VERSION, 'schema-example', 7, 4)
        self.validator('ResetRequest').validate(config.to_dict())
        env = DemoWorld()
        frame = env.reset(config)
        self.validator().validate(frame.to_dict())
        for i, action in enumerate(('inspect', 'wait', 'choose_left')):
            request = ActionRequest(config.episode_id, i, 'req-' + str(i), action)
            self.validator('ActionRequest').validate(request.to_dict())
            frame = env.step(request)
            validate_json_value(frame.to_dict())
            self.validator().validate(frame.to_dict())

    def test_bad_frames_rejected(self):
        frame = DemoWorld().reset(ResetRequest('demo.ledger', WORLD_VERSION, 'e', 0, 4)).to_dict()
        changes = [
            {'extra': True}, {'schema_version': '1.0.0'}, {'episode_id': ''},
            {'world_id': None}, {'step_index': True}, {'episode_id': 'x' * 129},
            {'terminated': True, 'truncated': True}, {'terminated': True},
            {'step_index': 1}, {'observation': {'text': '', 'facts': []}},
        ]
        for change in changes:
            bad = dict(copy.deepcopy(frame), **change)
            with self.subTest(change=change):
                self.assertFalse(self.validator().is_valid(bad))

    def test_bad_requests_rejected_and_json_examples_validate(self):
        config = ResetRequest('demo.ledger', WORLD_VERSION, 'e', 0, 4).to_dict()
        for change in ({'seed': True}, {'seed': -1}, {'max_steps': 0}, {'max_steps': 100001}, {'extra': 1}):
            self.assertFalse(self.validator('ResetRequest').is_valid(dict(config, **change)))
        for path in (ROOT / 'examples').glob('*.json'):
            definition = 'ResetRequest' if path.stem == 'reset_request' else 'ActionRequest' if path.stem == 'action_request' else 'Frame'
            obj = json.loads(path.read_text())
            validate_json_value(obj)
            self.validator(definition).validate(obj)


if __name__ == '__main__':
    unittest.main()
