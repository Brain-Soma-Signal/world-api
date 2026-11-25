"""Small deterministic interface exercise, no model load or training."""
import json
import uuid
from .api import ActionRequest, ResetRequest
from .mock import DemoWorld, WORLD_VERSION


def main():
    env = DemoWorld()
    try:
        frame = env.reset(ResetRequest(
            world_id='demo.ledger', world_version=WORLD_VERSION,
            episode_id=str(uuid.uuid4()), seed=7, max_steps=4,
        ))
        for action_id in ('inspect', 'wait', 'choose_left'):
            # An actual actor receives only frame, NOT env or the reset request.
            print(json.dumps(frame.to_dict(), ensure_ascii=False))
            request = ActionRequest(frame.episode_id, frame.step_index,
                                    str(uuid.uuid4()), action_id)
            frame = env.step(request)
            assert env.step(request) == frame  # same request: no second action
            if frame.done:
                break
        print(json.dumps(frame.to_dict(), ensure_ascii=False))
    finally:
        env.close()


if __name__ == '__main__':
    main()
