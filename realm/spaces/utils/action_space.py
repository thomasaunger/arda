import gym

from .action_type import ActionType


class ActionSpace:

    def __new__(cls, action_space, action_types):
        assert isinstance(action_space, str)
        assert isinstance(action_types, dict)

        class_attributes = {}
        for i, (action_type, q) in enumerate(action_types.items()):
            assert isinstance(action_type, str)
            ACTION_TYPE = action_type.upper()
            _ACTION_TYPE = f"_{ACTION_TYPE}"

            class_attributes[ACTION_TYPE]  = property(lambda self, _ACTION_TYPE=_ACTION_TYPE: getattr(self, _ACTION_TYPE))
            class_attributes[_ACTION_TYPE] = ActionType(action_type, i, q)

        return type(f"ActionSpace{action_space}", (gym.spaces.MultiDiscrete,), class_attributes)(tuple(len(action_type) for action_type in action_types.values()))
