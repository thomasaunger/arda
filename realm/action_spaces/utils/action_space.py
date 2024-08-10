import gym

from .action_type import ActionType


class ActionSpace:

    def __new__(cls, action_space, action_types, num_positions=0):
        assert isinstance(action_space, str)
        assert isinstance(action_types, dict)

        class_attributes = {}
        for i, (action_type, actions) in enumerate(action_types.items()):
            assert isinstance(action_type, str)
            ACTION_TYPE = action_type.upper()
            _ACTION_TYPE = f"_{ACTION_TYPE}"

            class_attributes[ACTION_TYPE]  = property(lambda self, _ACTION_TYPE=_ACTION_TYPE: getattr(self, _ACTION_TYPE))
            class_attributes[_ACTION_TYPE] = ActionType(action_type, i, actions)
        
        class_attributes["num_positions"]  = property(lambda self: getattr(self, "_num_positions"))
        class_attributes["_num_positions"] = num_positions

        return type(f"ActionSpace{action_space}", (gym.spaces.MultiDiscrete,), class_attributes)(tuple(len(action_type) for action_type in action_types.values()))
