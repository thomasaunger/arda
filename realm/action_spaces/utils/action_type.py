from .action import Action


class ActionType:

    def __new__(cls, action_type, value, actions):
        assert isinstance(action_type, str)
        assert isinstance(value, int)

        class_attributes = {}
        for i, action in enumerate(actions):
            assert isinstance(action, str)
            ACTION = action.upper()
            _ACTION = f"_{ACTION}"

            class_attributes[ACTION]  = property(lambda self, _ACTION=_ACTION: getattr(self, _ACTION))
            class_attributes[_ACTION] = Action(action, i)
        
        instance = type(f"Action{action_type}", (int,), class_attributes)(value)

        for i, action in enumerate(actions):
            ACTION = action.upper()
            _ACTION = f"_{ACTION}"

            class_attributes[_ACTION]._type = instance

        return instance
