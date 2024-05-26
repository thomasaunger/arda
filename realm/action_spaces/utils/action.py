class Action:

    def __new__(cls, action, value):
        assert isinstance(action, str)
        assert isinstance(value, int)

        class_attributes = {
            "type": property(lambda self: getattr(self, "_type")),
            "_type": None,
        }

        return type(f"Action{action}", (int,), class_attributes)(value)
