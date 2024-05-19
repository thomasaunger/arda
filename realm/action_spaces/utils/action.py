class Action:

    def __new__(cls, action, value):
        assert isinstance(action, str)
        assert isinstance(value, int)

        return type(f"Action{action}", (int,), {})(value)
