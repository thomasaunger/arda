from .action_space import ActionSpace


class Space:

    _action_space = ActionSpace(
        "Spirit",
        {
            "Turn": [
                "None",
                "Left",
                "Right",
            ],
            "Move": [
                "None",
                "Forward",
            ]
        }
    )

    def __init__(self, int_dtype, np_random, length):
        assert length >= 0, "Space length must be non-negative"

        self._int_dtype = int_dtype
        self._np_random = np_random
        self._length    = length
    
    @property
    def action_space(self):
        return self._action_space
    
    @property
    def int_dtype(self):
        return self._int_dtype
    
    @property
    def np_random(self):
        return self._np_random
    
    @property
    def length(self):
        return self._length
