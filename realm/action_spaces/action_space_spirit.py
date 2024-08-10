from .utils import ActionSpace


class ActionSpaceSpirit(ActionSpace):
    
    def __new__(cls, num_positions=0, num_symbols=0):
        return super().__new__(cls,
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
                ],
                **{f"Position_{m}": [f"Symbol_{n}" for n in range(num_symbols)] for m in range(num_positions)}
            }
        )
