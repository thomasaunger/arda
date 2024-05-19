from .utils import ActionSpace


class ActionSpaceSpirit(ActionSpace):
    
    def __new__(cls):
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
                ]
            }
        )
