class Tiling:
    
    def __init__(self, length, np_random, int_dtype):
        assert length >= 0, "Space length must be non-negative"

        self.length    = length
        self.np_random = np_random
        self.int_dtype = int_dtype
