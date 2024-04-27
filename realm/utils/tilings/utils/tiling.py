class Tiling:
    
    def __init__(self, length, int_dtype):
        assert length >= 0, "Surface length must be non-negative"

        self.length    = length
        self.int_dtype = int_dtype
