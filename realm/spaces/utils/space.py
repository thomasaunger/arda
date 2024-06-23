class Space:

    def __init__(self, int_dtype, np_random, radius):
        assert 0 <= radius, "Space radius must be non-negative"

        self._int_dtype = int_dtype
        self._np_random = np_random
        self._radius    = radius
    
    @property
    def int_dtype(self):
        return self._int_dtype
    
    @property
    def np_random(self):
        return self._np_random
    
    @property
    def radius(self):
        return self._radius
    
    @property
    def length(self):
        return 2*self.radius + 1
