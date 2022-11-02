import numpy

class LagrangeInterpolation:
    def __init__(self, x, y):

        if not isinstance(x, numpy.ndarray):
            raise TypeError(f"expected x to be type of numpy.ndarray, but got {type(x)}")

        if not isinstance(y, numpy.ndarray):
            raise TypeError(f"expected x to be type of numpy.ndarray, but got {type(y)}")

        if numpy.any(x[:-1] >= x[1:]):
            raise ValueError("expected x to be strictly monotonic increasing")

        if x.ndim != 1:
            raise ValueError(f"expected x to have shape [N], but got {x.shape}")

        if y.ndim != 1:
            raise ValueError(f"expected y to have shape [N], but got {y.shape}")

        if x.shape != y.shape:
            raise ValueError(f"expected x and y to have same size, but got {x.shape} and {y.shape}")

        if numpy.unique(x).shape != x.shape:
            raise ValueError(f"non unique x values")
        
        self.x = x
        self.y = y
        self.mask = ~numpy.diag([True]*x.shape[0])
        
    def __call__(self, value):
        if not isinstance(value, (int, float, numpy.ndarray)):
            raise TypeError(f"expected x to be scalar of type int or float, or numpy.ndarray, but got {type(x)}")
            
        if not isinstance(value, numpy.ndarray):
            value = numpy.array([value])
            
        return numpy.prod((value[:, None, None] - numpy.tile(self.x, (self.x.shape[0], 1))[self.mask].reshape(-1, self.x.shape[0] - 1))/
                    (numpy.tile(self.x, (self.x.shape[0], 1))[~self.mask].reshape(-1, 1) - 
                        numpy.tile(self.x, (self.x.shape[0], 1))[self.mask].reshape(-1, self.x.shape[0] - 1)), axis=-1) @ self.y