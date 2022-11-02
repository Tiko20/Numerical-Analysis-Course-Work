import numpy
from numpy import pi as PI

def differences(x, y):

    if not isinstance(x, numpy.ndarray):
        raise TypeError(f"expected x to be type of numpy.ndarray, but got {type(x)}")
    
    if not isinstance(y, numpy.ndarray):
        raise TypeError(f"expected x to be type of numpy.ndarray, but got {type(y)}")
    
    if x.ndim != 1:
        raise ValueError(f"expected x to have shape [N], but got {x.shape}")

    if y.ndim != 1:
        raise ValueError(f"expected y to have shape [N], but got {y.shape}")

    if x.shape != y.shape:
        raise ValueError(f"expected x and y to have same size, but got {x.shape} and {y.shape}")

    if numpy.unique(x).shape != x.shape:
        raise ValueError(f"non unique x values")

    result = [x, y]

    for it in range(x.shape[0] - 1):
        temp = numpy.diff(result[-1])/(x[1 + it:] - x[:-1 - it])
        result.append(temp)

    return result

def get_optimal_nodes(n, left=-1, right=1):

    if not isinstance(n, int):
        raise TypeError(f"expected n to be type of int, but got {type(n)}")

    if not (2 <= n <= 1000):
        raise ValueError(f"expected n to be in range [2, 1000], but got {n}")

    if not isinstance(left, (int, float)):
        raise TypeError(f"expected left tp be type of int or float, but got {type(left)}")

    if not isinstance(right, (int, float)):
        raise TypeError(f"expected right tp be type of int or float, but got {type(right)}")

    if not (left < right):
        raise ValueError(f"expected left < right")

    result = (right - left) / 2 * numpy.cos(PI / (2 * n) + PI * numpy.arange(n) / n) +\
             (right + left) / 2

    return result[::-1]

def lagrange_interpolation(x, y, value=None, left=None, right=None, density=10000):

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

    if (left is None) ^ (right is None):
        raise ValueError("left and right should be None or not None simultaneously")

    if left is None:
        left, right = x.min(), x.max()

    if not (left < right):
        raise ValueError(f"expected left < right")

    if not (x.shape[0] <= density <= 10000):
        raise ValueError(f"expected density to be in range [x.shape, 10000], but got value {density}")
    
    mask = ~numpy.diag([True]*x.shape[0])

    if value is None:

        t = numpy.linspace(left, right, density)

        result = numpy.prod((t[:, None, None] - numpy.tile(x, (x.shape[0], 1))[mask].reshape(-1, x.shape[0] - 1))/
            (numpy.tile(x, (x.shape[0], 1))[~mask].reshape(-1, 1) - numpy.tile(x, (x.shape[0], 1))[mask].reshape(-1, x.shape[0] - 1)), axis=-1) @ y

        return t, result
    
    else:
        if not isinstance(value, (int, float)):
            raise TypeError(f"expected value to be type of int or float, but got {type(value)}")
        
        t = numpy.array([value])
        
        result = numpy.prod((t[:, None, None] - numpy.tile(x, (x.shape[0], 1))[mask].reshape(-1, x.shape[0] - 1))/
            (numpy.tile(x, (x.shape[0], 1))[~mask].reshape(-1, 1) - numpy.tile(x, (x.shape[0], 1))[mask].reshape(-1, x.shape[0] - 1)), axis=-1) @ y
        
        return result
        
def newton_interpolation(x, y, value=None, left=None, right=None, density=10000):

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

    if (left is None) ^ (right is None):
        raise ValueError("left and right should be None or not None simultaneously")

    if left is None:
        left, right = x.min(), x.max()

    if not (left < right):
        raise ValueError(f"expected left < right")

    if not (x.shape[0] <= density <= 10000):
        raise ValueError(f"expected density to be in range [x.shape, 10000], but got value {density}")

    diffs = differences(x, y)


    if value is None:

        t = numpy.linspace(left, right, density)
        result = numpy.zeros(t.shape[0])

        for idx, diff in enumerate(diffs[1:]): 
            result += diff[0] * numpy.prod(t[:, None] - x[:idx], axis=-1)

        return t, result
    else:

        if not isinstance(value, (int, float)):
            raise TypeError(f"expected value to be type of int or float, but got {type(value)}")
        
        t = numpy.array([value])
        result = 0

        for idx, diff in enumerate(diffs[1:]): 
            t_0 += diff[0] * numpy.prod(t[:, None] - x[:idx], axis=-1)
        return result
