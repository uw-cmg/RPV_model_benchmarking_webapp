import numpy as np


def poly(x, *coefficients):
    values = np.asarray(x, dtype=float)
    result = np.zeros_like(values, dtype=float)
    for power, coefficient in enumerate(coefficients):
        result = result + coefficient * np.power(values, power)
    return result
