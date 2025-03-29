from math import sin, cos, pi
import numpy as np
from numpy.polynomial import Polynomial


def sigmaf(p: Polynomial, M, xi) -> np.array:
    """Decodes a polynomial by applying it to the M-th roots of unity."""

    outputs = []
    N = M // 2

    # We simply apply the polynomial on the roots
    for i in range(N):
        root = xi ** (2 * i + 1)
        output = p(root)
        outputs.append(output)
    return np.array(outputs)


def pi(z: np.array, M) -> np.array:
    """Projects a vector of H into C^{N/2}."""

    N = M // 4
    return z[:N]


def decode(p: Polynomial, scale, M, xi) -> np.array:
    """Decodes a polynomial by removing the scale, 
    evaluating on the roots, and project it on C^(N/2)"""
    rescaled_p = p / scale
    z = sigmaf(rescaled_p, M, xi)
    pi_z = pi(z, M)
    return pi_z
