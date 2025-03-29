import random
from numpy.polynomial import Polynomial
# from math import sin, cos, pi
import numpy as np


def Poly(coeffs):
    """
    Helper function to build polynomials, passing a dictionary of coefficients.
    For example, passing {0: 1, 1: -2, 2: 2} returns the polynomial
    2X**2 - 2X + 1
    """
    max_power = max(coeffs.keys())
    _coeffs = np.zeros(max_power + 1)

    for i, c in coeffs.items():
        _coeffs[i] = c

    return Polynomial(_coeffs)


def mod_on_coefficients(polynomial, modulo):
    """
    Apply the modulo on the coefficients of a polynomial.
    """
    coefs = polynomial.coef
    mod_coefs = []
    for c in coefs:
        mod_coefs.append(c % modulo)

    return Polynomial(mod_coefs)


def reduce_polynomial(polynomial, polynomial_modulous, q):
    """
    Reduce a polynomial modulo a cyclotomic polynomial and a modulus.
    """
    remainder = polynomial % polynomial_modulous  # polynomial modulo cyclotomic polynomial -> get the remainder
    # perfrom mod operation on coefficients
    return mod_on_coefficients(remainder, q)


def secret_key_pol(N, h):
    """
    Generate the secret key.
    N = degree of polynomial modulous
    h = number of non-zero coefficients in the polynomial generated
    """
    s = {}
    non_zero_indices = random.sample(range(N), h)

    # Assign ±1 randomly to the selected positions
    for idx in non_zero_indices:
        s[idx] = random.choice([-1, 1])

    return Poly(s)


def create_error_pol(N, Sigma, mu):
    """
    Generate an error polynomial.
    """
    e = np.random.normal(mu, Sigma, N)
    e_map = {}
    for i in range(len(e)):
        e_map[i] = round(e[i])

    return Poly(e_map)


def sample_pol(N, modulous):
    """
    Sample a polynomial.
    """
    coeffs = {}
    for i in range(N):
        c = np.random.randint(0, modulous, dtype=np.int64)
        coeffs[i] = c
    return Poly(coeffs)


def generate_keys(N, q, h, pol_modulus, mu=0, Sigma=3.2):

    # secret key
    sk = secret_key_pol(N, h)

    a = sample_pol(N, q)
    e = create_error_pol(N, Sigma, mu)

    b = (-a*sk)+e
    b = reduce_polynomial(b, pol_modulus, q)

    # public key
    pk = (b, a)

    return sk, pk


def encrypt_poly(m, N, pk, pol_modulus, q, Sigma=3.2, mu=0):
    e1 = create_error_pol(N, Sigma, mu)
    e2 = create_error_pol(N, Sigma, mu)

    coeffs = {}
    for i in range(0, N):
        coeffs[i] = np.random.randint(-1, 2)
    v = Poly(coeffs)

    ct0 = pk[0] * v + e1 + m
    ct0 = reduce_polynomial(ct0, pol_modulus, q)

    ct1 = pk[1] * v + e2
    ct1 = reduce_polynomial(ct1, pol_modulus, q)
    return ct0, ct1
