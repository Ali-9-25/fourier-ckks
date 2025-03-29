from numpy.polynomial import Polynomial


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


def decrypt_poly(ct, sk, pol_modulus, q):
    plaintext = ct[1] * sk + ct[0]
    plaintext = reduce_polynomial(plaintext, pol_modulus, q)

    return plaintext
