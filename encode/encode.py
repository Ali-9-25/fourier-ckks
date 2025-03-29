from math import sin, cos, pi
import numpy as np
from numpy.polynomial import Polynomial


def pi_inverse(z: np.array) -> np.array:
    """Expands a vector of C^{N/2} by expanding it with its
    complex conjugate."""

    z_conjugate = z[::-1]
    z_conjugate = [np.conjugate(x) for x in z_conjugate]
    return np.concatenate([z, z_conjugate])


def vandermonde(xi: np.complex128, M: int) -> np.array:
    """Computes the Vandermonde matrix from a m-th root of unity."""

    N = M // 2
    matrix = []
    # We will generate each row of the matrix
    for i in range(N):
        # For each row we select a different root
        root = xi ** (2 * i + 1)
        row = []

        # Then we store its powers
        for j in range(N):
            row.append(root ** j)
        matrix.append(row)
    return matrix


def sigma_inverse(b: np.array, M) -> Polynomial:
    """Encodes the vector b in a polynomial using an M-th root of unity."""
    xi = np.exp(2 * np.pi * 1j /
                M)  # Mth root of unity which will be used as a basis for our computation
    # First we create the Vandermonde matrix
    A = vandermonde(xi, M)

    # Then we solve the system
    coeffs = np.linalg.solve(A, b)

    # Finally we output the polynomial
    p = Polynomial(coeffs)

    # We round it afterwards due to numerical imprecision
    coef = np.round(np.real(p.coef)).astype(int)
    return Polynomial(coef)


def create_sigma_R_basis(xi, M):
    """Creates the basis (sigma(1), sigma(X), ..., sigma(X** N-1))."""
    return np.array(vandermonde(xi, M)).T


def compute_basis_coordinates(z, sigma_R_basis):
    """Computes the coordinates of a vector with respect to the orthogonal lattice basis."""
    output = np.array([np.real(np.vdot(z, b) / np.vdot(b, b))
                      for b in sigma_R_basis])
    return output


def round_coordinates(coordinates):
    """Gives the integral rest."""
    coordinates = coordinates - np.floor(coordinates)
    return coordinates


def coordinate_wise_random_rounding(coordinates):
    """Rounds coordinates randonmly."""
    r = round_coordinates(coordinates)
    f = np.array([np.random.choice([c, c-1], 1, p=[1-c, c])
                 for c in r]).reshape(-1)

    rounded_coordinates = coordinates - f
    rounded_coordinates = [int(coeff) for coeff in rounded_coordinates]
    return rounded_coordinates


def sigma_R_discretization(z, sigma_R_basis):
    """Projects a vector on the lattice using coordinate wise random rounding."""
    coordinates = compute_basis_coordinates(z, sigma_R_basis)

    rounded_coordinates = coordinate_wise_random_rounding(coordinates)
    y = np.matmul(sigma_R_basis.T, rounded_coordinates)
    return y


def encode(z: np.array, scale, M, xi) -> Polynomial:
    """Encodes a vector by expanding it first to H,
    scale it, project it on the lattice of sigma(R), and performs
    sigma inverse.
    """
    sigma_R_basis = create_sigma_R_basis(xi, M)
    pi_z = pi_inverse(z)
    scaled_pi_z = scale * pi_z
    rounded_scale_pi_zi = sigma_R_discretization(scaled_pi_z, sigma_R_basis)
    p = sigma_inverse(rounded_scale_pi_zi, M)

    # We round it afterwards due to numerical imprecision
    coef = np.round(np.real(p.coef)).astype(int)
    p = Polynomial(coef)
    return p
