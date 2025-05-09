import math

# ---------------------------------------------------------------------
# -- Helpers for power-of-two check -----------------------------------
# ---------------------------------------------------------------------

def _is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two (and non-zero)."""
    return n and (n & (n - 1) == 0)

# ---------------------------------------------------------------------
# -- 1-D NTT / INTT implementations -----------------------------------
# ---------------------------------------------------------------------

def _ntt_radix2(x, mod, root):
    """
    Recursive radix-2 Cooley-Tukey NTT.
    Assumes len(x) is a power of two and root is a primitive n-th root of unity mod `mod`.
    """
    n = len(x)
    if n == 1:
        return [x[0] % mod]
    # Split into even/odd
    x_even = _ntt_radix2(x[0::2], mod, (root*root) % mod)
    x_odd  = _ntt_radix2(x[1::2], mod, (root*root) % mod)
    # Twiddle factors
    factor = 1
    result = [0] * n
    for k in range(n // 2):
        t = (factor * x_odd[k]) % mod
        result[k] = (x_even[k] + t) % mod
        result[k + n // 2] = (x_even[k] - t) % mod
        factor = (factor * root) % mod
    return result


def _dntt(x, mod, primitive_root):
    """
    Direct O(N^2) discrete NTT. Requires that n divides mod-1.
    """
    n = len(x)
    if n == 0:
        return []
    if (mod - 1) % n != 0:
        raise ValueError(f"Length {n} does not divide mod-1; no primitive root of unity exists")
    # Compute primitive n-th root
    root = pow(primitive_root, (mod - 1) // n, mod)
    # Precompute powers
    W = [[pow(root, (i * j) % n, mod) for j in range(n)] for i in range(n)]
    return [sum((W[k][j] * x[j]) for j in range(n)) % mod for k in range(n)]


def NTT1D(x, mod=998244353, primitive_root=3):
    """
    Compute the 1-D NTT of `x` modulo `mod`.
    For lengths that are powers of two, uses radix-2 recursion;
    otherwise falls back to O(N^2) direct transform (provided n divides mod-1).
    """
    # Ensure sequence of ints
    a = [int(val) % mod for val in x]
    n = len(a)
    if n == 0:
        return []
    # Decide method
    if _is_power_of_two(n) and (mod - 1) % n == 0:
        root = pow(primitive_root, (mod - 1) // n, mod)
        return _ntt_radix2(a, mod, root)
    else:
        return _dntt(a, mod, primitive_root)


def INTT1D(A, mod=998244353, primitive_root=3):
    """
    Compute the inverse 1-D NTT.
    Uses inverse radix-2 recursion or direct inverse transform.
    """
    n = len(A)
    if n == 0:
        return []
    if (mod - 1) % n != 0:
        raise ValueError(f"Length {n} does not divide mod-1; cannot invert transform")
    # Compute inverses
    inv_n = pow(n, mod - 2, mod)
    root = pow(primitive_root, (mod - 1) // n, mod)
    inv_root = pow(root, mod - 2, mod)
    # Inverse transform via NTT with inv_root
    a = _ntt_radix2([int(val) % mod for val in A], mod, inv_root)
    # Scale by inv_n
    return [(val * inv_n) % mod for val in a]

# ---------------------------------------------------------------------
# -- 2-D NTT / INTT --------------------------------------------------
# ---------------------------------------------------------------------

def NTT2D(matrix, mod=998244353, primitive_root=3):
    """
    Compute the 2-D NTT by applying 1-D NTT to rows then columns.
    """
    # Convert to list of lists
    mat = [[int(val) % mod for val in row] for row in matrix]
    # NTT on rows
    temp = [NTT1D(row, mod, primitive_root) for row in mat]
    # NTT on columns
    # transpose
    t_t = list(map(list, zip(*temp)))
    t2   = [NTT1D(col, mod, primitive_root) for col in t_t]
    # transpose back
    return [list(row) for row in zip(*t2)]


def INTT2D(matrix, mod=998244353, primitive_root=3):
    """
    Compute the 2-D inverse NTT.
    """
    mat = [[int(val) % mod for val in row] for row in matrix]
    temp = [INTT1D(row, mod, primitive_root) for row in mat]
    t_t   = list(map(list, zip(*temp)))
    t2    = [INTT1D(col, mod, primitive_root) for col in t_t]
    return [list(row) for row in zip(*t2)]

# ---------------------------------------------------------------------
# -- Convolution utilities -------------------------------------------
# ---------------------------------------------------------------------

def linear_convolution_direct_mod(x, h, mod=998244353):
    """
    Direct convolution modulo `mod`.
    """
    n, m = len(x), len(h)
    y = [0] * (n + m - 1)
    for i in range(n):
        for j in range(m):
            y[i + j] = (y[i + j] + x[i] * h[j]) % mod
    return y


def linear_convolution_ntt(x, h, mod=998244353, primitive_root=3):
    """
    Convolution via NTT (requires N a power of two dividing mod-1).
    """
    n, m = len(x), len(h)
    size = n + m - 1
    # Next power of two >= size
    N = 1 << (size - 1).bit_length()
    if (mod - 1) % N != 0:
        raise ValueError(f"Transform length {N} not supported (must divide mod-1)")
    # pad
    a = x + [0] * (N - n)
    b = h + [0] * (N - m)
    # NTT-based conv
    A = NTT1D(a, mod, primitive_root)
    B = NTT1D(b, mod, primitive_root)
    C = [(A[i] * B[i]) % mod for i in range(N)]
    c = INTT1D(C, mod, primitive_root)
    return c[:size]
