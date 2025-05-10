"""A module to split a large number into its prime factors using the Chinese Remainder Theorem (CRT).
"""

import util.number_theory as nbtheory
from util.ntt import NTTContext


class CRTContext:

    """An instance of Chinese Remainder Theorem parameters.

    We split a large number into its prime factors.

    Attributes:
        poly_degree (int): Polynomial ring degree.
        primes (list): List of primes.
        modulus (int): Large modulus, product of all primes.
    """

    def __init__(self, num_primes, prime_size, poly_degree):
        """Inits CRTContext with a list of primes.

        Args:
            num_primes (int): Number of primes.
            prime_size (int): Minimum number of bits in primes.
            poly_degree (int): Polynomial degree of ring.
        """
        self.poly_degree = poly_degree
        self.generate_primes(num_primes, prime_size, mod=2*poly_degree)
        self.generate_ntt_contexts()

        self.modulus = 1
        for prime in self.primes:
            self.modulus *= prime

        self.precompute_crt()

    def generate_primes(self, num_primes, prime_size, mod):
        """Generates primes that are 1 (mod M), where M is twice the polynomial degree.

        Args:
            num_primes (int): Number of primes.
            prime_size (int): Minimum number of bits in primes.
            mod (int): Value M (must be a power of two) such that primes are 1 (mod M).
        """
        # NOTE: primes are chosen 1 mod M for some reason related to NTT
        # Initialize list of primes
        self.primes = [1] * num_primes
        # NOTE: I believe mod is related to how NTT maps polynomials of degree 'n' to values at '2n' roots of unity
        # NOTE: Notice that mod/M = 2n here
        # Start with a prime number that is 1 mod M (I don't really understand this), right now all I know is we start with a prime that is the minimum size + 1
        # So 2^prime_size mod M = 0 Why? Well M is 2 * poly_degree. Since poly_degree is a power of 2 (e.g 2 ^x), M = 2 ^ (x + 1).
        # Now dividing a power of by another power of 2 will give a remainder of 0 as long as the power of 2 we are dividing by is smaller than the power of 2 we are dividing
        # So I guess when choosing prime_size we must ensure it is bigger than poly_degree + 1 (otherwise the remainder here will be 2^prime_size)
        # By adding 1 we go from 2^prime_size mod M = 0 to (2^prime_size + 1) mod M = 1
        possible_prime = (1 << prime_size) + 1
        # NOTE: Now we have explained why the initial prime is 1 mod M, we now need to understand why it needed to be 1 mod M
        # NOTE: So notice that a mth/2nth root of unity is a value that when taken to the mth/2nth power then taken mod whatever factor will be chosen below should be 1
        # NOTE: Now the initia
        for i in range(num_primes):
            # NOTE: Keep adding M to the prime number until we find a prime number, I believe the number must be prime because of a condition related to CRT
            # TODO: Recheck the CRT condition to understand why the factor must be a prime number, I remember it having to be coprime not prime
            # NOTE: Since we keep adding M, we are guaranteed to be 1 mod M
            possible_prime += mod
            while not nbtheory.is_prime(possible_prime):
                possible_prime += mod
            self.primes[i] = possible_prime

    def generate_ntt_contexts(self):
        """Generates NTTContexts for each primes.
        """
        # TODO: Understand how NTT is constructed since this will be passed to the NTT kernel I believe as NTT context
        self.ntts = []
        self.generators = []
        self.inv_generators = []
        for prime in self.primes:
            ntt = NTTContext(self.poly_degree, prime)
            self.ntts.append(ntt)
            self.generators.append(ntt.generator)
            self.inv_generators.append(ntt.inv_generator)

    def precompute_crt(self):
        """Perform precomputations required for switching representations.
        """
        # TODO: Examine later what this function is doing
        num_primes = len(self.primes)
        self.crt_vals = [1] * num_primes
        self.crt_inv_vals = [1] * num_primes
        for i in range(num_primes):
            self.crt_vals[i] = self.modulus // self.primes[i]
            self.crt_inv_vals[i] = nbtheory.mod_inv(
                self.crt_vals[i], self.primes[i])

    def crt(self, value):
        """Transform value to CRT representation.

        Args:
            value (int): Value to be transformed to CRT representation.
            primes (list): List of primes to use for CRT representation.
        """
        # TODO: We need to parallelize this function
        # NOTE: Going from big modulus to CRT/RNS representation
        return [value % p for p in self.primes]

    def reconstruct(self, values):
        """Reconstructs original value from vals from the CRT representation to the regular representation.

        Args:
            values (list): List of values which are x_i (mod p_i).
        """
        # TODO: We need to parallelize this function
        assert len(values) == len(self.primes)
        regular_rep_val = 0

        for i in range(len(values)):
            intermed_val = (int(values[i]) * self.crt_inv_vals[i]) % self.primes[i]
            intermed_val = (intermed_val * self.crt_vals[i]) % self.modulus
            regular_rep_val += intermed_val
            regular_rep_val %= self.modulus

        return regular_rep_val
