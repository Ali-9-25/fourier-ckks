import numpy as np
from ckks.ckks_parameters import CKKSParameters
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_encoder import CKKSEncoder
from fft.fft import FFT1D, IFFT1D


class FourierCKKS:
    """
    A helper class to perform FFT-based homomorphic encryption using the CKKS scheme.
    """

    def __init__(self,
                 poly_degree=1024,
                 ciph_modulus=(1 << 200),
                 big_modulus=(1 << 300),
                 scaling_factor=(1 << 15)):
        # Initialize CKKS parameters and keys
        self.params = CKKSParameters(
            poly_degree=poly_degree,
            ciph_modulus=ciph_modulus,
            big_modulus=big_modulus,
            scaling_factor=scaling_factor
        )
        keygen = CKKSKeyGenerator(self.params)
        self.public_key = keygen.public_key
        self.secret_key = keygen.secret_key
        self.relin_key = keygen.relin_key

        # Encoder / Encryptor / Decryptor / Evaluator
        self.encoder = CKKSEncoder(self.params)
        self.encryptor = CKKSEncryptor(self.params, self.public_key, self.secret_key)
        self.decryptor = CKKSDecryptor(self.params, self.secret_key)
        self.evaluator = CKKSEvaluator(self.params)

        # Data lengths
        self.data_len = poly_degree // 2
        self.max_message_length = self.data_len // 2

    def forward(self, message: np.ndarray) -> object:
        """
        Perform FFT, encode, and encrypt on a complex-valued message.

        :param message: 1D complex numpy array of length <= max_message_length
        :return: Encrypted ciphertext
        """
        if message.shape[0] > self.max_message_length:
            raise ValueError(f"Message length must be <= {self.max_message_length}")

        # Embed into vector of length data_len
        vec = np.zeros(self.data_len, dtype=complex)
        vec[:message.shape[0]] = message

        # FFT domain
        freq = FFT1D(vec)
        # Encode and encrypt
        pt = self.encoder.encode(freq, self.params.scaling_factor)
        ct = self.encryptor.encrypt(pt)
        return ct

    def backward(self, ciphertext: object) -> np.ndarray:
        """
        Decrypt, decode, and perform inverse FFT to recover the plaintext array.

        :param ciphertext: Encrypted ciphertext
        :return: 1D complex numpy array of length data_len
        """
        pt = self.decryptor.decrypt(ciphertext)
        freq = self.encoder.decode(pt)
        vec = IFFT1D(freq)
        return vec

    def cipher_add(self, ct1: object, ct2: object) -> object:
        """
        Homomorphically add two ciphertexts.
        """
        return self.evaluator.add(ct1, ct2)

    def cipher_conv(self, ct1: object, ct2: object) -> object:
        """
        Homomorphically multiply (convolve) two ciphertexts.
        """
        return self.evaluator.multiply(ct1, ct2, self.relin_key)