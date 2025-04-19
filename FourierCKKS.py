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
    A helper class to perform FFT-based homomorphic encryption using the CKKS scheme,
    with support for messages larger than the native slot count via segmentation.
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

        # Encoder, Encryptor, Decryptor, Evaluator
        self.encoder = CKKSEncoder(self.params)
        self.encryptor = CKKSEncryptor(self.params, self.public_key, self.secret_key)
        self.decryptor = CKKSDecryptor(self.params, self.secret_key)
        self.evaluator = CKKSEvaluator(self.params)

        # Maximum number of complex slots per ciphertext
        self.data_len = poly_degree // 2

    def forward(self, message: np.ndarray, target_length: int = None) -> list:
        """
        Encode and encrypt a (possibly large) complex-valued message via segmented FFT:
        1) Determine target_length (for padded FFT) as either provided or len(message).
        2) Pad message up to the nearest multiple of data_len.
        3) Compute FFT of the full padded vector.
        4) Split the frequency-domain vector into chunks of length data_len,
           encoding and encrypting each chunk separately.

        :param message: 1D complex numpy array of length L
        :param target_length: desired output length (for convolution/addition), must be >= L
        :return: list of ciphertexts, one per segment
        """
        L = message.shape[0]
        # Determine full FFT length
        full_len = target_length if target_length is not None else L
        if full_len < L:
            raise ValueError(f"target_length ({full_len}) must be >= message length ({L})")

        # Number of segments
        segs = int(np.ceil(full_len / self.data_len))
        padded_len = segs * self.data_len

        # Embed and pad
        vec = np.zeros(padded_len, dtype=complex)
        vec[:L] = message

        # Full FFT
        freq_full = FFT1D(vec)

        # Encode+encrypt per segment
        ct_list = []
        for i in range(segs):
            start = i * self.data_len
            end = start + self.data_len
            freq_chunk = freq_full[start:end]
            pt = self.encoder.encode(freq_chunk, self.params.scaling_factor)
            ct = self.encryptor.encrypt(pt)
            ct_list.append(ct)
        return ct_list

    def backward(self, ct_list: list, target_length: int = None) -> np.ndarray:
        """
        Decrypt and decode a list of ciphertexts, reassemble frequency-domain vector,
        perform inverse FFT, and return the first target_length time-domain samples.

        :param ct_list: list of ciphertexts (from forward or homomorphic ops)
        :param target_length: number of time-domain samples to return
        :return: 1D complex numpy array of length target_length or full padded length
        """
        segs = len(ct_list)
        padded_len = segs * self.data_len

        # Decrypt+decode per segment
        freq_full = np.zeros(padded_len, dtype=complex)
        for i, ct in enumerate(ct_list):
            pt = self.decryptor.decrypt(ct)
            freq_chunk = self.encoder.decode(pt)
            freq_full[i*self.data_len:(i+1)*self.data_len] = freq_chunk

        # Inverse FFT
        time_full = IFFT1D(freq_full)

        # Truncate
        if target_length is not None:
            if target_length > padded_len:
                raise ValueError(f"target_length ({target_length}) > padded length ({padded_len})")
            return time_full[:target_length]
        return time_full

    def cipher_add(self, ct_list1: list, ct_list2: list) -> list:
        """
        Homomorphically add two lists of ciphertexts segment-wise.
        """
        if len(ct_list1) != len(ct_list2):
            raise ValueError("Ciphertext lists must have the same number of segments for addition.")
        return [self.evaluator.add(a, b) for a, b in zip(ct_list1, ct_list2)]

    def cipher_conv(self, ct_list1: list, ct_list2: list) -> list:
        """
        Homomorphically multiply (frequency-domain) two lists of ciphertexts segment-wise.
        """
        if len(ct_list1) != len(ct_list2):
            raise ValueError("Ciphertext lists must have the same number of segments for convolution.")
        return [self.evaluator.multiply(a, b, self.relin_key)
                for a, b in zip(ct_list1, ct_list2)]
