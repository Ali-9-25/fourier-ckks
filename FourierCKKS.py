import numpy as np
from ckks.ckks_parameters import CKKSParameters
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_encoder import CKKSEncoder
from fft.fft import *


class FourierCKKS:
    """
    A class to perform FFT-based homomorphic encryption using the CKKS scheme,
    supporting both 1D and 2D data via segmented FFT packing.
    """

    def __init__(self,
                 poly_degree=512,
                 ciph_modulus=(1 << 200),
                 big_modulus=(1 << 300),
                 scaling_factor=(1 << 30),
                 prime_size=30,
                 is_parallel=True):
        self.is_parallel = is_parallel
        # We begin by initializing the CKKS parameters and keys
        self.params = CKKSParameters(
            poly_degree=poly_degree,
            ciph_modulus=ciph_modulus,
            big_modulus=big_modulus,
            scaling_factor=scaling_factor,
            prime_size=prime_size
        )
        keygen = CKKSKeyGenerator(self.params, is_parallel=is_parallel)
        self.public_key = keygen.public_key
        self.secret_key = keygen.secret_key
        self.relin_key = keygen.relin_key

        # We then proceed to instantiate the Encoder, Encryptor, Decryptor, Evaluator
        self.encoder = CKKSEncoder(self.params)
        self.encryptor = CKKSEncryptor(self.params, self.public_key, self.secret_key)
        self.decryptor = CKKSDecryptor(self.params, self.secret_key)
        self.evaluator = CKKSEvaluator(self.params)

        # We define the maximum number of complex slots per ciphertext
        self.data_len = poly_degree // 2
        # For 2D support, we ensure that data_len is a perfect square (so that when we segment the FFT, we partition the problem
        # into squares)
        self.img_side = int(np.sqrt(self.data_len))
        if self.img_side * self.img_side != self.data_len:
            raise ValueError(f"data_len={self.data_len} is not a perfect square; cannot support 2D FFT packing.")

    def forward(self,
                message: np.ndarray,
                target_height: int = None,
                target_width: int = None) -> list:
        """
        Encode and encrypt a 1D or 2D message via segmented FFT packing.

        For 1D: provide target_length (>= message length).
        For 2D: provide target_height and target_width (>= each message dimension).

        Returns a list of ciphertext segments.
        """
        H, W = message.shape
        if target_width == 1:
            nv = int(np.ceil(target_height / self.data_len))
            ph = nv * self.data_len 
            nh = 1
            pw = 1
        else: 
            nv = int(np.ceil(target_height / self.img_side))
            ph = nv * self.img_side
            nh = int(np.ceil(target_width / self.img_side))
            pw = nh * self.img_side
        mat = np.zeros((ph, pw), dtype=complex)
        mat[:H, :W] = message
        if target_width == 1:
            freq = FFT1D(mat.flatten())
        else:
            freq2d = FFT2D(mat)
            freq = freq2d.flatten()
        ct_list = []
        for i in range(nv * nh):
            chunk = freq[i*self.data_len:(i+1)*self.data_len]
            pt = self.encoder.encode(chunk, self.params.scaling_factor)
            ct_list.append(self.encryptor.encrypt(pt, is_parallel=self.is_parallel))
        return ct_list
        

    def backward(self,
                 ct_list: list,
                 target_height: int = None,
                 target_width: int = None) -> np.ndarray:
        """
        Decrypt, decode, and inverse-FFT a list of ciphertexts,
        returning the real 1D array or 2D image truncated to target dimensions.
        """
        segs = len(ct_list)
        padded_len = segs * self.data_len

        freq = np.zeros(padded_len, dtype=complex)
        for i, ct in enumerate(ct_list):
            pt = self.decryptor.decrypt(ct, is_parallel=self.is_parallel)
            freq[i*self.data_len:(i+1)*self.data_len] = self.encoder.decode(pt)
        if target_width == 1:
            time = IFFT1D(freq)
            return time[:target_height]
        else: 
            nv = int(np.ceil(target_height / self.img_side))
            ph = nv * self.img_side
            nh = int(np.ceil(target_width / self.img_side))
            pw = nh * self.img_side
            freq = freq.reshape((ph, pw))
            time = IFFT2D(freq)
            return time[:target_height, :target_width]


    def cipher_add(self, ct_list1: list, ct_list2: list) -> list:
        """
        Homomorphically add two lists of ciphertexts segment-wise.
        """
        if len(ct_list1) != len(ct_list2):
            raise ValueError("Ciphertext lists must match segment count for addition.")
        return [self.evaluator.add(a, b) for a, b in zip(ct_list1, ct_list2)]

    def cipher_conv(self, ct_list1: list, ct_list2: list) -> list:
        """
        Homomorphically multiply two lists of ciphertexts segment-wise.
        """
        if len(ct_list1) != len(ct_list2):
            raise ValueError("Ciphertext lists must match segment count for convolution.")
        return [self.evaluator.multiply(a, b, self.relin_key, is_parallel=self.is_parallel)
                for a, b in zip(ct_list1, ct_list2)]


