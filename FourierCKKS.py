import numpy as np
from ckks.ckks_parameters import CKKSParameters
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_encoder import CKKSEncoder
from fft.fft import FFT1D, IFFT1D, FFT2D, IFFT2D


class FourierCKKS:
    """
    A helper class to perform FFT-based homomorphic encryption using the CKKS scheme,
    supporting both 1D and 2D data via segmented FFT packing.
    """

    def __init__(self,
                 poly_degree=512,
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
        # For 2D support, ensure data_len is a perfect square
        self.img_side = int(np.sqrt(self.data_len))
        if self.img_side * self.img_side != self.data_len:
            raise ValueError(f"data_len={self.data_len} is not a perfect square; cannot support 2D FFT packing.")

    def forward(self,
                message: np.ndarray,
                target_length: int = None,
                target_height: int = None,
                target_width: int = None) -> list:
        """
        Encode and encrypt a 1D or 2D message via segmented FFT packing.

        For 1D: provide target_length (>= message length).
        For 2D: provide target_height and target_width (>= each message dimension).

        Returns a list of ciphertext segments.
        """
        # 1D case
        if message.ndim == 1:
            L = message.shape[0]
            T = target_length if target_length is not None else L
            if T < L:
                raise ValueError(f"target_length ({T}) must be >= message length ({L})")

            segs = int(np.ceil(T / self.data_len))
            padded_len = segs * self.data_len

            # pad
            vec = np.zeros(padded_len, dtype=complex)
            vec[:L] = message
            freq_full = FFT1D(vec)

            ct_list = []
            for i in range(segs):
                chunk = freq_full[i*self.data_len:(i+1)*self.data_len]
                pt = self.encoder.encode(chunk, self.params.scaling_factor)
                ct_list.append(self.encryptor.encrypt(pt))
            return ct_list

        # 2D case
        elif message.ndim == 2:
            H, W = message.shape
            Ht = target_height if target_height is not None else H
            Wt = target_width if target_width is not None else W
            if Ht < H or Wt < W:
                raise ValueError(f"target dimensions ({Ht}, {Wt}) must be >= message dimensions ({H}, {W})")

            # segmentation counts
            nv = int(np.ceil(Ht / self.img_side))
            nh = int(np.ceil(Wt / self.img_side))
            ph = nv * self.img_side
            pw = nh * self.img_side

            # pad
            mat = np.zeros((ph, pw), dtype=complex)
            mat[:H, :W] = message

            # 2D FFT
            freq2d = FFT2D(mat)
            freq_flat = freq2d.flatten()

            ct_list = []
            for i in range(nv * nh):
                chunk = freq_flat[i*self.data_len:(i+1)*self.data_len]
                pt = self.encoder.encode(chunk, self.params.scaling_factor)
                ct_list.append(self.encryptor.encrypt(pt))
            return ct_list

        else:
            raise ValueError("Input must be a 1D or 2D numpy array.")

    def backward(self,
                 ct_list: list,
                 target_length: int = None,
                 target_height: int = None,
                 target_width: int = None) -> np.ndarray:
        """
        Decrypt, decode, and inverse-FFT a list of ciphertexts,
        returning the real 1D array or 2D image truncated to target dimensions.
        """
        # 1D case
        if target_height is None and target_width is None:
            segs = len(ct_list)
            padded_len = segs * self.data_len

            freq = np.zeros(padded_len, dtype=complex)
            for i, ct in enumerate(ct_list):
                pt = self.decryptor.decrypt(ct)
                freq[i*self.data_len:(i+1)*self.data_len] = self.encoder.decode(pt)

            time = IFFT1D(freq)
            L = target_length if target_length is not None else padded_len
            return time[:L]

        # 2D case
        else:
            segs = len(ct_list)
            padded_len = segs * self.data_len

            freq = np.zeros(padded_len, dtype=complex)
            for i, ct in enumerate(ct_list):
                pt = self.decryptor.decrypt(ct)
                freq[i*self.data_len:(i+1)*self.data_len] = self.encoder.decode(pt)

            nv = int(np.ceil(target_height / self.img_side))
            nh = int(np.ceil(target_width / self.img_side))
            ph = nv * self.img_side
            pw = nh * self.img_side
            freq2d = freq.reshape((ph, pw))
            time2d = IFFT2D(freq2d)
            return time2d[:target_height, :target_width]

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
        return [self.evaluator.multiply(a, b, self.relin_key)
                for a, b in zip(ct_list1, ct_list2)]


