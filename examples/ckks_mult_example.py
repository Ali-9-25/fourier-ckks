"""Example of CKKS multiplication."""

from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters


def main():

    poly_degree = 4
    ciph_modulus = 1 << 600
    q0 = 2**24
    big_modulus = 1 << 1200
    scaling_factor = 2**7-1
    ciph_modulus = scaling_factor**2 * q0  # ql
    params = CKKSParameters(poly_degree=poly_degree,
                            ciph_modulus=ciph_modulus,
                            big_modulus=big_modulus,
                            scaling_factor=scaling_factor)
    key_generator = CKKSKeyGenerator(params)
    public_key = key_generator.public_key
    secret_key = key_generator.secret_key
    relin_key = key_generator.relin_key
    encoder = CKKSEncoder(params)
    encryptor = CKKSEncryptor(params, public_key, secret_key)
    decryptor = CKKSDecryptor(params, secret_key)
    evaluator = CKKSEvaluator(params)

    message1 = [4.0 + 0j, 3 + 0j]
    message2 = [4.0 + 0j, 3 + 0j]
    plain_product = [m1 * m2 for m1, m2 in zip(message1, message2)]
    print("Plaintext multiplication result:", plain_product)
    plain1 = encoder.encode(message1, scaling_factor)
    plain2 = encoder.encode(message2, scaling_factor)
    ciph1 = encryptor.encrypt(plain1)
    ciph2 = encryptor.encrypt(plain2)
    ciph_prod = evaluator.multiply(ciph1, ciph2, relin_key)
    print("Polynomial multip result:", ciph_prod)
    decrypted_prod = decryptor.decrypt(ciph_prod)
    decoded_prod = encoder.decode(decrypted_prod)

    print("Ciphertext multiplication result:", decoded_prod)


if __name__ == '__main__':
    main()
