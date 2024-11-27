## CKKS and BFV:

### Data Representation:
**BFV scheme**
- Plaintext space: integers modulo a plaintext modulous q
- Coefficients:  
    - coefficients of the polynomials are whole numbers, and are the remainder relative to some other number k
    - they are in the ring $Z[X] / (X^n + 1)$
- Data type: exact integers

**CKKS scheme**
- Plaintext space: Real or complex numbers are represented as polynomials with integer coefficients
- Coefficients: Plaintexts are encoded into polynomials where coefficients are scaled by a factor $\Delta$ (the scaling factor) to manage precision
- Data type: approximate real numbers


### Encoding and Decoding:
**BFV scheme**
- Encoding: Maps integer values directly into polynomial coefficients modulo q
- Decoding: Direct extraction of integer coefficients from the decrypted polynomial

**CKKS scheme**
- Encoding: Involves scaling real numbers by a factor $\Delta$ and mapping them into polynomial coefficients. This process must carefully manage precision and rounding errors
- Decoding: Requires rescaling and rounding operations to retrieve approximate real values from the decrypted polynomial

### Arithmetic Operations:
**BFV scheme**
- Operations supported: Exact addition and multiplication modulo q
- Noise management: Noise increases with each operation; parameters must be chosen to keep noise below a threshold for correct decryption

**CKKS scheme**
- Operations supported: Approximate addition and multiplication of real numbers
- Noise and Error Management: Involves both noise from encryption and approximation errors from floating-point operations
- Rescaling Required: After multiplication, the scale of the ciphertext increases, requiring a rescaling operation to maintain manageable scales and precision

###  Key Switching and Relinearization:

**Both Schemes**
- Purpose: Reduce the size of ciphertexts after multiplication (which increases ciphertext size) back to the original size to keep computations efficient
- Implementation: Involves generating and using relinearization keys

**Differences:**
- BFV: Relinearization deals with integer coefficients
- CKKS: Must account for scaling factors and ensure that relinearization doesn't introduce significant approximation errors