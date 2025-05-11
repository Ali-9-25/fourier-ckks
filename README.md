# Fourier-CKKS &nbsp;🚀  
**GPU-accelerated homomorphic convolution for audio & images**

Fourier-CKKS is a research-grade prototype that lets an untrusted server **apply linear filters to encrypted signals** without ever seeing the raw data.  
It combines CKKS approximate homomorphic encryption with the convolution theorem: the client FFT-encodes the signal, the server does a *single* element-wise multiplication on ciphertexts, and the client finishes with an inverse FFT after decryption — yielding orders-of-magnitude speed-ups while preserving privacy. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

---

## ✨ Why this repo?

| Goal | How Fourier-CKKS meets it |
|------|---------------------------|
| **Privacy** | All audio/image samples are encrypted before leaving the client; the server only manipulates ciphertexts. |
| **Quantum-safety** | Security relies on Ring-LWE (CKKS) — believed hard even for quantum computers. :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3} |
| **Floating-point support** | CKKS natively packs complex/real values, perfect for DSP workloads. |
| **Efficiency** | Heavy FFT/NTT and polynomial arithmetic are parallelised on CUDA GPUs, giving 10–100× speed-ups over CPU implementations. :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5} |

---

## 🔍 Repository layout

```

FOURIER-CKKS/
├── ckks/                # Core CKKS implementation (keys, codec, evaluator…)
├── encode/              # Helper encoders/decoders
├── encrypt\_decrypt/     # High-level wrappers for encryption pipelines
├── fft/                 # Pure-CPU radix-2 FFT (+ tests)
├── fft\_cuda/            # Custom CUDA kernels for 1-D / 2-D FFT (+ tests)
├── ntt\_cuda/            # CUDA kernels for NTT & negacyclic poly-mult (+ tests)
├── images/              # Test images used by the notebooks
├── util/                # Misc. helpers (bit-rev tables, timing, RNS utils, …)
├── tests/               # PyTest suites for CKKS & arithmetic back-ends
│
├── FourierCKKS.py       # **High-level API** that glues CKKS ↔ FFT packing
├── FourierCKKS\_Parallel.py  # Experimental multi-GPU version
│
├── pipeline\_audio\_test.ipynb   # **Run me:** end-to-end audio demo
├── pipeline\_image\_test.ipynb   # **Run me:** end-to-end image demo
├── ckks\_test.ipynb            # Unit tests for CKKS correctness
│
├── examples/ (legacy)         # Older scratch notebooks / scripts
└── README.md

````

### Key module – `FourierCKKS.py`

* Instantiates CKKS parameters, keys, codec and evaluator  
* Packs **_n/2_ complex slots** per ciphertext; checks `poly_degree` is a square so 2-D data pack nicely  
* `forward()`  
  * Pads / FFTs the input (1-D or 2-D)  
  * Splits frequency vector into segments, encodes & encrypts each into CKKS slots  
* `cipher_add()` / `cipher_conv()`  
  * Slot-wise add / multiply ciphertext lists with automatic relinearisation  
* `backward()`  
  * Decrypts, decodes and inverse-FFTs, cropping back to original shape

The class therefore offers a simple three-liner for privacy-preserving convolution:

```python
fhe = FourierCKKS()
ct_signal  = fhe.forward(x)
ct_filtered = fhe.cipher_conv(ct_signal, ct_filter)
y = fhe.backward(ct_filtered, *x.shape)
````

Implementation details follow the design in §2 of the report .

---

## 🚀 Quick start

> **Prerequisites**
>
> * Python ≥ 3.9
> * CUDA-capable GPU (compute ≥ 7.0) with CUDA 11+ tool-kit installed
> * `pip install -r requirements.txt`   *(NumPy, Numba, PyCUDA, Cupy, Jupyter, …)*

```bash
git clone https://github.com/yourname/FOURIER-CKKS.git
cd FOURIER-CKKS

# launch the audio demo
jupyter notebook pipeline_audio_test.ipynb

# or the image demo
jupyter notebook pipeline_image_test.ipynb
```

Each notebook:

1. Loads/plots the plaintext signal or image
2. Encrypts & FFT-packs it (client step)
3. Sends ciphertext to “server” cell that does *one* homomorphic multiplication
4. Decrypts & inverse-FFTs, then visually / numerically compares to plain-domain convolution
5. Prints wall-clock timings (≈ 0.1 s for 16 k-sample audio, ≈ 0.25 s for 256² image on RTX 3060)&#x20;

---

## 🧪 Testing

Inside the fft_cuda and ntt_cuda folders, you will find many correctness and efficiency tests in the form of .cu files.

Tests cover:

* Correctness of CKKS encode → encrypt → decrypt → decode round-trip
* NTT-based polynomial multiplication vs naïve convolution

---

## 📖 Theory & project report

For an in-depth discussion of CKKS, the Fourier-CKKS trick, GPU kernels and benchmark results, see **`Fourier_CKKS.pdf`** in the repo (Sections 1–6) .

---

## 📜 License

MIT — see `LICENSE`.

If you use this code in academic work, please cite the accompanying report and this repository.

---

*Happy privacy-preserving signal processing!* ✌️

```
```
