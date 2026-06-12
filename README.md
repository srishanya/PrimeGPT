# PrimeGPT
## I made chatGPT inside a prime number.
A fully functional generative transformer packaged entirely inside a single, mathematically indivisible 3,000+ digit prime number.

Inspired by Andrej Karpathy's nanoGPT, Kuber Mehta's picoGPT, and Phil Carmody's legendary 2001 "illegal prime number" protest, this project compresses an entire deep learning framework from scratch into a single-file python script, then uses number theory to force its raw byte representation into a prime number.

If you take this prime number, convert its base-10 value back into raw bytes, and decompress it, you get a fully operational transformer with its own autograd engine, training loops, and inference execution block.

## 🛠️ What's Packed Inside the Prime?
This is not a mock script or an API wrapper. The code embedded inside this integer compiles a complete deep learning architecture from first principles using only numpy:

1. #### Custom Autograd Engine:
   A hand-rolled, tape-based backpropagation system tracking gradients across scalars and matrices     without torch.
2. #### Multi-Head Attention (MHA):
   Full causal masking, query/key/value ($Q, K, V$) projections, and scaled dot-product calculation.
3. #### Feed-Forward MLP:
   Multi-layer perceptron block running a fast element-wise $\text{GELU}$ activation approximation.
4. #### AdamW Optimizer:
   Explicit weight decay loops, first/second-moment bias corrections ($\beta_1, \beta_2$), and         numerical stability safeguards.
5. #### Training & Inference Loops:
   Character-level tokenization, cross-entropy loss computation, and temperature/top-$k$ scaled text generation.

## 📐 How It Works: Turning Code into Math
picoGPT.py (84 lines) ──> Minification ──> PrimeGPT.py (53 lines) ──> gzip + Padding Bytes ──> Big Integer ──> gmpy2.next_prime() ──> ⭐ THE PRIME ⭐
### CHECK OUT THE BLOG FOR MORE: 

## 🚀 Quick Start & Usage1. 
1.Installation
Clone the repository and install the minimal dependencies required for number processing:
#### Bashgit clone https://github.com/MatchaOnMuffins/femtoGPT.git
#### cd femtoGPT
#### pip install numpy gmpy2

2. Extracting the GPT from the PrimeTo unpack the code from the mathematical prime, copy the prime string from prime_number.txt and run the extraction routine:
#### Pythonimport gzip
#### THE_PRIME = 48565392019482039482039482039429443... 
*Convert the integer back to raw bytes*
#### byte_length = (THE_PRIME.bit_length() + 7) // 8
#### raw_bytes = THE_PRIME.to_bytes(byte_length, byteorder='big')
*Decompress the hidden valid gzip payload*
#### try:
####    gpt_source = gzip.decompress(raw_bytes).decode('utf-8')
####    with open("extracted_transformer.py", "w") as f:
####        f.write(gpt_source)
####    print("🔥 Success! extracted_transformer.py generated.")
#### except Exception as e:
####    print(f"Extraction failed: {e}")

3. Compiling Your Own Code into a Prime
If you modify femtoGPT.py and want to compute your own unique prime number, run the pipeline engine:
#### Bashpython make_prime.py --src femtoGPT.py --out my_prime.txt

## 🔬 Core Code Preview

The minified core operations are packed tightly to keep the resulting prime number within a clean digit threshold:

import numpy as np
class Tensor:
    def __init__(self, data, creators=None, op=None):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.creators, self.op = creators or [], op

    def backward(self, grad=None):
        if grad is None: grad = np.ones_like(self.data)
        self.grad += grad
        for c in self.creators:
            c.backward(self.op.apply_grad(self, c))

def attention(q, k, v, mask=None):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, k.swapaxes(-1, -2)) * scale
    if mask is not None: scores += (mask == 0) * -1e9
    probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.matmul(probs, v)
    
## 🤝 Contributing
Contributions, optimizations, and deeper mathematical simplifications are welcome! If you find a way to golf down the model architecture by even a few bytes (lowering the prime's digit count), feel free to open a pull request.
1. Fork the Project.
2. Create your Feature Branch (git checkout -b feature/GlowUp).
3. Commit your Changes (git commit -m 'Trimmed 4 bytes from MLP').
4. Push to the Branch (git push origin feature/GlowUp).
5. Open a Pull Request.

## 📜 Credits & Acknowledgments
Henry Zhang (femtoGPT): For the baseline architectural minification boundaries.
Andrej Karpathy (nanoGPT) & Kuber Mehta (picoGPT): For proving how clean a first-principles transformer can look.Phil Carmody (2001): For the original implementation of the "illegal prime" discovery algorithm.

## Legal Disclaimer: This prime number contains an operational machine learning model. 
## Distribute responsibly under standard mathematical and open-source complianc
