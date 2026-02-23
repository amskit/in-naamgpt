# in-naamGPT 🇮🇳

> A ~5,400 parameter GPT trained **from scratch in pure Python** on 1,609 Sanskrit & Hindi names.
> No PyTorch. No TensorFlow. No NumPy. Just `math`, `random`, and hand-written backpropagation.
> Generates brand new **हिंदी** names that sound real but never existed — entirely in your browser.

**[✦ Live Demo →](https://in-naamgpt.vercel.app)** &nbsp;|&nbsp; **[Source Code →](https://github.com/RaikaSurendra/in-naamgpt)**

---

## Why This Exists

Every ML tutorial begins with `import torch`. But what actually happens inside `nn.Linear`? What does `loss.backward()` really compute? How does a Transformer learn to "understand" sequences?

This project strips away every abstraction. The entire training pipeline — forward pass, backpropagation, Adam optimizer, cross-entropy loss, RMSNorm, multi-head attention — is written from first principles in ~400 lines of Python. The result is a working character-level GPT that learns the phonotactic patterns of Hindi names and generates new ones.

The frontend goes further: it doesn't just generate names, it **visualizes every stage of the Transformer pipeline** interactively — embeddings, attention weights, loss gradients, and training dynamics — all running in the browser with no backend.

---

## At a Glance

| Component | Details |
|---|---|
| **Model** | 1-layer decoder-only Transformer, 4 attention heads, ~5,400 parameters |
| **Vocab** | 57 Unicode NFD characters (Devanagari consonants, vowels, matras, virama) |
| **Dataset** | 1,609 curated Sanskrit & Hindi names in Devanagari script |
| **Training** | 1,000 steps, Adam optimizer (β₁=0.85, β₂=0.99), cross-entropy loss |
| **Frontend** | React 19 + Vite 6 + Tailwind CSS 4, Bauhaus design system |
| **Inference** | Runs entirely in the browser via JavaScript — zero API calls, zero backend |
| **Dependencies** | Zero ML libraries. 4 frontend packages (react, react-dom, vite, tailwind) |

---

## How It Works

### 1. The Dataset

The model trains on **1,609 Sanskrit and Hindi names** in Devanagari script — names like आरव, प्रिया, कृष्ण, अग्निवेश, and यूथिका. The names were curated from multiple sources to cover a wide range of traditional and modern Indian naming patterns.

### 2. The Unicode Challenge: NFD Decomposition

Devanagari is a complex script. A single visible character like **की** is actually *two* Unicode code points:
- `क` — the consonant "Ka"
- `ी` — the dependent vowel sign "II" (a matra)

The model tokenizes all names into their **NFD (Normalized Form Decomposed)** characters. This splits every name into its atomic building blocks:

| Category | Examples | Count |
|---|---|---|
| **Consonants** | क, ख, ग, घ, च, छ, ज, ... | 34 |
| **Independent vowels** | अ, आ, इ, ई, उ, ऊ, ऋ, ए, ओ | 9 |
| **Dependent vowel signs (matras)** | ा, ि, ी, ु, ू, ृ, े, ै, ो, ौ | 12 |
| **Virama (halant)** | ् | 1 |
| **Nukta** | ़ | 1 |

This gives us a clean **57-token vocabulary** (55 characters + BOS token + padding). The model reads and writes in NFD internally, then normalizes back to **NFC** for human-readable display.

### 3. The Architecture

```
Input Token (character ID, 0–56)
    │
    ▼
Token Embedding + Position Embedding  →  16-dimensional vector
    │
    ▼
RMSNorm
    │
    ▼
Multi-Head Self-Attention (4 heads × 4 dims each)
    │   ┌─ Head 0: Q·K^T / √4 → softmax → weighted V
    │   ├─ Head 1: ...
    │   ├─ Head 2: ...
    │   └─ Head 3: ...
    │   → Concatenate → Output projection
    │
    ▼
+ Residual Connection (skip around attention)
    │
    ▼
RMSNorm → MLP (16 → 64 → ReLU → 64 → 16)
    │
    ▼
+ Residual Connection (skip around MLP)
    │
    ▼
Output Head: Linear → 57 logits → softmax → next character probability
```

**Key design choices:**
- **RMSNorm** instead of LayerNorm — simpler, no mean subtraction, just root-mean-square scaling
- **No bias terms** — reduces parameter count without hurting quality at this scale
- **4 attention heads** with 4-dimensional queries/keys/values each (4 × 4 = 16-dim embedding)
- **ReLU activation** in the MLP — simplest nonlinearity, works well for this size

### 4. Custom Autograd Engine

The training script includes a hand-built autograd engine. Every number in the model is wrapped in a `Value` object that tracks:
- Its current value (`.data`)
- Its gradient (`.grad`)
- What operations created it (computation graph)

When you call `.backward()` on the loss, gradients flow backward through the entire computation graph via the chain rule. This is the same algorithm PyTorch uses internally — just written out explicitly so every step is visible.

```python
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

Operations like `+`, `*`, matrix-vector products, softmax, and log all create new `Value` objects that remember their inputs and partial derivatives.

### 5. Training

The model trains for **1,000 steps** using Adam optimizer with linear learning rate decay:
- **Learning rate**: 0.003 → 0 (linear decay)
- **β₁**: 0.85, **β₂**: 0.99
- **Loss function**: Cross-entropy (negative log probability of the correct next character)

Each step picks a name from the dataset, tokenizes it into NFD characters, runs the full forward pass for every position, computes the loss, backpropagates gradients, and updates all ~5,400 parameters.

Training takes about **30 seconds** on a laptop CPU.

### 6. In-Browser Inference

After training, the model weights are exported as a JSON file (~118KB). The frontend loads this file and runs inference using a JavaScript engine that mirrors the Python forward pass exactly:
- Same matrix-vector multiplications
- Same RMSNorm computation
- Same multi-head attention with KV caching
- Same temperature-scaled sampling

This means **no server is needed**. The model runs entirely on your device. Nothing leaves your browser.

---

## Interactive Chapters

The frontend is an interactive walkthrough of how Transformers work, built with React and canvas:

### Chapter 01 — Name Cloud
Browse all 1,609 training names as floating Devanagari text. Hover to highlight individual names.

### Chapter 02 — Tokenization
Watch how Devanagari text decomposes into NFD characters. See each character's category (consonant, vowel, matra, virama) color-coded.

### Chapter 03 — Embedding Space
Select any token and position to explore the 16-dimensional embedding vectors. See how `wte[token] + wpe[position]` combines into the input representation, displayed as color-coded bars.

### Chapter 04 — Attention Pipeline
The most detailed visualization — see the full Q/K/V computation:
- Query and Key vectors for each head
- Attention scores (Q·K^T / √d)
- Softmax attention weights
- Value-weighted outputs
- Multi-head concatenation and output projection
- Final logits and next-token probabilities

### Chapter 05 — Loss & Gradients
Per-position predictions with cross-entropy loss. See which positions the model gets right and wrong, and visualize gradient magnitudes flowing backward.

### Chapter 06 — Training Replay
An animated loss curve you can replay step-by-step. Watch the loss decrease over 1,000 iterations. Hover to see the specific name, step, loss value, and learning rate at any point.

### Chapter 07 — Name Generator
The main event — generate new Hindi names with a temperature slider:
- **Low temperature (0.1–0.3)**: Conservative, common patterns
- **Balanced (0.3–0.7)**: Realistic sounding names
- **Creative (0.7–1.2)**: More variety and surprises
- **Wild (1.2+)**: Chaotic, unusual combinations

Click any generated name to see its step-by-step token selection with top-5 probabilities at each position.

---

## Project Structure

```
in-microgpt/
├── model/
│   ├── data/
│   │   └── in_name.txt                    # Dataset: 1,609 Hindi names (one per line)
│   ├── checkpoints/
│   │   └── in_model.pkl                   # Trained model checkpoint (gitignored)
│   ├── in_main.py                         # Training script + autograd engine + inference
│   └── scripts/
│       └── export_training_trace.py       # Re-trains while recording loss at each step
│
├── app/
│   ├── public/
│   │   ├── data/
│   │   │   ├── in_embedding_snapshot.json # Model weights for browser inference (~118KB)
│   │   │   ├── in_training_trace.json     # Step-by-step training data for visualization
│   │   │   └── in_name.txt               # Dataset copy for frontend name cloud
│   │   └── favicon.svg                   # Bauhaus geometric favicon
│   ├── src/
│   │   ├── components/
│   │   │   ├── FloatingShapes.jsx        # Background geometric decorations
│   │   │   ├── HeroSection.jsx           # Landing section with stats
│   │   │   ├── NameCloudSection.jsx      # Ch.01 — Floating name cloud
│   │   │   ├── TokenizationSection.jsx   # Ch.02 — NFD tokenization demo
│   │   │   ├── EmbeddingSection.jsx      # Ch.03 — Embedding vector explorer
│   │   │   ├── AttentionSection.jsx      # Ch.04 — Full attention pipeline
│   │   │   ├── LossGradientSection.jsx   # Ch.05 — Loss & gradient visualization
│   │   │   ├── TrainingSection.jsx       # Ch.06 — Training loss replay
│   │   │   ├── GeneratorSection.jsx      # Ch.07 — Name generator with temperature
│   │   │   ├── HowItWorksSection.jsx     # 4-step overview
│   │   │   ├── ArchitectureSection.jsx   # Model architecture diagram
│   │   │   └── FooterSection.jsx         # Credits and links
│   │   ├── gptInference.js              # Browser-side GPT forward pass
│   │   ├── App.jsx                      # Main app with lazy loading
│   │   ├── main.jsx                     # React entry point
│   │   └── index.css                    # Bauhaus design tokens & utilities
│   ├── index.html                       # Entry HTML with OG meta tags
│   ├── vercel.json                      # Security headers (CSP, X-Frame-Options, etc.)
│   ├── vite.config.js
│   └── package.json
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions: build verification
├── .coderabbit.yaml                     # CodeRabbit auto-review config
├── BLOG_POST.md                         # Ready-to-publish blog post
└── README.md
```

---

## Run Locally

### Prerequisites
- **Python 3.8+** (for training — no pip packages needed)
- **Node.js 18+** (for frontend)

### Train the model (optional — weights are already exported)

```bash
cd model
python3 in_main.py
```

This will:
1. Load the dataset from `data/in_name.txt`
2. Build the 57-token NFD vocabulary
3. Initialize a random Transformer (~5,400 parameters)
4. Train for 1,000 steps (~30 seconds)
5. Export weights to `app/public/data/in_embedding_snapshot.json`
6. Generate 20 sample names to verify the model works

### Export training trace (optional)

```bash
cd model
python3 scripts/export_training_trace.py
```

Re-trains from scratch while recording loss, learning rate, and parameter snapshots at each step. Exports to `app/public/data/in_training_trace.json` for the Training Replay visualization.

### Start the frontend

```bash
cd app
npm install
npx vite --port 5180
```

Open **http://localhost:5180** in your browser.

### Build for production

```bash
cd app
npx vite build
```

Output goes to `app/dist/` — a fully static site ready for any hosting provider.

---

## Design System: Bauhaus

The UI follows the [Bauhaus design philosophy](https://www.designprompts.dev/bauhaus) — form follows function, stripped of decoration:

| Token | Value | Usage |
|---|---|---|
| **Background** | `#F0F0F0` | Off-white canvas |
| **Foreground** | `#121212` | Stark black text |
| **Red** | `#D02020` | Primary accent, CTAs, loss indicators |
| **Blue** | `#1040C0` | Secondary accent, attention weights, links |
| **Yellow** | `#F0C020` | Tertiary accent, highlights, selections |
| **Borders** | 4px solid `#121212` | Thick, geometric, no border-radius |
| **Shadows** | `4px 4px 0px #121212` | Hard offset, no blur |
| **Font** | Outfit (400, 500, 700, 900) | Geometric sans-serif |

The aesthetic mirrors the project's philosophy: **strip away the unnecessary, reveal the structure**.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Training** | Pure Python (~400 LOC) | No abstractions — every gradient computed by hand |
| **Autograd** | Custom `Value` class | Tracks computation graph for backpropagation |
| **Optimizer** | Adam (hand-implemented) | Industry standard, good convergence |
| **Frontend** | React 19 + Vite 6 | Fast dev, fast builds, modern JSX |
| **Styling** | Tailwind CSS 4 | Utility-first, Bauhaus design tokens |
| **Visualizations** | Canvas API + CSS | No chart libraries — all custom drawn |
| **Inference** | Vanilla JavaScript | Mirrors Python forward pass exactly |
| **Deploy** | Vercel | Auto-deploy on push to `main` |
| **CI** | GitHub Actions | Build verification on every PR |
| **Code Review** | CodeRabbit | AI-powered PR reviews |

**Zero ML dependencies. Zero backend. Zero API keys.**

---

## Security

The app ships with hardened HTTP headers via `vercel.json`:
- **Content-Security-Policy** — restricts scripts, styles, fonts, and connections to trusted origins
- **X-Frame-Options: DENY** — prevents clickjacking
- **X-Content-Type-Options: nosniff** — prevents MIME sniffing
- **Referrer-Policy: strict-origin-when-cross-origin**
- **Permissions-Policy** — denies camera, microphone, geolocation

GitHub Actions CI runs with **least-privilege permissions** (`contents: read` only).

---

## Credits & Inspiration

This project stands on the shoulders of:
- **[Andrej Karpathy's microGPT](https://github.com/karpathy/microGPT)** — the original character-level GPT implementation
- **[ko-microgpt](https://github.com/woduq1414/ko-microgpt)** — Korean adaptation with interactive visualizations that inspired the chapter-based frontend
- **[Bauhaus design principles](https://www.designprompts.dev/bauhaus)** — geometric purity, primary colors, functional honesty

---

## License

MIT — use it, learn from it, build on it.
