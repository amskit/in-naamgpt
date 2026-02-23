const COLORS = ['#D02020', '#1040C0', '#F0C020']

const chapters = [
  {
    id: 'language-model',
    num: '01',
    title: 'What is a Language Model?',
    color: COLORS[0],
    subtitle: 'Predicting the next character — that\'s the whole game.',
    paragraphs: [
      'A language model is a system that learns to predict what comes next in a sequence. Give it "आर" and it might predict "व" to make "आरव". Give it "प्रि" and it predicts "या" to make "प्रिया". It doesn\'t "understand" Hindi — it learns statistical patterns from thousands of examples.',
      'Our model works at the character level. Instead of predicting whole words (like GPT-4), it predicts one Unicode character at a time. This is simpler but powerful enough to learn the phonotactic rules of Hindi names — which consonant-vowel combinations sound natural, how names typically start and end, and what lengths are common.',
      'The core loop is: start with a special [BOS] (beginning of sequence) token → predict the most likely next character → add it to the sequence → repeat until a [END] token is predicted or we hit the maximum length.',
    ],
    diagram: `[BOS] → predict → "आ" → predict → "र" → predict → "व" → predict → [END]
   ↑                ↑                ↑                ↑
   The model assigns a probability to every character
   at each step, then samples from that distribution.`,
    code: {
      file: 'app/src/gptInference.js',
      label: 'The generation loop in JavaScript',
      snippet: `for (let pos = 0; pos < block_size; pos++) {
  const logits = gptForward(tokenId, pos, keysCache, valuesCache, snapshot)
  const scaledLogits = logits.map((l) => l / temperature)
  const probs = softmax(scaledLogits)

  // Sample from distribution
  const r = Math.random()
  let cumulative = 0
  let nextToken = 0
  for (let i = 0; i < vocabSize; i++) {
    cumulative += probs[i]
    if (r < cumulative) { nextToken = i; break }
  }

  if (nextToken === bos) break  // [END] token
  chars.push(uchars[nextToken])
  tokenId = nextToken
}`,
    },
    takeaways: [
      'A language model predicts the next token in a sequence',
      'Our model predicts one Unicode character at a time',
      'Generation works by repeatedly sampling from predicted probabilities',
      'The same architecture scales from 5K params (ours) to 1.8T params (GPT-4)',
    ],
    further: [
      { title: 'Andrej Karpathy: Let\'s build GPT from scratch', url: 'https://www.youtube.com/watch?v=kCc8FmEb1nY' },
      { title: 'The Illustrated GPT-2', url: 'https://jalammar.github.io/illustrated-gpt2/' },
      { title: 'Language Modeling (Wikipedia)', url: 'https://en.wikipedia.org/wiki/Language_model' },
    ],
  },
  {
    id: 'tokenization',
    num: '02',
    title: 'Tokenization & Unicode NFD',
    color: COLORS[1],
    subtitle: 'Breaking Devanagari into its atomic building blocks.',
    paragraphs: [
      'Before the model can process text, we need to convert characters into numbers. This process is called tokenization. Most modern LLMs use subword tokenizers (BPE, SentencePiece) that split text into common chunks. Our model is simpler — it works at the individual Unicode character level.',
      'Devanagari is a complex script where a single visible "character" can be multiple Unicode code points. For example, "की" looks like one character but is actually two: क (Ka consonant) + ी (II vowel sign/matra). This is called a grapheme cluster.',
      'We use Unicode NFD (Normalized Form Decomposed) to split every name into its true atomic characters — base consonants, independent vowels, dependent vowel signs (matras), the virama (halant ्) that joins consonants, and the nukta (़). This gives us a clean 57-token vocabulary that the model can work with.',
      'After the model generates characters in NFD form, we normalize back to NFC (Normalized Form Composed) for human-readable display. This round-trip (NFC → NFD → model → NFD → NFC) is invisible to the user but crucial for the model.',
    ],
    diagram: `"कृष्ण" (Krishna)
    ↓ NFD decomposition
┌───┬───┬───┬───┬───┬───┐
│ क │ ृ │ ष │ ् │ ण │   │
└───┴───┴───┴───┴───┴───┘
  ↑   ↑   ↑   ↑   ↑
  Ka  Ri  Sha Virama Na
  (consonant) (matra) (consonant) (halant) (consonant)

Each box = one token in our vocabulary (ID 0–56)`,
    code: {
      file: 'model/in_main.py',
      label: 'Building the tokenizer from the dataset',
      snippet: `def build_tokenizer(docs):
    # Collect all unique NFD characters across all names
    chars = set()
    for doc in docs:
        for ch in doc:
            chars.add(ch)
    uchars = sorted(chars)
    stoi = {ch: i for i, ch in enumerate(uchars)}  # char → ID
    BOS = len(uchars)  # special start/end token
    vocab_size = len(uchars) + 1
    return {"uchars": uchars, "stoi": stoi, "BOS": BOS, "vocab_size": vocab_size}`,
    },
    takeaways: [
      'Tokenization converts text into numbers the model can process',
      'Devanagari characters can be multiple Unicode code points (grapheme clusters)',
      'NFD decomposition splits text into atomic code points (consonants, matras, virama)',
      'Our vocabulary has 57 tokens — 55 Devanagari characters + BOS + padding',
      'NFC normalization converts model output back to readable text',
    ],
    further: [
      { title: 'Unicode NFD vs NFC Explained', url: 'https://unicode.org/reports/tr15/' },
      { title: 'Devanagari Unicode Block', url: 'https://en.wikipedia.org/wiki/Devanagari_(Unicode_block)' },
      { title: 'Karpathy: Let\'s build the GPT tokenizer', url: 'https://www.youtube.com/watch?v=zduSFxRajkE' },
      { title: 'BPE Tokenization (Hugging Face)', url: 'https://huggingface.co/learn/nlp-course/chapter6/5' },
    ],
  },
  {
    id: 'embeddings',
    num: '03',
    title: 'Embeddings',
    color: COLORS[2],
    subtitle: 'Turning token IDs into rich numerical vectors the model can reason about.',
    paragraphs: [
      'A token ID like "14" means nothing to a neural network. We need to convert it into a vector — a list of numbers — that captures the character\'s properties. This is what an embedding does.',
      'Our model uses two embedding tables: Token Embeddings (wte) — a 57×16 matrix where each row is a 16-dimensional vector for a character. Position Embeddings (wpe) — a 32×16 matrix where each row encodes "what position in the name are we at?"',
      'The input to the model at each step is simply: x = wte[token_id] + wpe[position]. This 16-number vector now encodes both "which character" and "where in the sequence". The model learns these vectors during training — similar characters (like vowels) end up with similar vectors.',
      'Think of it like coordinates in a 16-dimensional space. The model learns to place similar characters nearby and different characters far apart. Position embeddings let the model know that the first character of a name behaves differently from the fifth.',
    ],
    diagram: `Token ID: 14 (क)     Position: 2

wte (57 × 16):              wpe (32 × 16):
┌────────────────┐          ┌────────────────┐
│ row 0  [....] │          │ row 0  [....] │
│ row 1  [....] │          │ row 1  [....] │
│ row 14 [0.3, -0.1, ...]│  │ row 2  [0.5, 0.2, ...]│  ← lookup
│ ...           │          │ ...           │
└────────────────┘          └────────────────┘
         ↓                           ↓
         └──────── + ────────────────┘
                   ↓
         x = [0.8, 0.1, ...] (16 numbers)
         This vector represents "क at position 2"`,
    code: {
      file: 'app/src/gptInference.js',
      label: 'Embedding lookup in the forward pass',
      snippet: `// Embedding: look up token vector + position vector, add them
let x = wte[tokenId].map((t, i) => t + wpe[posId][i])

// wte[tokenId] → 16-dim vector for this character
// wpe[posId]   → 16-dim vector for this position
// x            → 16-dim combined representation`,
    },
    takeaways: [
      'Embeddings convert discrete IDs into continuous vectors',
      'Token embeddings encode "which character" — learned during training',
      'Position embeddings encode "where in the sequence" — also learned',
      'The input is simply: token_embedding + position_embedding',
      'Similar characters end up with similar embedding vectors',
    ],
    further: [
      { title: 'Word2Vec Explained (Stanford)', url: 'https://jalammar.github.io/illustrated-word2vec/' },
      { title: 'The Illustrated Transformer — Embeddings', url: 'https://jalammar.github.io/illustrated-transformer/' },
      { title: 'Positional Encoding (Lil\'log)', url: 'https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#positional-encoding' },
    ],
  },
  {
    id: 'rmsnorm',
    num: '04',
    title: 'RMSNorm',
    color: COLORS[0],
    subtitle: 'Keeping numbers in a healthy range so training stays stable.',
    paragraphs: [
      'As data flows through the network, the numbers in our vectors can grow very large or very small. This makes training unstable — gradients explode or vanish, and the model fails to learn. Normalization fixes this.',
      'RMSNorm (Root Mean Square Normalization) is a simpler alternative to the more common LayerNorm. It scales a vector so that the root-mean-square of its values is 1. Unlike LayerNorm, it doesn\'t subtract the mean — it only rescales.',
      'The formula is: for a vector x of dimension d, compute RMS = √(mean(x²)), then output x / RMS. This keeps vectors at a consistent scale regardless of what transformations happened before.',
      'Our model applies RMSNorm twice per layer: once before attention, and once before the MLP. This is called "pre-norm" architecture (used by LLaMA, GPT-J) as opposed to "post-norm" (original Transformer). Pre-norm is more stable for training.',
    ],
    diagram: `Input vector x = [3.0, -1.0, 4.0, -2.0]

Step 1: Square each element
         x² = [9.0, 1.0, 16.0, 4.0]

Step 2: Mean of squares
         mean(x²) = 30.0 / 4 = 7.5

Step 3: Root mean square
         RMS = √(7.5 + ε) ≈ 2.739

Step 4: Divide
         output = x / RMS = [1.10, -0.37, 1.46, -0.73]

The vector now has a consistent scale (RMS ≈ 1.0).`,
    code: {
      file: 'app/src/gptInference.js',
      label: 'RMSNorm implementation',
      snippet: `export function rmsNorm(x, eps = 1e-5) {
  // Mean of squares
  const ms = x.reduce((a, v) => a + v * v, 0) / x.length
  // Scale factor = 1 / sqrt(mean_square + epsilon)
  const scale = 1 / Math.sqrt(ms + eps)
  // Multiply each element by scale
  return x.map((v) => v * scale)
}`,
    },
    takeaways: [
      'Normalization prevents numbers from growing too large or small',
      'RMSNorm = divide by root-mean-square, simpler than LayerNorm',
      'ε (epsilon, 1e-5) prevents division by zero',
      'Pre-norm (normalize before each sub-layer) is more stable than post-norm',
      'Used by modern models: LLaMA, Gemma, Mistral all use RMSNorm',
    ],
    further: [
      { title: 'RMSNorm Paper (Zhang & Sennrich, 2019)', url: 'https://arxiv.org/abs/1910.07467' },
      { title: 'LayerNorm vs RMSNorm', url: 'https://www.youtube.com/watch?v=pMEVfEwzOHI' },
      { title: 'Why Pre-Norm Works Better', url: 'https://arxiv.org/abs/2002.04745' },
    ],
  },
  {
    id: 'attention',
    num: '05',
    title: 'Self-Attention',
    color: COLORS[1],
    subtitle: 'The mechanism that lets each character "look at" every previous character.',
    paragraphs: [
      'Self-attention is the core innovation of the Transformer. It allows each position in the sequence to gather information from all previous positions. When predicting the next character after "प्रि", the model can attend to "प", "्", "र", and "ि" simultaneously — deciding how much to "pay attention" to each.',
      'The mechanism works through three projections: Query (Q) — "what am I looking for?", Key (K) — "what do I contain?", and Value (V) — "what information do I provide?". The attention score between two positions is Q·K (dot product) — if a query matches a key, that position gets high attention.',
      'Multi-head attention splits the embedding into multiple "heads" that attend independently. Our model has 4 heads, each operating on 4 dimensions (4×4 = 16-dim embedding). Each head can learn different patterns — one might track consonant-vowel pairs, another might track name length patterns.',
      'Scores are scaled by √d (where d = head dimension = 4) to prevent large dot products from pushing softmax into extreme values. The model is autoregressive — each position can only attend to positions before it (causal masking), preventing it from "cheating" by looking at future characters.',
    ],
    diagram: `Position 3 attends to positions 0, 1, 2, 3:

Query (pos 3) · Key (pos 0) = 0.2  →  softmax  → 0.05  ← low attention
Query (pos 3) · Key (pos 1) = 1.8  →  softmax  → 0.40  ← high attention!
Query (pos 3) · Key (pos 2) = 0.5  →  softmax  → 0.15  ← some attention
Query (pos 3) · Key (pos 3) = 1.2  →  softmax  → 0.40  ← high attention!
                                                   ────
                                                   1.00  (probabilities sum to 1)

Output = 0.05 × V₀ + 0.40 × V₁ + 0.15 × V₂ + 0.40 × V₃
       = weighted mix of information from all positions`,
    code: {
      file: 'app/src/gptInference.js',
      label: 'Multi-head attention forward pass',
      snippet: `// Project input into Q, K, V
const q = matVec(attn_wq, x)   // Query: "what am I looking for?"
const k = matVec(attn_wk, x)   // Key: "what do I contain?"
const v = matVec(attn_wv, x)   // Value: "what info do I provide?"

keysCache.push(k)       // Store K for future positions
valuesCache.push(v)     // Store V for future positions

// For each attention head...
for (let h = 0; h < n_head; h++) {
  const qH = q.slice(hs, hs + head_dim)  // This head's query

  // Score against all previous keys
  for (let t = 0; t < keysCache.length; t++) {
    const kH = keysCache[t].slice(hs, hs + head_dim)
    score = dotProduct(qH, kH) / Math.sqrt(head_dim)  // Scaled dot product
  }
  const attnWeights = softmax(attnLogits)  // Normalize to probabilities

  // Weighted sum of values
  for (let t = 0; t < valuesCache.length; t++) {
    val += attnWeights[t] * valuesCache[t][hs + j]
  }
}
// Concatenate all heads → output projection
x = matVec(attn_wo, xAttn)`,
    },
    takeaways: [
      'Attention lets each position gather info from all previous positions',
      'Q·K dot product measures how "relevant" two positions are to each other',
      'Scaling by √d prevents softmax from saturating',
      'Multi-head = multiple independent attention patterns in parallel',
      'KV-cache stores past keys/values so we don\'t recompute them during generation',
      'Causal masking ensures the model can\'t peek at future tokens',
    ],
    further: [
      { title: 'Attention Is All You Need (original paper)', url: 'https://arxiv.org/abs/1706.03762' },
      { title: 'The Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/' },
      { title: '3Blue1Brown: Attention in Transformers', url: 'https://www.youtube.com/watch?v=eMlx5fFNoYc' },
      { title: 'The Annotated Transformer (Harvard NLP)', url: 'https://nlp.seas.harvard.edu/2018/04/03/attention.html' },
    ],
  },
  {
    id: 'residual',
    num: '06',
    title: 'Residual Connections',
    color: COLORS[2],
    subtitle: 'Skip connections that let gradients flow and preserve information.',
    paragraphs: [
      'A residual connection (skip connection) is deceptively simple: instead of replacing the input with the layer\'s output, you add the layer\'s output to the input. In code: output = layer(x) + x. That "+ x" is the entire idea.',
      'Why does this matter? Without residual connections, each layer must learn the complete transformation from input to output. With them, each layer only needs to learn the "correction" or "residual" — what to add to improve the representation. This is much easier to learn.',
      'The second benefit is gradient flow. During backpropagation, gradients must travel backward through every layer. Without skip connections, gradients multiply through many matrix operations and can vanish to zero. The skip connection provides a "gradient highway" — gradients flow directly through the addition operation with a gradient of 1.',
      'Our model has two residual connections per layer: one around the attention block and one around the MLP. This pattern (attention + skip → MLP + skip) is used in every Transformer ever built, from the original 2017 paper to GPT-4.',
    ],
    diagram: `WITHOUT residual:              WITH residual:

x ──→ [Attention] ──→ y       x ──→ [Attention] ──→ (+) ──→ y
                                │                      ↑
                                └──────────────────────┘
                                    (skip connection)

y = Attention(x)               y = Attention(x) + x

The layer only needs to         The "+" provides a direct
learn the FULL transform.       path for gradients to flow.
                                Layer learns the DELTA only.`,
    code: {
      file: 'app/src/gptInference.js',
      label: 'Both residual connections in the forward pass',
      snippet: `// === Residual connection #1: around Attention ===
const xResidual1 = x              // Save input
x = rmsNorm(x)                    // Normalize before attention
// ... attention computation ...
x = matVec(attn_wo, xAttn)        // Attention output
x = x.map((a, i) => a + xResidual1[i])  // ADD input back! ← residual

// === Residual connection #2: around MLP ===
const xResidual2 = x              // Save input
x = rmsNorm(x)                    // Normalize before MLP
x = relu(matVec(mlp_fc1, x))      // MLP hidden layer
x = matVec(mlp_fc2, x)            // MLP output
x = x.map((a, i) => a + xResidual2[i])  // ADD input back! ← residual`,
    },
    takeaways: [
      'Residual = output + input — just add the input back',
      'Layers learn "corrections" (deltas) instead of full transformations',
      'Provides a gradient highway — prevents vanishing gradients',
      'Invented by ResNet (2015) — enabled 152-layer deep networks',
      'Every Transformer uses this pattern: [Norm → Sub-layer → + Residual]',
    ],
    further: [
      { title: 'Deep Residual Learning (ResNet paper)', url: 'https://arxiv.org/abs/1512.03385' },
      { title: 'Why ResNets Work (3Blue1Brown)', url: 'https://www.youtube.com/watch?v=sLJKhkvhbBo' },
      { title: 'Residual Connections in Transformers', url: 'https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention' },
    ],
  },
  {
    id: 'mlp',
    num: '07',
    title: 'Feed-Forward Network (MLP)',
    color: COLORS[0],
    subtitle: 'The "thinking" layer — where the model processes information gathered by attention.',
    paragraphs: [
      'After attention gathers information from across the sequence, the MLP (Multi-Layer Perceptron) processes that information. If attention is "what to look at", the MLP is "what to do with what you saw".',
      'Our MLP has two linear layers with a ReLU activation in between. The first layer expands the 16-dim vector to 64 dimensions (4× expansion), applies ReLU to introduce nonlinearity, then the second layer projects back to 16 dimensions. This expand-then-compress pattern is standard in Transformers.',
      'ReLU (Rectified Linear Unit) is the simplest nonlinearity: max(0, x). Negative values become zero, positive values pass through unchanged. Without nonlinearity, stacking multiple linear layers would just be one big linear layer — you need nonlinearity for the network to learn complex patterns.',
      'The 4× expansion ratio means the MLP has a 64-dimensional "hidden layer" where it can represent more complex features before compressing back to 16 dimensions. Modern LLMs use the same ratio (GPT-3 uses 4× expansion) or wider (LLaMA uses 8/3×).',
    ],
    diagram: `Input x (16-dim)
     │
     ▼
┌──────────┐
│ Linear 1 │  16 → 64  (expand 4×)
│ W₁·x     │  768 weights + 0 bias
└──────────┘
     │
     ▼
┌──────────┐
│   ReLU   │  max(0, x) — kill negatives
└──────────┘
     │
     ▼
┌──────────┐
│ Linear 2 │  64 → 16  (compress back)
│ W₂·h     │  1024 weights + 0 bias
└──────────┘
     │
     ▼
Output (16-dim)  →  + residual`,
    code: {
      file: 'app/src/gptInference.js',
      label: 'MLP forward pass',
      snippet: `// MLP: expand → nonlinearity → compress
x = rmsNorm(x)                    // Normalize first
x = relu(matVec(mlp_fc1, x))      // 16 → 64, then ReLU
x = matVec(mlp_fc2, x)            // 64 → 16

// relu: simply max(0, value) for each element
export function relu(x) {
  return x.map((v) => Math.max(0, v))
}`,
    },
    takeaways: [
      'MLP processes the information that attention gathered',
      'Two linear layers with nonlinearity (ReLU) in between',
      '4× expansion: 16 → 64 → 16 gives more processing capacity',
      'Without ReLU, the network would just be one linear transformation',
      'The MLP contains most of the model\'s parameters',
    ],
    further: [
      { title: 'ReLU and Friends (Activation Functions)', url: 'https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity' },
      { title: 'Why Do MLPs in Transformers Work?', url: 'https://arxiv.org/abs/2202.06709' },
      { title: 'Universal Approximation Theorem', url: 'https://en.wikipedia.org/wiki/Universal_approximation_theorem' },
    ],
  },
  {
    id: 'softmax-loss',
    num: '08',
    title: 'Softmax & Cross-Entropy Loss',
    color: COLORS[1],
    subtitle: 'Converting raw scores into probabilities, then measuring how wrong we are.',
    paragraphs: [
      'The model\'s output head produces 57 raw numbers called logits — one for each character in our vocabulary. These can be any value (positive, negative, large, small). Softmax converts them into a probability distribution: all positive, summing to 1.0.',
      'Softmax works by exponentiating each logit, then dividing by the sum: P(i) = exp(logit_i) / Σ exp(logit_j). This amplifies differences — the largest logit gets the highest probability, and much smaller logits get near-zero probability. The "temperature" parameter divides logits before softmax to control this sharpness.',
      'Cross-entropy loss measures how wrong the model\'s prediction is. If the correct next character is "र" (token 14), the loss is: -log(P(14)). If the model assigns 90% probability to "र", loss = -log(0.9) = 0.105 (low, good!). If it assigns 1% probability, loss = -log(0.01) = 4.605 (high, bad!).',
      'The genius of cross-entropy: it\'s -log(probability of the correct answer). Perfect prediction (probability 1.0) gives loss 0. Terrible prediction (probability → 0) gives loss → ∞. The gradients naturally push the model to increase the probability of correct characters.',
    ],
    diagram: `Logits (raw output):     [2.1, 0.3, 5.8, -1.0, ...]  ← 57 numbers
                                              ↓
Softmax:                 [0.02, 0.004, 0.95, 0.001, ...] ← probabilities (sum=1.0)
                                   ↑
                         Correct token is index 2

Loss = -log(0.95) = 0.051  ← Low loss! Model is confident and correct.

If model predicted P(correct) = 0.01:
Loss = -log(0.01) = 4.605  ← High loss! Model is wrong.`,
    code: {
      file: 'app/src/gptInference.js',
      label: 'Softmax implementation with numerical stability',
      snippet: `export function softmax(values) {
  if (!values.length) return []
  const max = Math.max(...values)          // Subtract max for stability
  const exps = values.map((v) => Math.exp(v - max))  // Exponentiate
  const sum = exps.reduce((a, b) => a + b, 0)        // Sum
  if (!Number.isFinite(sum) || sum <= 0)
    return values.map(() => 1 / values.length)  // Fallback: uniform
  return exps.map((v) => v / sum)                     // Normalize
}

// Loss = -log(probability of correct token)
// In training: losses.push(-probs[target_id].log())`,
    },
    takeaways: [
      'Softmax converts raw logits into probabilities (positive, sum to 1)',
      'Subtracting the max before exp() prevents numerical overflow',
      'Cross-entropy loss = -log(probability of correct answer)',
      'Perfect prediction → loss 0, wrong prediction → loss → ∞',
      'Gradients naturally push the model to increase correct probabilities',
    ],
    further: [
      { title: 'Softmax Function (Deep Learning Book)', url: 'https://www.deeplearningbook.org/contents/numerical.html' },
      { title: 'Cross-Entropy Loss Explained', url: 'https://machinelearningmastery.com/cross-entropy-for-machine-learning/' },
      { title: 'Visual Guide to Cross-Entropy', url: 'https://colah.github.io/posts/2015-09-Visual-Information/' },
    ],
  },
  {
    id: 'backprop',
    num: '09',
    title: 'Backpropagation & Autograd',
    color: COLORS[2],
    subtitle: 'The chain rule — how the model learns which parameters to adjust.',
    paragraphs: [
      'Training a neural network means finding the right values for its parameters (weights). Backpropagation tells us which direction to adjust each parameter to reduce the loss. It works by computing the gradient (partial derivative) of the loss with respect to every parameter.',
      'The chain rule from calculus is the key: if z = f(g(x)), then dz/dx = dz/dg · dg/dx. In a neural network, the loss is the result of many nested functions. Backprop traverses this chain backward — from the loss, through each layer, back to the input — multiplying local gradients along the way.',
      'Our custom autograd engine wraps every number in a Value object that remembers what operation created it and what its local gradients are. When you call loss.backward(), it performs a topological sort of the computation graph and propagates gradients in reverse order.',
      'This is exactly what PyTorch does behind the scenes with torch.autograd. By implementing it ourselves, we can see every gradient computation explicitly. Each Value stores .data (the number) and .grad (how much the loss changes when this number changes).',
    ],
    diagram: `Forward pass (compute loss):

  wte[14] ──→ (+wpe) ──→ RMSNorm ──→ Attention ──→ MLP ──→ Logits ──→ Loss
                                                                       = 2.3

Backward pass (compute gradients):

  ∂L/∂wte ←── ∂L/∂x ←── ∂L/∂norm ←── ∂L/∂attn ←── ∂L/∂mlp ←── ∂L/∂logits ←── 1.0
   = 0.03       = 0.1      = 0.15       = 0.08        = 0.2        = 0.5

Each gradient says: "if I increase this value by a tiny amount,
how much does the loss change?"

wte[14].grad = 0.03  →  decrease wte[14] by 0.03 × learning_rate`,
    code: {
      file: 'model/in_main.py',
      label: 'Custom autograd Value class with backward pass',
      snippet: `class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0               # Gradient accumulates here
        self._children = children    # What created this value
        self._local_grads = local_grads  # Local partial derivatives

    def backward(self):
        # Topological sort: process nodes in reverse order
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)

        self.grad = 1  # dL/dL = 1
        for node in reversed(topo):
            for child, lg in zip(node._children, node._local_grads):
                child.grad += node.grad * lg  # Chain rule!`,
    },
    takeaways: [
      'Backpropagation = chain rule applied recursively through the network',
      'Each parameter gets a gradient: how much the loss changes if we tweak it',
      'Autograd tracks the computation graph automatically',
      'Topological sort ensures we process nodes in the correct order',
      'loss.backward() computes all gradients in one pass',
    ],
    further: [
      { title: 'Karpathy: micrograd — Autograd from Scratch', url: 'https://www.youtube.com/watch?v=VMj-3S1tku0' },
      { title: 'Calculus on Computational Graphs (Colah)', url: 'https://colah.github.io/posts/2015-08-Backprop/' },
      { title: '3Blue1Brown: Backpropagation', url: 'https://www.youtube.com/watch?v=Ilg3gGewQ5U' },
      { title: 'CS231n: Backpropagation', url: 'https://cs231n.github.io/optimization-2/' },
    ],
  },
  {
    id: 'adam',
    num: '10',
    title: 'Adam Optimizer',
    color: COLORS[0],
    subtitle: 'The smart way to update parameters — with momentum and adaptive learning rates.',
    paragraphs: [
      'Once backpropagation gives us gradients, we need to update the parameters. The simplest approach is SGD: param -= learning_rate × gradient. But SGD has problems — it\'s slow on flat surfaces, oscillates on steep ones, and uses the same learning rate for all parameters.',
      'Adam (Adaptive Moment Estimation) fixes these issues by tracking two things for each parameter: the first moment (m) — a running average of gradients (momentum), and the second moment (v) — a running average of squared gradients (adaptive rate).',
      'Momentum (m) smooths out noisy gradients — if gradients consistently point in one direction, momentum builds up and accelerates. Adaptive rate (v) gives each parameter its own effective learning rate — parameters with large gradients get smaller steps, parameters with small gradients get larger steps.',
      'We also use linear learning rate decay: lr starts at 0.003 and decreases to 0 over 1,000 steps. This lets the model take big steps early (explore) and small steps later (fine-tune). Bias correction (dividing by 1-β^t) compensates for the zero initialization of m and v in early steps.',
    ],
    diagram: `For each parameter p:

Step 1: Update momentum (smoothed gradient)
         m = 0.85 × m + 0.15 × gradient

Step 2: Update velocity (smoothed squared gradient)
         v = 0.99 × v + 0.01 × gradient²

Step 3: Bias correction (important for early steps)
         m̂ = m / (1 - 0.85^step)
         v̂ = v / (1 - 0.99^step)

Step 4: Update parameter
         p -= lr × m̂ / (√v̂ + ε)
                   ↑          ↑
         direction from    adaptive scale:
         momentum          big gradients → small step
                           small gradients → big step`,
    code: {
      file: 'model/in_main.py',
      label: 'Adam optimizer update with linear LR decay',
      snippet: `# Linear learning rate decay
lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)

# Adam update for each parameter
for i, p in enumerate(params):
    # Momentum: exponential moving average of gradient
    m_buf[i] = BETA1 * m_buf[i] + (1 - BETA1) * p.grad

    # Velocity: exponential moving average of gradient²
    v_buf[i] = BETA2 * v_buf[i] + (1 - BETA2) * p.grad ** 2

    # Bias correction
    m_hat = m_buf[i] / (1 - BETA1 ** (step + 1))
    v_hat = v_buf[i] / (1 - BETA2 ** (step + 1))

    # Update: momentum direction, adaptive scale
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
    p.grad = 0  # Reset gradient for next step`,
    },
    takeaways: [
      'Adam = momentum + adaptive learning rates per parameter',
      'β₁=0.85 controls momentum smoothing, β₂=0.99 controls rate adaptation',
      'Bias correction compensates for zero initialization in early steps',
      'Linear LR decay: big steps early (explore), small steps later (fine-tune)',
      'Adam is the default optimizer for most modern deep learning',
    ],
    further: [
      { title: 'Adam Paper (Kingma & Ba, 2014)', url: 'https://arxiv.org/abs/1412.6980' },
      { title: 'Why Adam Works (Ruder\'s Optimizer Overview)', url: 'https://ruder.io/optimizing-gradient-descent/' },
      { title: 'SGD vs Adam vs AdamW', url: 'https://www.youtube.com/watch?v=JXQT_vxqwIs' },
    ],
  },
  {
    id: 'temperature',
    num: '11',
    title: 'Temperature & Sampling',
    color: COLORS[1],
    subtitle: 'Controlling creativity — from conservative to wild.',
    paragraphs: [
      'After the model produces probabilities for the next character, we need to pick one. The simplest approach is "greedy" — always pick the highest probability. But this produces the same output every time. Instead, we sample randomly from the distribution, weighted by probabilities.',
      'Temperature controls the "sharpness" of the distribution. Before softmax, we divide logits by the temperature value. Temperature = 1.0 keeps the original distribution. Temperature < 1.0 makes it sharper — the model becomes more confident, picks common patterns. Temperature > 1.0 flattens it — the model becomes more random, produces unusual combinations.',
      'At temperature → 0, the distribution collapses to a spike on the most likely character (greedy decoding). At temperature → ∞, all characters become equally likely (random). The sweet spot for our model is around 0.3–0.7 for realistic-sounding names.',
      'This is the same parameter you see in ChatGPT\'s settings. Low temperature = predictable/factual, high temperature = creative/surprising. The math is identical whether you have 5,400 parameters or 1.8 trillion.',
    ],
    diagram: `Original logits: [2.0, 5.0, 1.0, 0.5]

Temperature = 0.3 (conservative):
  logits / 0.3 = [6.7, 16.7, 3.3, 1.7]
  softmax = [0.00, 1.00, 0.00, 0.00]  ← almost always picks token 1

Temperature = 1.0 (balanced):
  logits / 1.0 = [2.0, 5.0, 1.0, 0.5]
  softmax = [0.05, 0.84, 0.02, 0.01]  ← usually picks token 1, sometimes others

Temperature = 2.0 (creative):
  logits / 2.0 = [1.0, 2.5, 0.5, 0.25]
  softmax = [0.13, 0.58, 0.08, 0.06]  ← more variety, more surprises`,
    code: {
      file: 'app/src/gptInference.js',
      label: 'Temperature scaling and sampling',
      snippet: `// Scale logits by temperature before softmax
const scaledLogits = logits.map((l) => l / temperature)
const probs = softmax(scaledLogits)

// Sample from the probability distribution
const r = Math.random()        // Random number 0–1
let cumulative = 0
let nextToken = 0
for (let i = 0; i < vocabSize; i++) {
  cumulative += probs[i]       // Walk through CDF
  if (r < cumulative) {
    nextToken = i              // Pick this token
    break
  }
}
// Higher probability tokens have larger "slices" of the 0–1 range
// so they're more likely to be picked`,
    },
    takeaways: [
      'Temperature divides logits before softmax to control randomness',
      'Low temperature (0.1–0.3): conservative, repetitive, safe patterns',
      'High temperature (1.0+): creative, surprising, sometimes nonsensical',
      'Sampling from probabilities (not greedy) enables diverse outputs',
      'Same concept used in ChatGPT, Claude, Gemini — just scaled up',
    ],
    further: [
      { title: 'Temperature in Language Models', url: 'https://lukesalamone.github.io/posts/what-is-temperature/' },
      { title: 'Top-k and Top-p Sampling', url: 'https://huggingface.co/blog/how-to-generate' },
      { title: 'Sampling Strategies Compared', url: 'https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277' },
    ],
  },
]

export default chapters
