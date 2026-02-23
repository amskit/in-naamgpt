# in-naamGPT 🇮🇳

A tiny GPT trained from scratch on **1,609 Sanskrit & Hindi names** in pure Python — no PyTorch, no TensorFlow, just math. Generates brand new हिंदी names that sound real but never existed.

**[Live Demo →](https://in-naamgpt.vercel.app)**

## What's Inside

| Component | Details |
|---|---|
| **Model** | 1-layer Transformer, 4 attention heads, ~5,400 parameters |
| **Vocab** | 57 Unicode NFD characters (Devanagari consonants, vowels, matras, virama) |
| **Dataset** | 1,609 curated Sanskrit & Hindi names |
| **Training** | 1,000 steps, Adam optimizer, cross-entropy loss |
| **Frontend** | React + Vite + Tailwind CSS, Bauhaus design system |
| **Inference** | Runs entirely in the browser via JavaScript |

## Interactive Chapters

1. **Name Cloud** — Browse the full dataset
2. **Tokenization** — See NFD decomposition into Unicode characters
3. **Embedding** — Explore token & position embedding vectors
4. **Attention** — Full Q/K/V pipeline visualization with multi-head attention
5. **Loss & Gradient** — Per-position predictions and gradient flow
6. **Training** — Animated loss curve with replay
7. **Name Generator** — Generate names with temperature control

## Project Structure

```
├── model/
│   ├── data/in_name.txt          # Dataset (1,609 names)
│   ├── in_main.py                # Training script + weight export
│   └── scripts/
│       └── export_training_trace.py
├── app/
│   ├── public/data/              # Exported model weights & training trace
│   ├── src/
│   │   ├── components/           # React components (12 sections)
│   │   ├── gptInference.js       # In-browser GPT inference engine
│   │   └── App.jsx
│   └── index.html
```

## Run Locally

```bash
# Train model (optional — weights are pre-exported)
cd model && python3 in_main.py

# Start frontend
cd app && npm install && npx vite --port 5180
```

## Tech Stack

- **Backend**: Pure Python (custom autograd, no ML frameworks)
- **Frontend**: React 19, Vite 6, Tailwind CSS 4
- **Design**: Bauhaus — geometric purity, primary colors, hard shadows
- **Deploy**: Vercel (auto-deploy on push)

## Credits

Inspired by [Karpathy's microGPT](https://github.com/karpathy/microGPT) and [ko-microgpt](https://github.com/woduq1414/ko-microgpt).
