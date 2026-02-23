import { useState, useMemo } from 'react'
import { softmax, matVec, rmsNorm, dotProduct } from '../gptInference'

const COLORS = ['#D02020', '#1040C0', '#F0C020']
const EXAMPLE_NAMES = ['आरव', 'प्रिया', 'कृष्ण']

function fullForwardPass(name, snapshot) {
  if (!snapshot?.wte || !snapshot?.tokenizer?.uchars) return null
  const { wte, wpe, attention, mlp, lm_head, block_size } = snapshot
  const { attn_wq, attn_wk, attn_wv, attn_wo, n_head, head_dim } = attention
  const { mlp_fc1, mlp_fc2 } = mlp
  const { uchars, bos } = snapshot.tokenizer
  const vocabSize = uchars.length + 1
  const nfd = name.normalize('NFD')
  const chars = [...nfd]
  const tokenIds = [bos, ...chars.map(ch => uchars.indexOf(ch)).filter(id => id >= 0), bos]
  const tokenLabels = ['[BOS]', ...chars.map(ch => ch.normalize('NFC') || ch), '[BOS]']
  const seqLen = Math.min(block_size, tokenIds.length - 1)
  const keysCache = [], valsCache = [], positions = []

  for (let pos = 0; pos < seqLen; pos++) {
    const tid = tokenIds[pos], targetId = tokenIds[pos + 1]
    let x = wte[tid].map((t, i) => t + wpe[pos][i])
    x = rmsNorm(x)
    const xR1 = [...x]; x = rmsNorm(x)
    const q = matVec(attn_wq, x), k = matVec(attn_wk, x), v = matVec(attn_wv, x)
    keysCache.push(k); valsCache.push(v)
    const xA = []
    for (let h = 0; h < n_head; h++) {
      const hs = h * head_dim, qH = q.slice(hs, hs + head_dim), scores = []
      for (let t = 0; t <= pos; t++) scores.push(dotProduct(qH, keysCache[t].slice(hs, hs + head_dim)) / Math.sqrt(head_dim))
      const w = softmax(scores)
      for (let j = 0; j < head_dim; j++) { let val = 0; for (let t = 0; t <= pos; t++) val += w[t] * valsCache[t][hs + j]; xA.push(val) }
    }
    x = matVec(attn_wo, xA).map((a, i) => a + xR1[i])
    const xR2 = [...x]; let mx = rmsNorm(x); mx = matVec(mlp_fc1, mx).map(v => Math.max(0, v)); mx = matVec(mlp_fc2, mx)
    x = mx.map((a, i) => a + xR2[i])
    const logits = matVec(lm_head, x), probs = softmax(logits)
    const indexed = probs.map((p, i) => ({ prob: p, id: i })).sort((a, b) => b.prob - a.prob)
    const targetProb = probs[targetId] || 1e-10
    const tokenLoss = -Math.log(Math.max(targetProb, 1e-10))
    positions.push({
      pos, inputId: tid, inputLabel: tokenLabels[pos], targetId,
      targetLabel: tokenLabels[pos + 1], top5: indexed.slice(0, 5),
      targetProb, tokenLoss,
      lossGradient: 1 / seqLen,
      targetProbGradient: -(1 / seqLen) / Math.max(targetProb, 1e-10),
    })
  }
  return { positions, totalLoss: positions.reduce((s, p) => s + p.tokenLoss, 0) / seqLen, seqLen, tokenLabels }
}

export default function LossGradientSection({ snapshot }) {
  const [nameIdx, setNameIdx] = useState(0)
  const name = EXAMPLE_NAMES[nameIdx]
  const result = useMemo(() => fullForwardPass(name, snapshot), [name, snapshot])
  const { uchars, bos } = snapshot?.tokenizer || {}
  const getLabel = (id) => id === bos ? '[BOS]' : (uchars?.[id]?.normalize('NFC') || `ID${id}`)

  if (!snapshot || !result) return (
    <section className="py-16 px-6 text-center border-b-4 border-bh-black"><span className="font-bold text-bh-fg/50">Loading...</span></section>
  )

  const maxTL = Math.max(...result.positions.map(p => p.tokenLoss), 0.01)

  return (
    <section className="relative py-12 md:py-24 px-4 md:px-8 border-b-4 border-bh-black bg-bh-red/5">
      <div className="max-w-7xl mx-auto">
        <span className="inline-block px-3 py-1 bg-bh-yellow text-bh-black font-bold text-xs uppercase tracking-widest border-2 border-bh-black shadow-bh-sm mb-4">Chapter 05</span>
        <h2 className="font-heading font-black text-4xl sm:text-6xl lg:text-7xl uppercase tracking-tighter leading-[0.9] text-bh-black mb-3">
          Loss <span className="text-bh-red">&</span> Gradient
        </h2>
        <p className="text-base text-bh-fg/60 font-medium max-w-3xl mb-8">
          At each position, we compare prediction and target token probability to compute loss.
          We then compute how much each parameter contributes to that loss via backpropagation.
        </p>

        {/* Name selector */}
        <div className="flex items-center gap-3 mb-8">
          <span className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/50">Name</span>
          <button onClick={() => setNameIdx(p => (p - 1 + 3) % 3)} className="btn-press w-8 h-8 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">◀</button>
          <div className="px-5 py-2 border-4 border-bh-black bg-bh-white shadow-bh font-heading font-black text-xl">{name}</div>
          <button onClick={() => setNameIdx(p => (p + 1) % 3)} className="btn-press w-8 h-8 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">▶</button>
        </div>

        {/* Per-position predictions */}
        <div className="mb-8 overflow-x-auto">
          <div className="flex gap-2 min-w-max pb-3">
            {result.positions.map((pos, pi) => (
              <div key={pi} className="w-44 shrink-0 border-2 border-bh-black bg-bh-white p-2.5 shadow-bh-sm">
                <div className="font-heading font-bold text-xs uppercase tracking-widest mb-1.5" style={{ color: COLORS[pi % 3] }}>POS {pos.pos}</div>
                <div className="text-[9px] text-bh-fg/40 font-bold mb-2">Target: <span className="text-bh-fg/70">{pos.targetLabel}</span></div>
                <div className="space-y-0.5">
                  {pos.top5.map((item, i) => {
                    const isTarget = item.id === pos.targetId
                    return (
                      <div key={item.id} className={`flex items-center gap-1 px-1 py-0.5 ${isTarget ? 'bg-bh-yellow/30 border border-bh-yellow' : ''}`}>
                        <span className={`text-[8px] font-bold ${isTarget ? 'text-bh-red' : 'text-bh-fg/30'}`}>#{i + 1}</span>
                        <span className={`text-[9px] font-bold flex-1 truncate ${isTarget ? 'text-bh-black' : 'text-bh-fg/50'}`}>{getLabel(item.id)}</span>
                        <span className={`text-[8px] font-mono ${isTarget ? 'text-bh-red font-bold' : 'text-bh-fg/40'}`}>{item.prob.toFixed(3)}</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Token Loss */}
        <div className="mb-6">
          <div className="font-heading font-bold text-xs uppercase tracking-widest text-bh-fg/50 mb-2">Token Loss = -log(prob)</div>
          <div className="flex gap-2 overflow-x-auto pb-2">
            {result.positions.map((pos, pi) => {
              const intensity = Math.min(pos.tokenLoss / maxTL, 1)
              return (
                <div key={pi} className="shrink-0 w-24 text-center">
                  <div className="text-[9px] text-bh-fg/40 font-bold mb-1">POS {pos.pos}</div>
                  <div className="h-7 border-2 border-bh-black flex items-center justify-center font-mono text-xs font-bold"
                    style={{ backgroundColor: `rgba(208,32,32,${(0.15 + intensity * 0.55).toFixed(2)})`, color: intensity > 0.4 ? '#fff' : '#333' }}>
                    {pos.tokenLoss.toFixed(3)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Final Loss */}
        <div className="flex justify-center mb-8">
          <div className="px-10 py-5 bg-bh-black text-white text-center border-4 border-bh-black shadow-bh-lg">
            <div className="text-[10px] font-bold uppercase tracking-widest text-white/60 mb-1">Final Loss</div>
            <div className="font-heading font-black text-4xl">{result.totalLoss.toFixed(3)}</div>
          </div>
        </div>

        {/* Gradient rows */}
        <div className="mb-4">
          <div className="font-heading font-bold text-xs uppercase tracking-widest text-bh-fg/50 mb-2">Token Loss Gradient (1/N)</div>
          <div className="flex gap-2 overflow-x-auto pb-2">
            {result.positions.map((pos, pi) => (
              <div key={pi} className="shrink-0 w-24 text-center">
                <div className="text-[9px] text-bh-fg/40 font-bold mb-1">POS {pos.pos}</div>
                <div className="h-7 border-2 border-bh-black flex items-center justify-center font-mono text-xs font-bold" style={{ backgroundColor: 'rgba(16,64,192,0.2)', color: '#1040C0' }}>
                  {pos.lossGradient.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="mb-6">
          <div className="font-heading font-bold text-xs uppercase tracking-widest text-bh-fg/50 mb-2">Target Probability Gradient</div>
          <div className="flex gap-2 overflow-x-auto pb-2">
            {result.positions.map((pos, pi) => {
              const maxG = Math.max(...result.positions.map(p => Math.abs(p.targetProbGradient)), 0.01)
              const intensity = Math.min(Math.abs(pos.targetProbGradient) / maxG, 1)
              return (
                <div key={pi} className="shrink-0 w-24 text-center">
                  <div className="text-[9px] text-bh-fg/40 font-bold mb-1">POS {pos.pos}</div>
                  <div className="h-7 border-2 border-bh-black flex items-center justify-center font-mono text-[10px] font-bold"
                    style={{ backgroundColor: `rgba(208,32,32,${(0.1 + intensity * 0.4).toFixed(2)})`, color: intensity > 0.3 ? '#fff' : '#555' }}>
                    {pos.targetProbGradient.toFixed(3)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Formula */}
        <div className="inline-flex items-center gap-2 px-5 py-3 bg-bh-white border-4 border-bh-black shadow-bh font-mono text-sm">
          <span className="text-bh-red font-bold">loss</span>
          <span className="text-bh-fg/40">=</span>
          <span className="text-bh-blue font-bold">1/{result.seqLen}</span>
          <span className="text-bh-fg/40">×</span>
          <span style={{ color: COLORS[2] }} className="font-bold">Σ -log(P(target))</span>
          <span className="text-bh-fg/40">=</span>
          <span className="text-bh-black font-bold">{result.totalLoss.toFixed(4)}</span>
        </div>
      </div>
    </section>
  )
}
