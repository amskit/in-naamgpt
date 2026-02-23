import { useState, useMemo } from 'react'
import { rmsNorm } from '../gptInference'

const COLORS = ['#D02020', '#1040C0', '#F0C020']

function ValueBar({ value, maxAbs }) {
  const ratio = Math.min(Math.abs(value) / Math.max(maxAbs, 1e-6), 1)
  const isPos = value >= 0
  const barColor = isPos ? '#22c55e' : '#D02020'
  return (
    <div className="flex items-center gap-1.5 h-6">
      <div className="relative flex-1 h-4 bg-bh-muted border border-bh-black/10 overflow-hidden">
        {isPos ? (
          <div className="absolute left-1/2 top-0 h-full" style={{ width: `${ratio * 50}%`, backgroundColor: barColor + 'cc' }} />
        ) : (
          <div className="absolute right-1/2 top-0 h-full" style={{ width: `${ratio * 50}%`, backgroundColor: barColor + 'cc' }} />
        )}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-bh-black/20" />
      </div>
      <span className={`w-14 text-right text-[10px] font-mono font-bold ${isPos ? 'text-green-700' : 'text-red-700'}`}>
        {value.toFixed(2)}
      </span>
    </div>
  )
}

function VectorColumn({ title, titleColor, values, maxAbs }) {
  return (
    <div className="flex-1 min-w-[180px]">
      <div className="flex items-center justify-between mb-2 px-1">
        <h4 className="font-heading font-bold text-xs uppercase tracking-widest" style={{ color: titleColor }}>{title}</h4>
        <span className="text-[9px] text-bh-fg/30 cursor-help" title="Info">?</span>
      </div>
      <div className="border-2 border-bh-black bg-bh-white p-2 shadow-bh-sm space-y-[1px]">
        {values.map((val, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <span className="w-5 text-center text-[9px] font-mono font-bold bg-bh-muted text-bh-fg/50 border border-bh-black/10 shrink-0">{i}</span>
            <div className="flex-1"><ValueBar value={val} maxAbs={maxAbs} /></div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function EmbeddingSection({ snapshot }) {
  const [tokenIndex, setTokenIndex] = useState(0)
  const [posIndex, setPosIndex] = useState(0)

  const tokenChars = useMemo(() => {
    if (!snapshot?.tokenizer?.uchars) return []
    return [...snapshot.tokenizer.uchars.map((ch) => ch.normalize('NFC') || ch), '[BOS]']
  }, [snapshot])

  const vocabSize = tokenChars.length
  const blockSize = snapshot?.block_size || 32
  const nEmbd = snapshot?.n_embd || 16

  const tokenEmb = useMemo(() => snapshot?.wte?.[tokenIndex] || Array(nEmbd).fill(0), [snapshot, tokenIndex, nEmbd])
  const posEmb = useMemo(() => snapshot?.wpe?.[posIndex] || Array(nEmbd).fill(0), [snapshot, posIndex, nEmbd])
  const sumEmb = useMemo(() => tokenEmb.map((t, i) => t + posEmb[i]), [tokenEmb, posEmb])
  const finalEmb = useMemo(() => rmsNorm(sumEmb), [sumEmb])

  const maxAbs = useMemo(() => {
    let m = 0
    for (const arr of [tokenEmb, posEmb, sumEmb, finalEmb]) for (const v of arr) m = Math.max(m, Math.abs(v))
    return m
  }, [tokenEmb, posEmb, sumEmb, finalEmb])

  const getTokenCategory = (idx) => {
    if (!snapshot?.tokenizer?.uchars || idx >= snapshot.tokenizer.uchars.length) return 'BOS'
    const code = snapshot.tokenizer.uchars[idx].codePointAt(0)
    if (code >= 0x0915 && code <= 0x0939) return 'Consonant'
    if (code >= 0x0905 && code <= 0x0914) return 'Vowel'
    if (code >= 0x093E && code <= 0x094C) return 'Matra'
    if (code === 0x094D) return 'Virama'
    if (code === 0x0902) return 'Anusvara'
    return 'Other'
  }

  if (!snapshot) return (
    <section className="py-16 px-6 text-center border-b-4 border-bh-black">
      <span className="font-bold text-bh-fg/50">Loading embeddings...</span>
    </section>
  )

  return (
    <section className="relative py-12 md:py-24 px-4 md:px-8 border-b-4 border-bh-black">
      <div className="max-w-7xl mx-auto">
        <span className="inline-block px-3 py-1 bg-bh-blue text-white font-bold text-xs uppercase tracking-widest border-2 border-bh-black shadow-bh-sm mb-4">
          Chapter 03
        </span>
        <h2 className="font-heading font-black text-4xl sm:text-6xl lg:text-7xl uppercase tracking-tighter leading-[0.9] text-bh-black mb-3">
          Embedding
        </h2>
        <p className="text-base text-bh-fg/60 font-medium max-w-3xl mb-8">
          The model input embedding is built by adding <strong className="text-bh-red">token embedding</strong> and{' '}
          <strong className="text-bh-blue">position embedding</strong>. The final vector changes depending on which character appears at which position.
        </p>

        {/* Selectors */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
          <div className="p-4 bg-bh-white border-4 border-bh-black shadow-bh">
            <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/50 mb-2">Sample Character</div>
            <div className="flex items-center gap-3">
              <button onClick={() => setTokenIndex((p) => (p - 1 + vocabSize) % vocabSize)}
                className="btn-press w-9 h-9 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">◀</button>
              <div className="flex-1 text-center px-3 py-2 border-2 border-bh-black bg-bh-bg">
                <div className="font-heading font-black text-2xl">{tokenChars[tokenIndex]}</div>
                <div className="text-[9px] text-bh-fg/40 font-mono">{getTokenCategory(tokenIndex)} · ID {tokenIndex}</div>
              </div>
              <button onClick={() => setTokenIndex((p) => (p + 1) % vocabSize)}
                className="btn-press w-9 h-9 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">▶</button>
            </div>
          </div>
          <div className="p-4 bg-bh-white border-4 border-bh-black shadow-bh">
            <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/50 mb-2">Position Index</div>
            <div className="flex items-center gap-3">
              <button onClick={() => setPosIndex((p) => (p - 1 + blockSize) % blockSize)}
                className="btn-press w-9 h-9 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">◀</button>
              <div className="flex-1 text-center px-3 py-2 border-2 border-bh-black bg-bh-bg">
                <div className="font-heading font-black text-2xl">POS {posIndex}</div>
                <div className="text-[9px] text-bh-fg/40 font-mono">0 – {blockSize - 1}</div>
              </div>
              <button onClick={() => setPosIndex((p) => (p + 1) % blockSize)}
                className="btn-press w-9 h-9 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">▶</button>
            </div>
          </div>
        </div>

        {/* 4-column vector view */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          <VectorColumn title="Token Embedding" titleColor={COLORS[0]} values={tokenEmb} maxAbs={maxAbs} />
          <VectorColumn title="Position Embedding" titleColor={COLORS[1]} values={posEmb} maxAbs={maxAbs} />
          <VectorColumn title="Sum Embedding" titleColor={COLORS[2]} values={sumEmb} maxAbs={maxAbs} />
          <VectorColumn title="Final Embedding" titleColor="#121212" values={finalEmb} maxAbs={maxAbs} />
        </div>

        {/* Formula */}
        <div className="mt-6 inline-flex items-center gap-2 px-5 py-3 bg-bh-white border-4 border-bh-black shadow-bh font-mono text-sm">
          <span className="text-bh-red font-bold">wte[{tokenIndex}]</span>
          <span className="text-bh-fg/40">+</span>
          <span className="text-bh-blue font-bold">wpe[{posIndex}]</span>
          <span className="text-bh-fg/40">→</span>
          <span className="font-bold" style={{ color: COLORS[2] }}>sum</span>
          <span className="text-bh-fg/40">→</span>
          <span className="text-bh-black font-bold">RMSNorm → x</span>
        </div>
      </div>
    </section>
  )
}
