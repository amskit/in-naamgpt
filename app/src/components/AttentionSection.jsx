import { useState, useMemo } from 'react'
import { softmax, matVec, rmsNorm, dotProduct } from '../gptInference'

const COLORS = ['#D02020', '#1040C0', '#F0C020']
const EXAMPLE_NAMES = ['आरव', 'प्रिया', 'कृष्ण']

function ValueBar({ value, maxAbs }) {
  const ratio = Math.min(Math.abs(value) / Math.max(maxAbs, 1e-6), 1)
  const isPos = value >= 0
  const barColor = isPos ? '#22c55e' : '#D02020'
  return (
    <div className="flex items-center gap-1 h-5">
      <div className="relative flex-1 h-3.5 bg-bh-muted border border-bh-black/10 overflow-hidden">
        {isPos ? (
          <div className="absolute left-1/2 top-0 h-full" style={{ width: `${ratio*50}%`, backgroundColor: barColor+'cc' }} />
        ) : (
          <div className="absolute right-1/2 top-0 h-full" style={{ width: `${ratio*50}%`, backgroundColor: barColor+'cc' }} />
        )}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-bh-black/15" />
      </div>
      <span className={`w-12 text-right text-[9px] font-mono font-bold ${isPos?'text-green-700':'text-red-700'}`}>{value.toFixed(2)}</span>
    </div>
  )
}

function SmallVec({ values, color }) {
  return (
    <div className="flex gap-0.5">
      {values.map((v,i) => (
        <div key={i} className="flex-1 text-center text-[9px] font-mono font-bold py-1 border border-bh-black/10"
          style={{ backgroundColor: v>=0 ? `rgba(34,197,94,${Math.min(Math.abs(v)*0.25,0.5).toFixed(2)})` : `rgba(208,32,32,${Math.min(Math.abs(v)*0.25,0.5).toFixed(2)})`, color: Math.abs(v)>1.5?'#000':'#555' }}>
          {v.toFixed(2)}
        </div>
      ))}
    </div>
  )
}

function VectorPanel({ title, titleColor, values, maxAbs }) {
  return (
    <div className="border-2 border-bh-black bg-bh-white p-2 shadow-bh-sm">
      <div className="flex items-center justify-between mb-1.5 px-0.5">
        <h4 className="font-heading font-bold text-[10px] uppercase tracking-widest" style={{ color: titleColor }}>{title}</h4>
        <span className="text-[8px] text-bh-fg/20 cursor-help">?</span>
      </div>
      <div className="space-y-px">
        {values.map((val,i) => (
          <div key={i} className="flex items-center gap-1">
            <span className="w-4 text-center text-[8px] font-mono font-bold bg-bh-muted text-bh-fg/40 shrink-0">{i}</span>
            <div className="flex-1"><ValueBar value={val} maxAbs={maxAbs} /></div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function AttentionSection({ snapshot }) {
  const [nameIdx, setNameIdx] = useState(0)
  const [posIdx, setPosIdx] = useState(0)
  const [headIdx, setHeadIdx] = useState(0)
  const name = EXAMPLE_NAMES[nameIdx]

  const tokenData = useMemo(() => {
    if (!snapshot?.tokenizer?.uchars) return null
    const { uchars, bos } = snapshot.tokenizer
    const nfd = name.normalize('NFD')
    const chars = [...nfd]
    const ids = [bos, ...chars.map(ch => uchars.indexOf(ch)).filter(id => id >= 0)]
    const labels = ['[BOS]', ...chars.map(ch => ch.normalize('NFC') || ch)]
    return { ids, labels, seqLen: ids.length }
  }, [snapshot, name])

  const maxPos = tokenData ? tokenData.seqLen - 1 : 0

  const fwd = useMemo(() => {
    if (!snapshot || !tokenData || tokenData.seqLen === 0) return null
    const { wte, wpe, attention, mlp, lm_head } = snapshot
    const { attn_wq, attn_wk, attn_wv, attn_wo, n_head, head_dim } = attention
    const { mlp_fc1, mlp_fc2 } = mlp
    const { ids } = tokenData
    const tp = Math.min(posIdx, ids.length - 1)

    const allK = [], allV = []
    let finalX, finalQ, attnW, headOuts, xConcat, xLinear, xRes, xMlp

    for (let p = 0; p <= tp; p++) {
      const tid = ids[p]
      let x = wte[tid].map((t,i) => t + wpe[p][i])
      x = rmsNorm(x)
      const xR1 = [...x]
      x = rmsNorm(x)
      const q = matVec(attn_wq, x), k = matVec(attn_wk, x), v = matVec(attn_wv, x)
      allK.push(k); allV.push(v)

      if (p === tp) {
        finalQ = q
        const hOuts = [], hWeights = [], xA = []
        for (let h = 0; h < n_head; h++) {
          const hs = h * head_dim, qH = q.slice(hs, hs+head_dim)
          const scores = []
          for (let t = 0; t <= tp; t++) scores.push(dotProduct(qH, allK[t].slice(hs,hs+head_dim)) / Math.sqrt(head_dim))
          const w = softmax(scores)
          hWeights.push(w)
          const ho = []
          for (let j = 0; j < head_dim; j++) {
            let val = 0; for (let t = 0; t <= tp; t++) val += w[t] * allV[t][hs+j]
            ho.push(val)
          }
          hOuts.push(ho); xA.push(...ho)
        }
        attnW = hWeights; headOuts = hOuts; xConcat = xA
        xLinear = matVec(attn_wo, xA)
        xRes = xLinear.map((a,i) => a + xR1[i])
        const xR2 = [...xRes]
        let mx = rmsNorm(xRes)
        mx = matVec(mlp_fc1, mx).map(v => Math.max(0,v))
        mx = matVec(mlp_fc2, mx)
        xMlp = mx.map((a,i) => a + xR2[i])
        finalX = xMlp
      }
    }

    const logits = matVec(lm_head, finalX)
    const probs = softmax(logits)
    const indexed = probs.map((p,i) => ({ prob:p, logit:logits[i], id:i })).sort((a,b) => b.prob-a.prob)

    return {
      finalEmb: rmsNorm(wte[ids[tp]].map((t,i) => t+wpe[tp][i])),
      q: finalQ, attnW, headOuts, xConcat, xLinear, xRes, xMlp,
      logits, probs, top10: indexed.slice(0,10), bottom2: indexed.slice(-2).reverse(),
    }
  }, [snapshot, tokenData, posIdx])

  if (!snapshot || !tokenData) return (
    <section className="py-16 px-6 text-center border-b-4 border-bh-black bg-bh-blue/5">
      <span className="font-bold text-bh-fg/50">Loading attention data...</span>
    </section>
  )

  const { n_head, head_dim } = snapshot.attention
  const { uchars, bos } = snapshot.tokenizer
  const getLabel = (id) => id === bos ? '[BOS]' : (uchars[id]?.normalize('NFC') || `ID${id}`)

  return (
    <section className="relative py-12 md:py-24 px-4 md:px-8 border-b-4 border-bh-black bg-bh-blue/5">
      <div className="max-w-7xl mx-auto">
        <span className="inline-block px-3 py-1 bg-bh-red text-white font-bold text-xs uppercase tracking-widest border-2 border-bh-black shadow-bh-sm mb-4">Chapter 04</span>
        <h2 className="font-heading font-black text-4xl sm:text-6xl lg:text-7xl uppercase tracking-tighter leading-[0.9] text-bh-black mb-3">Attention</h2>
        <p className="text-base text-bh-fg/60 font-medium max-w-3xl mb-8">
          Given the selected name and index, we compute Q, K, and V from Final Embedding (x), derive Attention Output,
          and finally calculate the probability of each possible next token.
        </p>

        {/* Selectors */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
          <div className="p-4 bg-bh-white border-4 border-bh-black shadow-bh">
            <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/50 mb-2">Example Name</div>
            <div className="flex items-center gap-3">
              <button onClick={() => { setNameIdx(p => (p-1+3)%3); setPosIdx(0) }} className="btn-press w-8 h-8 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">◀</button>
              <div className="flex-1 text-center px-3 py-2 border-2 border-bh-black bg-bh-bg">
                <div className="font-heading font-black text-xl">{name}</div>
                <div className="text-[9px] text-bh-fg/40 font-mono">POS 0 – {maxPos}</div>
              </div>
              <button onClick={() => { setNameIdx(p => (p+1)%3); setPosIdx(0) }} className="btn-press w-8 h-8 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">▶</button>
            </div>
          </div>
          <div className="p-4 bg-bh-white border-4 border-bh-black shadow-bh">
            <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/50 mb-2">Target Index</div>
            <div className="flex items-center gap-3">
              <button onClick={() => setPosIdx(p => (p-1+tokenData.seqLen)%tokenData.seqLen)} className="btn-press w-8 h-8 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">◀</button>
              <div className="flex-1 text-center px-3 py-2 border-2 border-bh-black bg-bh-bg">
                <div className="font-heading font-black text-xl">POS {posIdx}</div>
                <div className="text-[9px] text-bh-fg/40 font-mono">{tokenData.labels[posIdx]} · ID {tokenData.ids[posIdx]}</div>
              </div>
              <button onClick={() => setPosIdx(p => (p+1)%tokenData.seqLen)} className="btn-press w-8 h-8 bg-bh-yellow border-2 border-bh-black shadow-bh-sm font-black text-sm">▶</button>
            </div>
          </div>
        </div>

        {fwd && (
          <>
            {/* Row 1: Embedding + Q/K/V + Attn Weights + Output */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 mb-6">
              <VectorPanel title={`Final Embedding · POS ${posIdx}`} titleColor={COLORS[0]} values={fwd.finalEmb} maxAbs={Math.max(...fwd.finalEmb.map(Math.abs),0.01)} />
              <div className="border-2 border-bh-black bg-bh-white p-2 shadow-bh-sm">
                <div className="flex items-center justify-between mb-1.5">
                  <h4 className="font-heading font-bold text-[10px] uppercase tracking-widest text-bh-blue">Q / K / V</h4>
                  <div className="flex gap-0.5">
                    {Array.from({length:n_head},(_,h) => (
                      <button key={h} onClick={() => setHeadIdx(h)}
                        className={`px-1.5 py-0.5 text-[8px] font-bold border border-bh-black transition-all duration-200 ${headIdx===h?'bg-bh-blue text-white':'bg-bh-bg text-bh-fg/40 hover:bg-bh-muted'}`}>
                        H{h}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="space-y-1.5">
                  <div><div className="text-[8px] text-bh-fg/40 font-bold mb-0.5">Q (POS {posIdx})</div>
                    <SmallVec values={fwd.q.slice(headIdx*head_dim,(headIdx+1)*head_dim)} color={COLORS[1]} /></div>
                  <div><div className="text-[8px] text-bh-fg/40 font-bold mb-0.5">K (POS {posIdx})</div>
                    <SmallVec values={fwd.headOuts[headIdx]} color={COLORS[2]} /></div>
                </div>
              </div>
              <div className="border-2 border-bh-black bg-bh-white p-2 shadow-bh-sm">
                <h4 className="font-heading font-bold text-[10px] uppercase tracking-widest text-bh-red mb-1.5">Attn Weights · H{headIdx}</h4>
                <div className="space-y-0.5">
                  {fwd.attnW[headIdx].map((w,t) => (
                    <div key={t} className="flex items-center gap-1.5">
                      <span className="text-[9px] font-bold w-10 shrink-0 text-bh-fg/60">POS {t}</span>
                      <div className="flex-1 h-4 bg-bh-muted border border-bh-black/10 overflow-hidden">
                        <div className="h-full" style={{ width:`${(w*100).toFixed(1)}%`, backgroundColor:'#22c55ecc' }} />
                      </div>
                      <span className="text-[9px] font-mono w-12 text-right text-bh-fg/50">{w.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="border-2 border-bh-black bg-bh-white p-2 shadow-bh-sm">
                <h4 className="font-heading font-bold text-[10px] uppercase tracking-widest mb-1.5" style={{color:COLORS[2]}}>Attention Output</h4>
                <div className="mb-2">
                  <div className="text-[8px] text-bh-fg/40 font-bold mb-0.5">Head {headIdx} ({head_dim}-dim)</div>
                  <SmallVec values={fwd.headOuts[headIdx]} color={COLORS[2]} />
                </div>
              </div>
            </div>

            {/* Row 2: Head outputs + concat + block result */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
              <div className="border-2 border-bh-black bg-bh-white p-2 shadow-bh-sm">
                <h4 className="font-heading font-bold text-[10px] uppercase tracking-widest text-bh-blue mb-1.5">Head 0–{n_head-1} Outputs</h4>
                {fwd.headOuts.map((ho,h) => (
                  <div key={h} className="mb-1.5">
                    <div className="text-[8px] font-bold mb-0.5" style={{color:COLORS[h%3]}}>HEAD {h}</div>
                    <SmallVec values={ho} color={COLORS[h%3]} />
                  </div>
                ))}
              </div>
              <VectorPanel title="Multi-Head → linear(W_O)" titleColor={COLORS[1]} values={fwd.xLinear} maxAbs={Math.max(...fwd.xLinear.map(Math.abs),0.01)} />
              <VectorPanel title="Attn Block Result (x+res)" titleColor={COLORS[0]} values={fwd.xRes} maxAbs={Math.max(...fwd.xRes.map(Math.abs),0.01)} />
            </div>

            {/* Row 3: Transformer output + Logits + Probs */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <VectorPanel title="Transformer Block Output" titleColor="#121212" values={fwd.xMlp} maxAbs={Math.max(...fwd.xMlp.map(Math.abs),0.01)} />
              <div className="border-2 border-bh-black bg-bh-white p-2 shadow-bh-sm">
                <div className="flex items-center justify-between mb-1.5">
                  <h4 className="font-heading font-bold text-[10px] uppercase tracking-widest text-bh-red">Logit</h4>
                  <span className="text-[8px] px-1.5 py-0.5 bg-bh-muted text-bh-fg/40 font-bold border border-bh-black/10">TOP10 + BOTTOM2</span>
                </div>
                <div className="space-y-px">
                  {fwd.top10.map((item,i) => (
                    <div key={item.id} className="flex items-center gap-1 py-0.5">
                      <span className="text-[9px] font-bold w-20 truncate" style={{color:COLORS[i%3]}}>{getLabel(item.id)} · {item.id}</span>
                      <div className="flex-1 h-3.5 bg-bh-muted border border-bh-black/10 overflow-hidden">
                        <div className="h-full" style={{width:`${(Math.abs(item.logit)/Math.max(...fwd.top10.map(t=>Math.abs(t.logit)),0.01)*100).toFixed(1)}%`,backgroundColor:item.logit>=0?'#22c55eaa':'#D02020aa'}} />
                      </div>
                      <span className={`text-[9px] font-mono w-12 text-right ${item.logit>=0?'text-green-700':'text-red-700'}`}>{item.logit.toFixed(2)}</span>
                    </div>
                  ))}
                  <div className="border-t-2 border-dashed border-bh-black/10 my-0.5" />
                  {fwd.bottom2.map(item => (
                    <div key={item.id} className="flex items-center gap-1 py-0.5 opacity-50">
                      <span className="text-[9px] font-bold w-20 truncate text-bh-fg/40">{getLabel(item.id)} · {item.id}</span>
                      <div className="flex-1 h-3.5 bg-bh-muted border border-bh-black/10 overflow-hidden"><div className="h-full bg-red-300/40" style={{width:'20%'}} /></div>
                      <span className="text-[9px] font-mono w-12 text-right text-red-700">{item.logit.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="border-2 border-bh-black bg-bh-white p-2 shadow-bh-sm">
                <div className="flex items-center justify-between mb-1.5">
                  <h4 className="font-heading font-bold text-[10px] uppercase tracking-widest text-bh-blue">Next Token Prob</h4>
                  <span className="text-[8px] px-1.5 py-0.5 bg-bh-muted text-bh-fg/40 font-bold border border-bh-black/10">SOFTMAX</span>
                </div>
                <div className="space-y-px">
                  {fwd.top10.map((item,i) => (
                    <div key={item.id} className="flex items-center gap-1 py-0.5">
                      <span className="text-[9px] font-bold" style={{color:COLORS[i%3]}}>#{i+1}</span>
                      <span className="text-[9px] font-bold w-20 truncate" style={{color:COLORS[i%3]}}>{getLabel(item.id)}</span>
                      <div className="flex-1 h-3.5 bg-bh-muted border border-bh-black/10 overflow-hidden">
                        <div className="h-full" style={{width:`${(item.prob/fwd.top10[0].prob*100).toFixed(1)}%`, backgroundColor:COLORS[i%3]+'80'}} />
                      </div>
                      <span className="text-[9px] font-mono w-16 text-right text-bh-fg/60">{item.prob.toFixed(6)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  )
}
