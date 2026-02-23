import { useState, useEffect, useMemo, useRef } from 'react'

const COLORS = ['#D02020', '#1040C0', '#F0C020']

export default function TrainingSection() {
  const [trace, setTrace] = useState(null)
  const [status, setStatus] = useState('loading')
  const [hoveredStep, setHoveredStep] = useState(null)
  const [playStep, setPlayStep] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const canvasRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => {
    let active = true
    fetch('/data/in_training_trace.json')
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then(data => { if (active) { setTrace(data); setStatus('ready') } })
      .catch(() => { if (active) setStatus('error') })
    return () => { active = false }
  }, [])

  const lossData = useMemo(() => {
    if (!trace?.steps) return []
    return trace.steps.filter(s => s.loss !== null).map(s => ({ step: s.step, loss: s.loss, word: s.word, lr: s.learning_rate }))
  }, [trace])

  const maxLoss = useMemo(() => lossData.length ? Math.max(...lossData.map(d => d.loss)) : 4, [lossData])
  const minLoss = useMemo(() => lossData.length ? Math.min(...lossData.map(d => d.loss)) : 0, [lossData])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !lossData.length) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const w = rect.width, h = rect.height
    const pad = { top: 20, right: 20, bottom: 40, left: 50 }
    const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom
    const displaySteps = playStep !== null ? lossData.slice(0, playStep + 1) : lossData
    const rangeTop = maxLoss * 1.05, rangeBottom = Math.max(0, minLoss * 0.9)

    ctx.clearRect(0, 0, w, h)

    // Grid
    ctx.strokeStyle = '#E0E0E0'; ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH * i) / 4
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke()
      ctx.fillStyle = '#888'; ctx.font = '11px monospace'; ctx.textAlign = 'right'
      ctx.fillText((rangeTop - ((rangeTop - rangeBottom) * i) / 4).toFixed(2), pad.left - 6, y + 4)
    }
    ctx.fillStyle = '#888'; ctx.textAlign = 'center'
    for (let s = 0; s <= 1000; s += 200) ctx.fillText(String(s), pad.left + (s / 1000) * plotW, h - pad.bottom + 20)

    if (displaySteps.length > 1) {
      // Smooth
      const smoothW = 20
      const smoothed = displaySteps.map((d, i) => {
        const start = Math.max(0, i - smoothW)
        return displaySteps.slice(start, i + 1).reduce((a, b) => a + b.loss, 0) / (i - start + 1)
      })

      // Raw (faint)
      ctx.beginPath(); ctx.strokeStyle = '#D0202030'; ctx.lineWidth = 1
      displaySteps.forEach((d, i) => {
        const x = pad.left + (d.step / 1000) * plotW
        const y = pad.top + ((rangeTop - d.loss) / (rangeTop - rangeBottom)) * plotH
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.stroke()

      // Smoothed (bold)
      ctx.beginPath(); ctx.lineWidth = 3; ctx.strokeStyle = '#D02020'
      smoothed.forEach((val, i) => {
        const x = pad.left + (displaySteps[i].step / 1000) * plotW
        const y = pad.top + ((rangeTop - val) / (rangeTop - rangeBottom)) * plotH
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.stroke()

      // Current dot
      const last = displaySteps[displaySteps.length - 1]
      const lastS = smoothed[smoothed.length - 1]
      const cx = pad.left + (last.step / 1000) * plotW
      const cy = pad.top + ((rangeTop - lastS) / (rangeTop - rangeBottom)) * plotH
      ctx.beginPath(); ctx.fillStyle = '#1040C0'; ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill()
      ctx.strokeStyle = '#121212'; ctx.lineWidth = 2; ctx.stroke()
    }

    if (hoveredStep !== null && lossData[hoveredStep]) {
      const d = lossData[hoveredStep]
      const x = pad.left + (d.step / 1000) * plotW
      ctx.beginPath(); ctx.strokeStyle = '#12121240'; ctx.lineWidth = 1; ctx.setLineDash([4, 4])
      ctx.moveTo(x, pad.top); ctx.lineTo(x, h - pad.bottom); ctx.stroke(); ctx.setLineDash([])
    }
  }, [lossData, maxLoss, minLoss, hoveredStep, playStep])

  const startPlayback = () => { setIsPlaying(true); setPlayStep(0) }
  useEffect(() => {
    if (!isPlaying || playStep === null) return
    if (playStep >= lossData.length - 1) { setIsPlaying(false); setPlayStep(null); return }
    timerRef.current = setTimeout(() => setPlayStep(s => Math.min((s || 0) + 5, lossData.length - 1)), 16)
    return () => clearTimeout(timerRef.current)
  }, [isPlaying, playStep, lossData.length])

  const handleHover = (e) => {
    if (!lossData.length || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const ratio = (e.clientX - rect.left - 50) / (rect.width - 70)
    setHoveredStep(Math.max(0, Math.min(lossData.length - 1, Math.round(ratio * (lossData.length - 1)))))
  }

  const hovered = hoveredStep !== null ? lossData[hoveredStep] : null

  return (
    <section className="relative py-12 md:py-24 px-4 md:px-8 border-b-4 border-bh-black bg-bh-yellow/20">
      <div className="max-w-7xl mx-auto">
        <span className="inline-block px-3 py-1 bg-bh-blue text-white font-bold text-xs uppercase tracking-widest border-2 border-bh-black shadow-bh-sm mb-4">Chapter 06</span>
        <h2 className="font-heading font-black text-4xl sm:text-6xl lg:text-7xl uppercase tracking-tighter leading-[0.9] text-bh-black mb-3">Training</h2>
        <p className="text-base text-bh-fg/60 font-medium max-w-xl mb-8">Watch the loss decrease over 1,000 training steps.</p>

        {status === 'loading' && <div className="text-center py-12 font-bold text-bh-fg/50">Loading training trace...</div>}
        {status === 'error' && <div className="text-center py-12"><div className="inline-block px-5 py-3 border-4 border-bh-black bg-bh-white shadow-bh"><span className="font-bold text-bh-red">Training trace not available. Run export_training_trace.py first.</span></div></div>}

        {status === 'ready' && lossData.length > 0 && (
          <>
            <div className="flex gap-3 mb-6">
              <button onClick={startPlayback} disabled={isPlaying}
                className="btn-press px-6 py-3 bg-bh-red text-white font-bold text-sm uppercase tracking-wider border-4 border-bh-black shadow-bh-lg disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-bh-blue focus:ring-offset-2">
                {isPlaying ? '▶ Playing...' : '▶ Replay Training'}
              </button>
              {isPlaying && (
                <button onClick={() => { setIsPlaying(false); setPlayStep(null) }}
                  className="btn-press px-5 py-3 bg-bh-white text-bh-black font-bold text-sm uppercase tracking-wider border-4 border-bh-black shadow-bh">
                  ■ Stop
                </button>
              )}
            </div>

            <div className="p-4 bg-bh-white border-4 border-bh-black shadow-bh-lg">
              <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/50 mb-3 text-center">Cross-Entropy Loss Over Training</div>
              <canvas ref={canvasRef} className="w-full cursor-crosshair" style={{ height: '280px' }} onMouseMove={handleHover} onMouseLeave={() => setHoveredStep(null)} />
              {hovered && (
                <div className="mt-3 flex flex-wrap justify-center gap-6">
                  {[
                    { label: 'Step', value: hovered.step, color: COLORS[0] },
                    { label: 'Loss', value: hovered.loss.toFixed(4), color: COLORS[1] },
                    { label: 'LR', value: hovered.lr.toFixed(6), color: COLORS[2] },
                    { label: 'Name', value: hovered.word, color: '#121212' },
                  ].map(item => (
                    <div key={item.label} className="text-center">
                      <div className="text-[9px] font-bold uppercase tracking-widest text-bh-fg/40">{item.label}</div>
                      <div className="font-heading font-bold text-base" style={{ color: item.color }}>{item.value}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {trace?.optimizer && (
              <div className="mt-6 grid grid-cols-2 md:grid-cols-5 border-4 border-bh-black divide-x-2 divide-bh-black bg-bh-white shadow-bh">
                {[
                  { k: 'Optimizer', v: trace.optimizer.name },
                  { k: 'LR', v: trace.optimizer.learning_rate },
                  { k: 'β₁', v: trace.optimizer.beta1 },
                  { k: 'β₂', v: trace.optimizer.beta2 },
                  { k: 'Steps', v: trace.num_steps },
                ].map((item, i) => (
                  <div key={i} className="px-3 py-3 text-center">
                    <div className="font-heading font-black text-base" style={{ color: COLORS[i % 3] }}>{item.v}</div>
                    <div className="text-[9px] text-bh-fg/40 font-bold uppercase tracking-widest">{item.k}</div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  )
}
