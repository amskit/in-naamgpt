import { useState, useCallback } from 'react'
import { generateNameWithSteps } from '../gptInference'

const COLORS = ['#D02020', '#1040C0', '#F0C020']

export default function GeneratorSection({ snapshot }) {
  const [names, setNames] = useState([])
  const [temperature, setTemperature] = useState(0.5)
  const [isGenerating, setIsGenerating] = useState(false)
  const [selectedName, setSelectedName] = useState(null)

  const handleGenerate = useCallback(() => {
    if (!snapshot) return
    setIsGenerating(true)
    setTimeout(() => {
      const batch = []
      for (let i = 0; i < 8; i++) batch.push(generateNameWithSteps(snapshot, temperature))
      setNames(batch)
      setSelectedName(null)
      setIsGenerating(false)
    }, 50)
  }, [snapshot, temperature])

  const handleSingle = useCallback(() => {
    if (!snapshot) return
    const result = generateNameWithSteps(snapshot, temperature)
    setNames(prev => [result, ...prev].slice(0, 24))
    setSelectedName(null)
  }, [snapshot, temperature])

  return (
    <section id="generator" className="relative py-12 md:py-24 px-4 md:px-8 border-b-4 border-bh-black bg-bh-blue">
      <div className="absolute inset-0 pattern-dots" style={{ opacity: 0.04 }} />
      <div className="relative z-10 max-w-7xl mx-auto">
        <span className="inline-block px-3 py-1 bg-bh-yellow text-bh-black font-bold text-xs uppercase tracking-widest border-2 border-bh-black shadow-bh-sm mb-4">
          Chapter 07
        </span>
        <h2 className="font-heading font-black text-4xl sm:text-6xl lg:text-7xl uppercase tracking-tighter leading-[0.9] text-white mb-3">
          Name Generator
        </h2>
        <p className="text-base text-white/70 font-medium max-w-xl mb-10">
          Click generate to hallucinate new Hindi names from a tiny neural network running entirely in your browser.
        </p>

        {/* Controls */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 mb-8">
          <div className="flex items-center gap-3 px-5 py-3 bg-bh-white border-4 border-bh-black shadow-bh">
            <label className="font-bold text-xs uppercase tracking-widest text-bh-fg/60">Temp</label>
            <input type="range" min="0.1" max="2.0" step="0.1" value={temperature}
              onChange={e => setTemperature(parseFloat(e.target.value))}
              className="w-28 accent-[#D02020]" />
            <span className="font-heading font-black text-lg text-bh-red w-10 text-center">{temperature.toFixed(1)}</span>
          </div>
          <button onClick={handleGenerate} disabled={!snapshot || isGenerating}
            className="btn-press px-8 py-3 bg-bh-red text-white font-bold text-sm uppercase tracking-wider border-4 border-bh-black shadow-bh-lg disabled:opacity-50 disabled:cursor-not-allowed">
            {isGenerating ? 'Generating...' : 'Generate 8 Names'}
          </button>
          <button onClick={handleSingle} disabled={!snapshot || isGenerating}
            className="btn-press px-6 py-3 bg-bh-yellow text-bh-black font-bold text-sm uppercase tracking-wider border-4 border-bh-black shadow-bh disabled:opacity-50 disabled:cursor-not-allowed">
            +1 More
          </button>
        </div>

        <div className="text-xs text-white/50 mb-8">
          {temperature < 0.3 && '❄ Very conservative — common patterns only'}
          {temperature >= 0.3 && temperature < 0.7 && '→ Balanced — realistic sounding names'}
          {temperature >= 0.7 && temperature < 1.2 && '→ Creative — more variety and surprises'}
          {temperature >= 1.2 && '→ Wild — expect chaotic, unusual combinations'}
        </div>

        {!snapshot && (
          <div className="text-center py-12">
            <div className="inline-block px-6 py-3 bg-bh-white border-4 border-bh-black shadow-bh">
              <span className="font-bold text-bh-fg/60">Loading model weights...</span>
            </div>
          </div>
        )}

        {snapshot && names.length === 0 && (
          <div className="text-center py-12">
            <div className="inline-block px-6 py-4 bg-bh-white border-4 border-bh-black shadow-bh">
              <p className="text-bh-fg/50">Hit <span className="text-bh-red font-bold">Generate</span> to create names!</p>
            </div>
          </div>
        )}

        {names.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {names.map((result, i) => {
              const colorIdx = i % 3
              const isSelected = selectedName === i
              return (
                <button key={`${result.name}-${i}`} onClick={() => setSelectedName(isSelected ? null : i)}
                  className={`card-lift text-left p-5 bg-bh-white border-4 border-bh-black ${isSelected ? 'shadow-bh-lg -translate-y-1' : 'shadow-bh'} transition-all duration-200`}>
                  {/* Corner decoration */}
                  <div className="absolute top-2 right-2">
                    <div className={i % 3 === 0 ? 'geo-circle' : 'geo-square'} style={{ backgroundColor: COLORS[colorIdx] }} />
                  </div>
                  <div className="font-heading font-black text-3xl md:text-4xl text-bh-black mb-2 leading-tight">
                    {result.name}
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: COLORS[colorIdx] }}>
                      {result.steps.length} tokens
                    </span>
                    <span className="text-[10px] text-bh-fg/30 font-mono">{result.nfdText.length} chars NFD</span>
                  </div>

                  {isSelected && result.steps.length > 0 && (
                    <div className="mt-3 pt-3 border-t-2 border-bh-black/10">
                      <div className="text-[9px] font-bold uppercase tracking-widest text-bh-fg/40 mb-2">Step-by-step</div>
                      <div className="space-y-1 max-h-40 overflow-y-auto">
                        {result.steps.map((step, si) => (
                          <div key={si} className="flex items-center gap-1.5 text-[10px]">
                            <span className="text-bh-fg/30 font-mono w-4">{si + 1}.</span>
                            <span className="font-bold text-bh-fg/50">{step.inputChar}</span>
                            <span className="text-bh-fg/20">→</span>
                            <span className="font-bold" style={{ color: COLORS[si % 3] }}>{step.selectedChar}</span>
                            <span className="text-bh-fg/20 ml-auto font-mono text-[8px]">
                              {step.topK.slice(0, 3).map(t => `${t.char.normalize?.('NFC') || t.char}(${(t.prob * 100).toFixed(0)}%)`).join(' ')}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </button>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}
