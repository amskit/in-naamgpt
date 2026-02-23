const COLORS = ['#D02020', '#1040C0', '#F0C020']

const LAYERS = [
  { label: 'Input Token', detail: 'Character ID (0-56)', color: 0 },
  { label: 'Token Embed + Pos Embed', detail: 'wte[id] + wpe[pos] → 16-dim', color: 1 },
  { label: 'RMSNorm', detail: 'Normalize values', color: 2 },
  { label: 'Multi-Head Attention', detail: '4 heads × 4 dims, Q·K/√d → softmax → V', color: 0 },
  { label: '+ Residual', detail: 'Skip connection', color: 1 },
  { label: 'RMSNorm → MLP', detail: '16→64 → ReLU → 64→16', color: 2 },
  { label: '+ Residual', detail: 'Skip connection', color: 0 },
  { label: 'Output Head', detail: 'Linear → 57 logits → softmax', color: 1 },
  { label: 'Sample Next Token', detail: 'Pick from distribution, repeat', color: 2 },
]

export default function ArchitectureSection() {
  return (
    <section className="relative py-12 md:py-24 px-4 md:px-8 border-b-4 border-bh-black bg-bh-red">
      <div className="max-w-4xl mx-auto">
        <h2 className="font-heading font-black text-4xl sm:text-6xl lg:text-7xl uppercase tracking-tighter leading-[0.9] text-white mb-3">
          The Architecture
        </h2>
        <p className="text-base text-white/70 font-medium max-w-lg mb-10">
          Every name passes through this pipeline — one token at a time.
        </p>

        <div className="space-y-3">
          {LAYERS.map((layer, i) => (
            <div key={i} className="flex items-stretch gap-3">
              <div className="flex flex-col items-center w-7 shrink-0">
                <div className="w-5 h-5 border-2 border-bh-black"
                  style={{
                    backgroundColor: COLORS[layer.color],
                    borderRadius: i === 0 || i === LAYERS.length - 1 ? '9999px' : '0',
                  }} />
                {i < LAYERS.length - 1 && <div className="w-0.5 flex-1 min-h-[12px] bg-white/30" />}
              </div>
              <div className="flex-1 px-5 py-3 bg-bh-white border-2 border-bh-black shadow-bh-sm card-lift">
                <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3">
                  <span className="font-heading font-black text-sm uppercase tracking-tight" style={{ color: COLORS[layer.color] }}>
                    {layer.label}
                  </span>
                  <span className="text-xs text-bh-fg/40 font-mono">{layer.detail}</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Model config */}
        <div className="mt-12 p-6 bg-bh-white border-4 border-bh-black shadow-bh-lg">
          <h3 className="font-heading font-black text-xl uppercase tracking-tight text-bh-black mb-4">Model Config</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { k: 'Layers', v: '1' }, { k: 'Heads', v: '4' }, { k: 'Embed Dim', v: '16' }, { k: 'Block Size', v: '32' },
              { k: 'Vocab Size', v: '57' }, { k: 'Parameters', v: '~5,408' }, { k: 'Activation', v: 'ReLU' }, { k: 'Optimizer', v: 'Adam' },
            ].map((item, i) => (
              <div key={i} className="text-center px-3 py-3 border-2 border-bh-black bg-bh-bg">
                <div className="font-heading font-black text-lg" style={{ color: COLORS[i % 3] }}>{item.v}</div>
                <div className="text-[9px] text-bh-fg/40 font-bold uppercase tracking-widest">{item.k}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
