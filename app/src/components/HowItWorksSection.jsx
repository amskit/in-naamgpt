const COLORS = ['#D02020', '#1040C0', '#F0C020']

const STEPS = [
  { num: '01', title: 'Dataset', desc: '1,609 Sanskrit & Hindi names in Devanagari — आरव, प्रिया, कृष्ण...', color: 0, shape: 'circle' },
  { num: '02', title: 'Tokenize', desc: 'Decompose into Unicode NFD characters — consonants, vowels, matras, virama — 57-token vocabulary.', color: 1, shape: 'square' },
  { num: '03', title: 'Train', desc: '1-layer Transformer with 4 attention heads. Adam optimizer, cross-entropy loss — all in pure Python.', color: 2, shape: 'triangle' },
  { num: '04', title: 'Generate', desc: 'Start with [BOS], predict next character, sample from distribution, repeat until [END]. Runs in your browser.', color: 0, shape: 'circle' },
]

export default function HowItWorksSection() {
  return (
    <section className="relative py-12 md:py-24 px-4 md:px-8 border-b-4 border-bh-black">
      <div className="max-w-7xl mx-auto">
        <h2 className="font-heading font-black text-4xl sm:text-6xl lg:text-7xl uppercase tracking-tighter leading-[0.9] text-bh-black mb-10">
          How It <span className="text-bh-blue">Works</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {STEPS.map((step, i) => (
            <div key={step.num} className="relative card-lift bg-bh-white border-4 border-bh-black shadow-bh-lg p-6">
              {/* Corner decoration */}
              <div className="absolute top-3 right-3">
                {step.shape === 'circle' && <div className="geo-circle" style={{ backgroundColor: COLORS[step.color] }} />}
                {step.shape === 'square' && <div className="geo-square" style={{ backgroundColor: COLORS[step.color] }} />}
                {step.shape === 'triangle' && <div className="geo-triangle" style={{ color: COLORS[step.color] }} />}
              </div>

              {/* Step number — rotated */}
              <div className="w-12 h-12 flex items-center justify-center border-4 border-bh-black mb-4"
                style={{ backgroundColor: COLORS[step.color], color: step.color === 2 ? '#121212' : '#fff', transform: i % 2 === 1 ? 'rotate(45deg)' : 'none' }}>
                <span className="font-heading font-black text-lg" style={{ transform: i % 2 === 1 ? 'rotate(-45deg)' : 'none' }}>{step.num}</span>
              </div>

              <h3 className="font-heading font-black text-xl uppercase tracking-tight text-bh-black mb-2">
                {step.title}
              </h3>
              <p className="text-sm text-bh-fg/60 leading-relaxed">{step.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
