import { useState, useMemo } from 'react'

const COLORS = ['#D02020', '#1040C0', '#F0C020']
const EXAMPLE_NAMES = ['कृष्ण', 'प्रिया', 'आदित्य', 'शौर्य', 'लक्ष्मी', 'ध्रुव']

const CATEGORY_COLORS = {
  consonant: '#D02020',
  vowel: '#1040C0',
  matra: '#F0C020',
  virama: '#121212',
  anusvara: '#1040C0',
  nukta: '#121212',
  other: '#888',
}

const CATEGORY_LABELS = {
  consonant: 'व्यंजन',
  vowel: 'स्वर',
  matra: 'मात्रा',
  virama: 'हलन्त',
  anusvara: 'अनुस्वार',
  nukta: 'नुक्ता',
}

function decomposeToNFD(name) {
  const nfd = name.normalize('NFD')
  const chars = [...nfd]
  return chars.map((ch, i) => {
    const code = ch.codePointAt(0)
    let category = 'other'
    if (code >= 0x0915 && code <= 0x0939) category = 'consonant'
    else if (code >= 0x0905 && code <= 0x0914) category = 'vowel'
    else if (code >= 0x093E && code <= 0x094C) category = 'matra'
    else if (code === 0x094D) category = 'virama'
    else if (code === 0x0902) category = 'anusvara'
    else if (code === 0x093C) category = 'nukta'
    return {
      char: ch, code, category, index: i,
      hex: `U+${code.toString(16).toUpperCase().padStart(4, '0')}`,
    }
  })
}

export default function TokenizationSection({ snapshot }) {
  const [sel, setSel] = useState(0)
  const name = EXAMPLE_NAMES[sel]
  const tokens = useMemo(() => decomposeToNFD(name), [name])

  const tokenIds = useMemo(() => {
    if (!snapshot?.tokenizer?.uchars) return null
    return tokens.map((t) => {
      const idx = snapshot.tokenizer.uchars.indexOf(t.char)
      return idx >= 0 ? idx : '?'
    })
  }, [snapshot, tokens])

  return (
    <section className="relative py-12 md:py-24 px-4 md:px-8 border-b-4 border-bh-black bg-bh-yellow">
      <div className="max-w-7xl mx-auto">
        <span className="inline-block px-3 py-1 bg-bh-black text-white font-bold text-xs uppercase tracking-widest border-2 border-bh-black mb-4">
          Chapter 02
        </span>
        <h2 className="font-heading font-black text-4xl sm:text-6xl lg:text-7xl uppercase tracking-tighter leading-[0.9] text-bh-black mb-3">
          Tokenization
        </h2>
        <p className="text-base text-bh-black/70 font-medium max-w-2xl mb-8">
          Each Hindi name is decomposed into Unicode NFD characters — consonants, vowels, matras, and virama marks become individual tokens.
        </p>

        {/* Name selector */}
        <div className="flex flex-wrap gap-2 mb-8">
          {EXAMPLE_NAMES.map((n, i) => (
            <button key={i} onClick={() => setSel(i)}
              className={`btn-press px-5 py-2 font-heading font-bold text-lg border-2 border-bh-black transition-all duration-200 ${sel === i ? 'bg-bh-black text-white shadow-none' : 'bg-bh-white text-bh-black shadow-bh'}`}>
              {n}
            </button>
          ))}
        </div>

        {/* Original */}
        <div className="inline-block px-8 py-4 bg-bh-white border-4 border-bh-black shadow-bh-lg mb-6">
          <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/50 mb-1">Original (NFC)</div>
          <div className="font-heading font-black text-5xl md:text-6xl text-bh-black">{name}</div>
        </div>

        <div className="text-3xl text-bh-black font-black mb-6">↓</div>

        {/* Token grid */}
        <div className="p-4 md:p-6 bg-bh-white border-4 border-bh-black shadow-bh-lg">
          <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/50 mb-4">
            Decomposed Tokens (NFD) — {tokens.length} tokens
          </div>
          <div className="flex flex-wrap gap-3">
            {tokens.map((tok, i) => (
              <div key={i} className="relative min-w-[72px] p-3 bg-bh-bg border-2 border-bh-black shadow-bh-sm card-lift text-center">
                {/* Index */}
                <div className="absolute -top-2.5 -left-1.5 w-5 h-5 flex items-center justify-center text-[9px] font-black bg-bh-black text-white">{i}</div>
                {/* Character */}
                <div className="text-3xl font-bold mb-1" style={{ color: CATEGORY_COLORS[tok.category] }}>{tok.char}</div>
                <div className="text-[9px] font-mono text-bh-fg/40 mb-1">{tok.hex}</div>
                <div className="text-[9px] font-bold uppercase tracking-wider" style={{ color: CATEGORY_COLORS[tok.category] }}>
                  {CATEGORY_LABELS[tok.category] || tok.category}
                </div>
                {tokenIds && (
                  <div className="mt-1 text-[9px] font-mono bg-bh-muted px-2 py-0.5 text-bh-fg/60">ID: {tokenIds[i]}</div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="mt-6 flex flex-wrap gap-4">
          {Object.entries(CATEGORY_LABELS).map(([cat, label]) => (
            <div key={cat} className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-bh-black" style={{ backgroundColor: CATEGORY_COLORS[cat] }} />
              <span className="text-xs font-bold uppercase tracking-wider text-bh-black">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
