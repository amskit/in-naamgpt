import { useState, useEffect, useMemo } from 'react'

const COLORS = ['#D02020', '#1040C0', '#F0C020']

const SAMPLE_NAMES = [
  'आरव', 'प्रिया', 'कृष्ण', 'अनन्या', 'विहान', 'ईशा', 'अर्जुन', 'काव्या',
  'रुद्र', 'मीरा', 'आदित्य', 'सिया', 'शौर्य', 'तारा', 'ध्रुव', 'दिव्या',
  'कबीर', 'नव्या', 'देव', 'राधा', 'अर्णव', 'लक्ष्मी', 'विक्रम', 'गौरी',
  'ओम', 'सान्वी', 'राज', 'पार्वती', 'वीर', 'अदिति', 'साई', 'रिया',
  'गौतम', 'महिका', 'रोहन', 'भूमि', 'हर्ष', 'कियारा', 'यश', 'सुहाना',
]

function getPos(index, total, isMobile) {
  const cols = isMobile ? 4 : 8
  const col = index % cols
  const row = Math.floor(index / cols)
  const maxRows = Math.ceil(total / cols)
  const xBase = cols > 1 ? (col / (cols - 1)) * 84 : 42
  const yBase = maxRows > 1 ? (row / (maxRows - 1)) * 72 : 36
  const jx = (((index * 13) % 7) - 3) * (isMobile ? 1.2 : 1.6)
  const jy = (((index * 17) % 7) - 3) * (isMobile ? 1.0 : 1.3)
  return {
    x: Math.max(3, Math.min(97, 8 + xBase + jx)),
    y: Math.max(5, Math.min(92, 12 + yBase + jy)),
  }
}

export default function NameCloudSection() {
  const [isMobile, setIsMobile] = useState(false)
  const [hovered, setHovered] = useState(null)

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 767px)')
    setIsMobile(mq.matches)
    const h = (e) => setIsMobile(e.matches)
    mq.addEventListener('change', h)
    return () => mq.removeEventListener('change', h)
  }, [])

  const positions = useMemo(() => SAMPLE_NAMES.map((_, i) => getPos(i, SAMPLE_NAMES.length, isMobile)), [isMobile])

  return (
    <section className="relative py-12 md:py-24 px-4 md:px-8 border-b-4 border-bh-black">
      <div className="max-w-7xl mx-auto">
        {/* Section heading */}
        <div className="mb-8">
          <span className="inline-block px-3 py-1 bg-bh-red text-white font-bold text-xs uppercase tracking-widest border-2 border-bh-black shadow-bh-sm mb-4">
            Chapter 01
          </span>
          <h2 className="font-heading font-black text-4xl sm:text-6xl lg:text-7xl uppercase tracking-tighter leading-[0.9] text-bh-black">
            The Dataset
          </h2>
          <p className="mt-3 text-base text-bh-fg/60 font-medium max-w-lg">
            1,609 Sanskrit & Hindi names in Devanagari — the model's entire world. Hover to highlight.
          </p>
        </div>

        {/* Cloud container */}
        <div
          className="relative border-4 border-bh-black bg-bh-white shadow-bh-lg rounded-none overflow-hidden"
          style={{ height: isMobile ? '380px' : '480px' }}
        >
          <div className="absolute inset-0 pattern-dots" />

          {SAMPLE_NAMES.map((name, i) => {
            const pos = positions[i]
            const color = COLORS[i % 3]
            const isH = hovered === i
            return (
              <button
                key={i}
                className="absolute font-heading font-bold transition-all duration-200 ease-out"
                style={{
                  left: `${pos.x}%`,
                  top: `${pos.y}%`,
                  transform: `translate(-50%, -50%) ${isH ? 'scale(1.6)' : 'scale(1)'}`,
                  color: isH ? '#FFFFFF' : color,
                  backgroundColor: isH ? color : 'transparent',
                  padding: isH ? '4px 10px' : '0',
                  border: isH ? '2px solid #121212' : '2px solid transparent',
                  fontSize: isMobile ? '0.8rem' : '1rem',
                  zIndex: isH ? 30 : 10,
                }}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
              >
                {name}
              </button>
            )
          })}

          {/* Corner decoration */}
          <div className="absolute top-3 right-3 flex items-center gap-1.5">
            <div className="geo-circle bg-bh-red" />
            <div className="geo-square bg-bh-blue" />
            <div className="geo-circle bg-bh-yellow" />
          </div>

          <div className="absolute bottom-3 left-3 px-3 py-1.5 bg-bh-black text-white text-[10px] font-bold uppercase tracking-widest">
            1,609 names · showing 40
          </div>
        </div>
      </div>
    </section>
  )
}
