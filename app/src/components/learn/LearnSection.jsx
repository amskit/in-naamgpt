import { useState } from 'react'
import chapters from './chapters'

const COLORS = ['#D02020', '#1040C0', '#F0C020']

function ChapterCard({ chapter, isOpen, onToggle }) {
  return (
    <div className="border-4 border-bh-black bg-white" id={`learn-${chapter.id}`}>
      {/* Header — always visible */}
      <button
        onClick={onToggle}
        className="w-full text-left px-6 py-5 flex items-start gap-4 hover:bg-bh-bg/50 transition-colors cursor-pointer"
      >
        <span
          className="shrink-0 w-12 h-12 flex items-center justify-center text-white font-heading font-black text-lg"
          style={{ backgroundColor: chapter.color }}
        >
          {chapter.num}
        </span>
        <div className="flex-1 min-w-0">
          <h3 className="font-heading font-black text-xl md:text-2xl uppercase tracking-tight leading-tight">
            {chapter.title}
          </h3>
          <p className="text-bh-fg/60 text-sm mt-1">{chapter.subtitle}</p>
        </div>
        <span className="shrink-0 text-2xl font-bold mt-1 transition-transform" style={{ transform: isOpen ? 'rotate(45deg)' : 'rotate(0deg)' }}>
          +
        </span>
      </button>

      {/* Expandable content */}
      {isOpen && (
        <div className="border-t-4 border-bh-black">
          {/* Explanation paragraphs */}
          <div className="px-6 py-6 space-y-4">
            {chapter.paragraphs.map((p, i) => (
              <p key={i} className="text-bh-fg/80 leading-relaxed text-[15px]">
                {p}
              </p>
            ))}
          </div>

          {/* Diagram */}
          {chapter.diagram && (
            <div className="mx-6 mb-6 border-4 border-bh-black bg-bh-bg p-4 overflow-x-auto">
              <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/40 mb-2">Diagram</div>
              <pre className="text-xs md:text-sm font-mono text-bh-fg whitespace-pre leading-relaxed">
                {chapter.diagram}
              </pre>
            </div>
          )}

          {/* Code snippet */}
          {chapter.code && (
            <div className="mx-6 mb-6 border-4 border-bh-black overflow-hidden">
              <div className="bg-bh-black text-white px-4 py-2 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-bh-red" />
                <div className="w-2 h-2 rounded-full bg-bh-yellow" />
                <div className="w-2 h-2 rounded-full bg-green-400" />
                <span className="ml-2 text-[10px] font-bold uppercase tracking-widest text-white/50">
                  {chapter.code.file}
                </span>
              </div>
              <div className="bg-[#1a1a1a] px-4 py-2">
                <div className="text-white/60 text-xs mb-1">{chapter.code.label}</div>
              </div>
              <div className="bg-[#1a1a1a] px-4 pb-4 overflow-x-auto">
                <pre className="text-xs md:text-sm font-mono text-green-300 whitespace-pre leading-relaxed">
                  {chapter.code.snippet}
                </pre>
              </div>
            </div>
          )}

          {/* Key Takeaways */}
          {chapter.takeaways && (
            <div className="mx-6 mb-6 border-4 border-bh-black p-4" style={{ borderLeftColor: chapter.color, borderLeftWidth: '8px' }}>
              <div className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: chapter.color }}>
                Key Takeaways
              </div>
              <ul className="space-y-2">
                {chapter.takeaways.map((t, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-bh-fg/80">
                    <span className="shrink-0 w-5 h-5 flex items-center justify-center text-white text-[10px] font-bold mt-0.5" style={{ backgroundColor: chapter.color }}>
                      {i + 1}
                    </span>
                    {t}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Further Reading */}
          {chapter.further && (
            <div className="mx-6 mb-6">
              <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/40 mb-3">
                Further Reading
              </div>
              <div className="flex flex-wrap gap-2">
                {chapter.further.map((link, i) => (
                  <a
                    key={i}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 px-3 py-1.5 border-2 border-bh-black text-xs font-bold hover:bg-bh-black hover:text-white transition-colors"
                  >
                    <span>↗</span> {link.title}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function LearnSection() {
  const [openChapters, setOpenChapters] = useState(new Set())

  const toggle = (id) => {
    setOpenChapters((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const expandAll = () => setOpenChapters(new Set(chapters.map((c) => c.id)))
  const collapseAll = () => setOpenChapters(new Set())

  return (
    <section className="py-16 md:py-24 px-4 md:px-8 bg-bh-bg" id="learn">
      <div className="max-w-4xl mx-auto">
        {/* Section header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 bg-bh-red" />
            <div className="w-8 h-8 bg-bh-blue" />
            <div className="w-8 h-8 bg-bh-yellow" />
          </div>
          <h2 className="font-heading font-black text-4xl md:text-6xl uppercase tracking-tight leading-none mb-4">
            Learn
          </h2>
          <p className="text-bh-fg/60 text-lg max-w-2xl leading-relaxed">
            Every concept used in this project — explained from first principles with code references,
            diagrams, and further reading. No prerequisites beyond basic programming.
          </p>

          {/* Table of contents */}
          <div className="mt-8 border-4 border-bh-black bg-white p-6">
            <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/40 mb-4">
              {chapters.length} Chapters
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {chapters.map((ch) => (
                <a
                  key={ch.id}
                  href={`#learn-${ch.id}`}
                  onClick={(e) => {
                    e.preventDefault()
                    setOpenChapters((prev) => new Set([...prev, ch.id]))
                    document.getElementById(`learn-${ch.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                  }}
                  className="flex items-center gap-2 px-3 py-2 hover:bg-bh-bg transition-colors group"
                >
                  <span
                    className="shrink-0 w-7 h-7 flex items-center justify-center text-white text-[10px] font-bold"
                    style={{ backgroundColor: ch.color }}
                  >
                    {ch.num}
                  </span>
                  <span className="text-sm font-bold group-hover:underline">{ch.title}</span>
                </a>
              ))}
            </div>

            <div className="flex gap-2 mt-4 pt-4 border-t-2 border-bh-black/10">
              <button
                onClick={expandAll}
                className="px-3 py-1 text-[10px] font-bold uppercase tracking-widest border-2 border-bh-black hover:bg-bh-black hover:text-white transition-colors cursor-pointer"
              >
                Expand All
              </button>
              <button
                onClick={collapseAll}
                className="px-3 py-1 text-[10px] font-bold uppercase tracking-widest border-2 border-bh-black hover:bg-bh-black hover:text-white transition-colors cursor-pointer"
              >
                Collapse All
              </button>
            </div>
          </div>
        </div>

        {/* Chapters */}
        <div className="space-y-4">
          {chapters.map((ch) => (
            <ChapterCard
              key={ch.id}
              chapter={ch}
              isOpen={openChapters.has(ch.id)}
              onToggle={() => toggle(ch.id)}
            />
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="mt-12 border-4 border-bh-black bg-bh-black text-white p-8 text-center">
          <h3 className="font-heading font-black text-2xl md:text-3xl uppercase tracking-tight mb-3">
            That's the entire GPT.
          </h3>
          <p className="text-white/60 max-w-lg mx-auto leading-relaxed">
            11 concepts. ~5,400 parameters. ~400 lines of Python.
            Everything above is implemented in this project — go read the source, tweak the numbers, train your own.
          </p>
          <div className="flex flex-wrap justify-center gap-3 mt-6">
            <a
              href="https://github.com/RaikaSurendra/in-naamgpt"
              target="_blank"
              rel="noopener noreferrer"
              className="px-6 py-2 bg-bh-red text-white font-bold text-sm uppercase tracking-widest border-2 border-white/20 hover:bg-red-700 transition-colors"
            >
              View Source →
            </a>
            <a
              href="#generator"
              className="px-6 py-2 bg-bh-yellow text-bh-black font-bold text-sm uppercase tracking-widest border-2 border-bh-black hover:bg-yellow-400 transition-colors"
            >
              Try the Generator ↑
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
