import VisitorStats from './VisitorStats'

export default function FooterSection() {
  return (
    <footer className="bg-bh-black text-white py-12 md:py-16 px-4 md:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
          {/* Logo area */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-4 h-4 rounded-full bg-bh-red" />
              <div className="w-4 h-4 bg-bh-blue" />
              <div style={{ width: 0, height: 0, borderLeft: '8px solid transparent', borderRight: '8px solid transparent', borderBottom: '14px solid #F0C020' }} />
              <span className="font-heading font-black text-xl uppercase tracking-tight ml-2">in-naamGPT</span>
            </div>
            <p className="text-white/50 text-sm max-w-sm leading-relaxed">
              A tiny GPT trained on Hindi names in pure Python. Based on{' '}
              <a href="https://github.com/karpathy/microGPT" target="_blank" rel="noopener noreferrer" className="text-bh-yellow underline hover:text-white transition-colors duration-200">
                Karpathy's microGPT
              </a>{' '}
              and inspired by{' '}
              <a href="https://github.com/woduq1414/ko-microgpt" target="_blank" rel="noopener noreferrer" className="text-bh-yellow underline hover:text-white transition-colors duration-200">
                ko-microgpt
              </a>.
            </p>
          </div>

          {/* Tags */}
          <div className="flex flex-wrap gap-2">
            {[
              { label: 'Pure Python', color: 'bg-bh-red' },
              { label: 'Zero Dependencies', color: 'bg-bh-blue' },
              { label: 'Browser Inference', color: 'bg-bh-yellow text-bh-black' },
              { label: 'Hindi Unicode', color: 'bg-bh-red' },
            ].map((tag, i) => (
              <span key={i} className={`${tag.color} px-3 py-1 text-[10px] font-bold uppercase tracking-widest border-2 border-white/20`}>
                {tag.label}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-10 pt-6 border-t-2 border-white/10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <span className="text-white/30 text-xs">
            Built with pure math — {new Date().getFullYear()}
          </span>
          <VisitorStats />
        </div>
      </div>
    </footer>
  )
}
