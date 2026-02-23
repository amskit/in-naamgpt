export default function HeroSection() {
  return (
    <section className="relative border-b-4 border-bh-black overflow-hidden">
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 min-h-[80vh]">
        {/* Left — text */}
        <div className="flex flex-col justify-center px-6 md:px-12 py-16 lg:py-24">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 mb-8 w-fit px-4 py-2 bg-bh-yellow border-2 border-bh-black shadow-bh-sm rounded-none">
            <div className="geo-circle bg-bh-red" />
            <div className="geo-square bg-bh-blue" />
            <span className="font-bold text-xs uppercase tracking-widest text-bh-black">Powered by microGPT</span>
          </div>

          <h1 className="font-heading font-black text-5xl sm:text-6xl lg:text-8xl uppercase tracking-tighter leading-[0.9] mb-6 text-bh-black">
            AI Hindi
            <br />
            <span className="text-bh-red">Name</span>
            <br />
            Generator
          </h1>

          <p className="text-base sm:text-lg text-bh-fg/70 font-medium max-w-md mb-10 leading-relaxed">
            A <strong className="text-bh-red font-bold">~5,400 parameter</strong> GPT
            trained from scratch in <strong className="text-bh-blue font-bold">pure Python</strong> —
            no PyTorch, no TensorFlow — generates brand new
            <strong className="text-bh-fg font-bold"> हिंदी </strong>
            names that sound real but never existed.
          </p>

          <a
            href="#generator"
            className="btn-press inline-flex items-center gap-3 w-fit px-8 py-4 bg-bh-red text-white font-bold text-sm uppercase tracking-wider border-4 border-bh-black shadow-bh-lg rounded-none focus:outline-none focus:ring-2 focus:ring-bh-blue focus:ring-offset-2"
          >
            Generate Names →
          </a>

          {/* Stats */}
          <div className="mt-12 grid grid-cols-2 sm:grid-cols-4 border-4 border-bh-black divide-x-2 divide-bh-black bg-bh-white shadow-bh-md rounded-none">
            {[
              { value: '~5.4K', label: 'Params', color: 'text-bh-red' },
              { value: '0', label: 'Deps', color: 'text-bh-blue' },
              { value: '1,609', label: 'Names', color: 'text-bh-red' },
              { value: '57', label: 'Vocab', color: 'text-bh-blue' },
            ].map((s, i) => (
              <div key={i} className="px-4 py-4 text-center">
                <div className={`font-heading font-black text-2xl ${s.color}`}>{s.value}</div>
                <div className="text-[10px] font-bold uppercase tracking-widest text-bh-fg/50">{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right — geometric composition */}
        <div className="relative bg-bh-blue hidden lg:flex items-center justify-center overflow-hidden">
          {/* Large circle */}
          <div className="absolute w-72 h-72 rounded-full bg-bh-yellow opacity-90" style={{ top: '15%', left: '10%' }} />
          {/* Rotated square */}
          <div className="absolute w-48 h-48 bg-bh-red opacity-80" style={{ top: '35%', right: '15%', transform: 'rotate(45deg)' }} />
          {/* Center square */}
          <div className="absolute w-36 h-36 bg-bh-white border-4 border-bh-black" style={{ top: '40%', left: '35%' }}>
            {/* Inner triangle */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div style={{ width: 0, height: 0, borderLeft: '40px solid transparent', borderRight: '40px solid transparent', borderBottom: '70px solid #121212' }} />
            </div>
          </div>
          {/* Small circle */}
          <div className="absolute w-20 h-20 rounded-full border-4 border-bh-black bg-bh-bg" style={{ bottom: '15%', left: '20%' }} />
          {/* Dot pattern overlay */}
          <div className="absolute inset-0 pattern-dots" />
          {/* Hindi text */}
          <div className="absolute bottom-8 right-8 font-heading font-black text-7xl text-white/20 leading-none">
            नाम
          </div>
        </div>
      </div>
    </section>
  )
}
