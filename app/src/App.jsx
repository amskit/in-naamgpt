import { useEffect, useState, lazy, Suspense } from 'react'
import FloatingShapes from './components/FloatingShapes'
import HeroSection from './components/HeroSection'
import NameCloudSection from './components/NameCloudSection'
import TokenizationSection from './components/TokenizationSection'
import EmbeddingSection from './components/EmbeddingSection'
import AttentionSection from './components/AttentionSection'
import LossGradientSection from './components/LossGradientSection'
const TrainingSection = lazy(() => import('./components/TrainingSection'))
import GeneratorSection from './components/GeneratorSection'
import HowItWorksSection from './components/HowItWorksSection'
import ArchitectureSection from './components/ArchitectureSection'
import FooterSection from './components/FooterSection'

function SectionFallback() {
  return (
    <div className="py-24 text-center">
      <div className="inline-block w-8 h-8 border-4 border-bh-black border-t-bh-red animate-spin" />
    </div>
  )
}

function App() {
  const [snapshot, setSnapshot] = useState(null)
  const [loadError, setLoadError] = useState(null)

  useEffect(() => {
    let active = true
    const controller = new AbortController()

    fetch('/data/in_embedding_snapshot.json', { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch model weights')
        return res.json()
      })
      .then((data) => {
        if (!active) return
        if (
          !data?.wte ||
          !data?.wpe ||
          !data?.attention ||
          !data?.mlp ||
          !data?.lm_head ||
          !data?.tokenizer?.uchars
        ) {
          throw new Error('Invalid snapshot format')
        }
        setSnapshot(data)
      })
      .catch((err) => {
        if (!active || err.name === 'AbortError') return
        setLoadError(err.message)
        console.error('Failed to load model:', err)
      })

    return () => {
      active = false
      controller.abort()
    }
  }, [])

  return (
    <div className="relative min-h-screen bg-bh-bg text-bh-fg font-body">
      <a href="#generator" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-bh-yellow focus:text-bh-black focus:font-bold focus:border-2 focus:border-bh-black">
        Skip to Generator
      </a>
      <FloatingShapes />

      <HeroSection />
      <NameCloudSection />
      <TokenizationSection snapshot={snapshot} />
      <EmbeddingSection snapshot={snapshot} />
      <AttentionSection snapshot={snapshot} />
      <LossGradientSection snapshot={snapshot} />
      <Suspense fallback={<SectionFallback />}>
        <TrainingSection />
      </Suspense>
      <GeneratorSection snapshot={snapshot} />
      <HowItWorksSection />
      <ArchitectureSection />
      <FooterSection />

      {loadError && (
        <div className="fixed bottom-6 right-6 max-w-sm z-50 p-4 bg-bh-white border-4 border-bh-black shadow-bh-lg">
          <div className="font-heading font-bold text-bh-red text-sm uppercase tracking-widest mb-1">
            Load Error
          </div>
          <div className="text-bh-fg/70 text-sm">{loadError}</div>
        </div>
      )}
    </div>
  )
}

export default App
