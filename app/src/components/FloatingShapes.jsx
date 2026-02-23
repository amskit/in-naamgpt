const COLORS = ['#D02020', '#1040C0', '#F0C020']

const SHAPES = [
  { type: 'circle', size: 60, top: '8%', left: '4%', color: 0 },
  { type: 'square', size: 40, top: '20%', right: '6%', color: 1, rotate: 45 },
  { type: 'circle', size: 80, top: '45%', left: '2%', color: 2 },
  { type: 'square', size: 50, top: '60%', right: '3%', color: 0 },
  { type: 'circle', size: 35, top: '75%', left: '8%', color: 1 },
  { type: 'square', size: 45, top: '85%', right: '7%', color: 2, rotate: 45 },
]

export default function FloatingShapes() {
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      {SHAPES.map((s, i) => (
        <div
          key={i}
          className="absolute"
          style={{
            top: s.top,
            left: s.left,
            right: s.right,
            width: s.size,
            height: s.size,
            backgroundColor: COLORS[s.color],
            borderRadius: s.type === 'circle' ? '9999px' : '0',
            transform: s.rotate ? `rotate(${s.rotate}deg)` : undefined,
            opacity: 0.08,
          }}
        />
      ))}
    </div>
  )
}
