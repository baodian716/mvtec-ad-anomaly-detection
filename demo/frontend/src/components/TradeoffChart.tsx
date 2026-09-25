import { rates } from '../api'

const W = 320
const H = 190
const PAD = { l: 40, r: 10, t: 14, b: 34 }
const GRID = Array.from({ length: 101 }, (_, i) => i / 100)

const x = (t: number) => PAD.l + t * (W - PAD.l - PAD.r)
const y = (v: number) => PAD.t + (1 - v) * (H - PAD.t - PAD.b)

/** 單一模型的門檻取捨曲線：紅線漏檢率、藍線過殺率、虛線為目前門檻。 */
export function TradeoffChart({ name, scores, threshold }: {
  name: string
  scores: { defect: number[]; good: number[] }
  threshold: number
}) {
  const points = GRID.map((t) => ({ t, ...rates(scores, t) }))
  const path = (key: 'escape' | 'overkill') =>
    points
      .map((p, i) => {
        const v = key === 'escape' ? p.escape / p.nDefect : p.overkill / p.nGood
        return `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(v).toFixed(1)}`
      })
      .join('')

  return (
    <figure className="chart">
      <figcaption>{name}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${name} 的漏檢率與過殺率隨門檻變化`}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line className="grid" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
            <text className="tick" x={PAD.l - 6} y={y(v) + 4} textAnchor="end">{v * 100}%</text>
          </g>
        ))}
        {[0, 0.5, 1].map((t) => (
          <text key={t} className="tick" x={x(t)} y={H - PAD.b + 16} textAnchor="middle">{t}</text>
        ))}
        <text className="axis" x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle">判定門檻</text>
        <path className="line escape" d={path('escape')} />
        <path className="line overkill" d={path('overkill')} />
        <line className="cursor" x1={x(threshold)} x2={x(threshold)} y1={PAD.t} y2={H - PAD.b} />
      </svg>
    </figure>
  )
}
