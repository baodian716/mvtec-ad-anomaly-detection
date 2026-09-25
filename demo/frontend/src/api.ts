export type ModelInfo = { id: string; name: string }
export type DemoCase = { label: string; category: string; image: string; kind: 'ok' | 'escape' | 'overkill' }
export type Meta = { categories: string[]; models: ModelInfo[]; demo_cases: DemoCase[]; threshold: number }
export type ImageItem = { image: string; is_defect: boolean; defect_type: string }
export type Scores = Record<string, { defect: number[]; good: number[] }>
export type Outcome = 'correct' | 'escape' | 'overkill'
export type ModelResult = {
  model: string
  name: string
  score: number
  verdict: 'OK' | 'NG'
  outcome: Outcome
  ms: number
  heatmap: string
}
export type InferResult = {
  is_defect: boolean
  defect_type: string
  input: string
  ground_truth: string
  results: ModelResult[]
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status} ${url}`)
  return res.json()
}

export const api = {
  meta: () => get<Meta>('/api/meta'),
  images: (category: string) => get<ImageItem[]>(`/api/images/${category}`),
  scores: (category: string) => get<Scores>(`/api/scores/${category}`),
  infer: async (category: string, image: string): Promise<InferResult> => {
    const res = await fetch('/api/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category, image }),
    })
    if (!res.ok) throw new Error(`${res.status} /api/infer`)
    return res.json()
  },
}

/** 門檻 t 下的漏檢率（瑕疵分數 < t）與過殺率（良品分數 ≥ t） */
export function rates(s: { defect: number[]; good: number[] }, t: number) {
  const escape = s.defect.filter((v) => v < t).length
  const overkill = s.good.filter((v) => v >= t).length
  return { escape, overkill, nDefect: s.defect.length, nGood: s.good.length }
}
