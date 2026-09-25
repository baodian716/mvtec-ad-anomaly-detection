const STAGES: { name: string; covered: boolean; note: string }[] = [
  { name: '取像', covered: false, note: '使用公開資料集' },
  { name: '前處理', covered: true, note: '縮放至 256×256' },
  { name: '模型推論', covered: true, note: '三模型並排' },
  { name: '判定', covered: true, note: '門檻滑桿' },
  { name: '人工複判', covered: false, note: '僅討論' },
  { name: '追溯與回饋', covered: false, note: '未涵蓋' },
]

/** 對照 AOI 產線流程：標出本 demo 涵蓋的環節，其餘明確標示未涵蓋。 */
export function AoiStrip() {
  return (
    <section className="card aoi">
      <div className="section-head">
        <h2>對應的 AOI 產線環節</h2>
        <p className="muted">本 demo 展示「推論」與「判定」；其餘環節未在此專案中實作。</p>
      </div>
      <div className="aoi-row">
        {STAGES.map((s) => (
          <div key={s.name} className={`aoi-stage ${s.covered ? 'covered' : ''}`}>
            <span className="aoi-name">{s.name}</span>
            <span className="aoi-note">{s.covered ? `✓ ${s.note}` : s.note}</span>
          </div>
        ))}
      </div>
    </section>
  )
}
