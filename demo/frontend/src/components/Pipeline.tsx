const STEPS = ['選擇類別', '選擇測試影像', '三模型同時推論', '比對熱圖與正確答案', '調整門檻，看漏檢與過殺']

/** 操作流程：已完成的步驟打勾，進行中的步驟亮起；推論時第 3 步顯示動畫。 */
export function Pipeline({ current, busy }: { current: number; busy: boolean }) {
  return (
    <ol className="pipeline" aria-label="操作流程">
      {STEPS.map((label, i) => {
        const step = i + 1
        const state = step < current ? 'done' : step === current ? 'active' : 'todo'
        return (
          <li key={label} className={`step ${state} ${busy && step === 3 ? 'busy' : ''}`}>
            <span className="dot">{state === 'done' ? '✓' : step}</span>
            <span className="label">{label}</span>
          </li>
        )
      })}
    </ol>
  )
}
