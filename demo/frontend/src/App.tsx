import { useCallback, useEffect, useRef, useState } from 'react'
import { api, rates, type ImageItem, type InferResult, type Meta, type Outcome, type Scores } from './api'
import { AoiStrip } from './components/AoiStrip'
import { Pipeline } from './components/Pipeline'
import { TradeoffChart } from './components/TradeoffChart'

const OUTCOME: Record<Outcome, string> = { correct: '正確', escape: '漏檢', overkill: '過殺' }
const CASE_HINT = { ok: '成功案例', escape: '失敗案例', overkill: '失敗案例' }

function initialTheme(): 'dark' | 'light' {
  try {
    const saved = localStorage.getItem('theme')
    if (saved === 'dark' || saved === 'light') return saved
  } catch { /* 無法存取 localStorage 時使用預設 */ }
  return 'dark'
}

const pct = (n: number, d: number) => `${((n / d) * 100).toFixed(1)}%`

export default function App() {
  const [theme, setTheme] = useState(initialTheme)
  const [meta, setMeta] = useState<Meta>()
  const [category, setCategory] = useState('')
  const [images, setImages] = useState<ImageItem[]>([])
  const [image, setImage] = useState('')
  const [result, setResult] = useState<InferResult>()
  const [scores, setScores] = useState<Scores>()
  const [threshold, setThreshold] = useState(0.5)
  const [touched, setTouched] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const request = useRef(0)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('theme', theme) } catch { /* 忽略 */ }
  }, [theme])

  const infer = useCallback(async (cat: string, img: string) => {
    const id = ++request.current
    setImage(img)
    setBusy(true)
    setError('')
    try {
      const res = await api.infer(cat, img)
      if (id === request.current) setResult(res)
    } catch (e) {
      if (id === request.current) setError(`推論失敗：${(e as Error).message}`)
    } finally {
      if (id === request.current) setBusy(false)
    }
  }, [])

  /** 切換類別（可指定影像；未指定時選第一張瑕疵），門檻歸位到 0.5 */
  const select = useCallback(async (cat: string, img?: string) => {
    setCategory(cat)
    setThreshold(0.5)
    setTouched(false)
    setResult(undefined)
    const [list, sc] = await Promise.all([api.images(cat), api.scores(cat)])
    setImages(list)
    setScores(sc)
    infer(cat, img ?? list.find((i) => i.is_defect)?.image ?? list[0].image)
  }, [infer])

  useEffect(() => {
    api.meta().then((m) => {
      setMeta(m)
      select(m.categories[0])
    }).catch(() => setError('無法連線到後端，請確認 demo 伺服器是否已啟動。'))
  }, [select])

  const current = !image ? 2 : busy || !result ? 3 : touched ? 6 : 5
  const models = meta?.models ?? []

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>MVTec AD 工業異常偵測</h1>
          <p className="muted">
            PatchCore、FastFlow、EfficientAD 三模型判定比較。三個模型都<b>只用良品影像訓練</b>，沒有看過任何瑕疵。
          </p>
        </div>
        <div className="header-side">
          <span className="badge">ONNX Runtime · CPU</span>
          <button className="theme-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label="切換深淺色">
            {theme === 'dark' ? '☀ 淺色' : '☾ 深色'}
          </button>
        </div>
      </header>

      <Pipeline current={current} busy={busy} />
      <AoiStrip />

      <section className="card controls">
        <div className="demo-buttons">
          <span className="demo-title">demo 按鈕</span>
          {meta?.demo_cases.map((c, i) => (
            <button key={c.label} className={`demo-btn ${c.kind}`} onClick={() => select(c.category, c.image)}>
              <span className="demo-index">{i + 1}</span>
              <span>{c.label}</span>
              <span className="demo-hint">{CASE_HINT[c.kind]}</span>
            </button>
          ))}
        </div>
        <div className="selects">
          <label>
            <span>類別</span>
            <select value={category} onChange={(e) => select(e.target.value)}>
              {meta?.categories.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label className="grow">
            <span>測試影像</span>
            <select value={image} onChange={(e) => infer(category, e.target.value)}>
              <optgroup label="瑕疵">
                {images.filter((i) => i.is_defect).map((i) => (
                  <option key={i.image} value={i.image}>{i.image}（{i.defect_type}）</option>
                ))}
              </optgroup>
              <optgroup label="良品">
                {images.filter((i) => !i.is_defect).map((i) => (
                  <option key={i.image} value={i.image}>{i.image}</option>
                ))}
              </optgroup>
            </select>
          </label>
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      <section className="panels">
        <Panel title="輸入影像" src={result?.input} busy={busy}
          tag={result && (result.is_defect ? `瑕疵：${result.defect_type}` : '良品')} />
        <Panel title="Ground Truth" src={result?.ground_truth} busy={busy}
          tag={result && (result.is_defect ? '紅色＝官方標註' : '無瑕疵')} />
        {models.map((m) => {
          const r = result?.results.find((x) => x.model === m.id)
          return (
            <Panel key={m.id} title={m.name} src={r?.heatmap} busy={busy} outcome={r?.outcome}
              tag={r && `${r.score.toFixed(3)} · ${r.verdict}`} />
          )
        })}
      </section>

      <section className="card">
        <div className="section-head">
          <h2>判定結果</h2>
          <p className="muted">異常分數 ≥ 門檻 0.5 判為 NG。推論時間為本機 CPU 上 ONNX 模型的執行時間。</p>
        </div>
        <table className="table">
          <thead>
            <tr><th>模型</th><th>異常分數</th><th>判定</th><th>結果</th><th>推論時間</th></tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const r = result?.results.find((x) => x.model === m.id)
              return (
                <tr key={m.id}>
                  <td>{m.name}</td>
                  <td className="num">{r ? r.score.toFixed(3) : '—'}</td>
                  <td>{r ? <span className={`verdict ${r.verdict}`}>{r.verdict}</span> : '—'}</td>
                  <td>{r ? <span className={`chip ${r.outcome}`}>{OUTCOME[r.outcome]}</span> : '—'}</td>
                  <td className="num">{r ? `${r.ms.toFixed(0)} ms` : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <details className="howto">
          <summary>怎麼看結果</summary>
          <ul>
            <li><b>異常分數</b>：0～1、沒有單位，越接近 1 越不正常。三個模型的原始分數單位不同（特徵距離、機率等），已換算到同一尺度才能比較。</li>
            <li><b>門檻 0.5</b>：每個模型在驗證資料上「誤報與漏檢最平衡」的位置，換算後剛好落在 0.5。</li>
            <li><b>熱圖</b>：只標出模型判為瑕疵的區域，越紅越確定；<b>白色輪廓</b>是官方標註的瑕疵位置。</li>
            <li>所有影像都來自<b>沒有參與門檻決定</b>的測試資料。</li>
          </ul>
        </details>
      </section>

      <section className="card">
        <div className="section-head">
          <h2>判定門檻與漏檢／過殺</h2>
          <p className="muted">
            門檻就是「分數多高才算 NG」。<b className="t-escape">漏檢</b>＝瑕疵品被判 OK（流到客戶）；
            <b className="t-overkill">過殺</b>＝良品被判 NG（需複檢或報廢）。門檻調低漏檢減少、過殺增加，反之亦然。
          </p>
        </div>
        <div className="slider-row">
          <input type="range" min={0} max={1} step={0.01} value={threshold}
            onChange={(e) => { setThreshold(Number(e.target.value)); setTouched(true) }} aria-label="判定門檻" />
          <span className="slider-value">{threshold.toFixed(2)}</span>
          <button className="link" onClick={() => setThreshold(0.5)}>重設 0.5</button>
        </div>
        {scores && (
          <>
            <table className="table">
              <thead><tr><th>模型</th><th>漏檢率</th><th>過殺率</th></tr></thead>
              <tbody>
                {models.map((m) => {
                  const r = rates(scores[m.id], threshold)
                  return (
                    <tr key={m.id}>
                      <td>{m.name}</td>
                      <td className="num t-escape">{pct(r.escape, r.nDefect)}（{r.escape}/{r.nDefect}）</td>
                      <td className="num t-overkill">{pct(r.overkill, r.nGood)}（{r.overkill}/{r.nGood}）</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="charts">
              {models.map((m) => <TradeoffChart key={m.id} name={m.name} scores={scores[m.id]} threshold={threshold} />)}
            </div>
            <p className="legend muted">
              <span className="swatch escape" />漏檢率　<span className="swatch overkill" />過殺率　<span className="swatch cursor" />目前門檻
              ｜ 統計範圍：{category} 類別 test 影像（瑕疵 {scores[models[0]?.id]?.defect.length} 張、良品 {scores[models[0]?.id]?.good.length} 張）
            </p>
          </>
        )}
      </section>

      <footer className="footer muted">
        推論時間不代表產線節拍（未含取像、傳輸與通訊）。資料集：MVTec AD（MVTec Software GmbH），CC BY-NC-SA 4.0，僅供非商業用途。
      </footer>
    </div>
  )
}

function Panel({ title, src, busy, tag, outcome }: {
  title: string
  src?: string
  busy: boolean
  tag?: string | false
  outcome?: Outcome
}) {
  return (
    <figure className={`panel ${outcome ?? ''}`}>
      <figcaption>
        <span>{title}</span>
        {tag && !busy && <span className={`panel-tag ${outcome ?? ''}`}>{tag}</span>}
      </figcaption>
      <div className="panel-body">
        {busy || !src ? <div className="skeleton" /> : <img src={src} alt={title} />}
      </div>
    </figure>
  )
}
