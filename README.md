# MVTec AD 工業影像異常偵測

本專案使用 MVTec AD 工業影像資料集，比較 PatchCore、FastFlow 與
EfficientAD 三種異常偵測模型在 15 個物件與紋理類別上的表現。專案包含
45 組基準實驗結果、瑕疵定位、Labelme 人工標註評估、資料擴增、解析度
比較與失敗案例分析。

## 專案內容

- 資料集：MVTec AD，共 15 個物件與紋理類別
- 模型：PatchCore、FastFlow、EfficientAD
- 影像級指標：Image AUROC、Image F1、FP、FN
- 像素級指標：Pixel AUROC、Pixel F1
- 工程指標：訓練／建庫時間、forward latency、模型大小
- 延伸實驗：Labelme 標註品質、旋轉與光照擴增、256/384 解析度比較、threshold 取捨與困難案例分析

![專案流程](figures/project_flow.png)

## 資料夾結構

```text
.
├── src/          模型執行與分析程式
├── results/      已完成實驗的精簡結果表
├── figures/      實驗圖表與專案流程圖
├── data/         資料集放置說明（實際資料不納入 Git）
└── outputs/      本機執行後產生的輸出（不納入 Git）
```

本儲存庫不包含 Notebook、MVTec AD 原始圖片、模型 checkpoint 或大型原始
輸出。已完成的實驗數據可直接從 `results/` 查看，不需要重新訓練模型。

## 直接查看既有結果

- `results/all_runs_compact.csv`：三個模型 × 15 類別，共 45 組基準結果
- `results/augmentation_comparison.csv`：資料擴增比較
- `results/labelme_quality.csv`：Labelme 人工標註品質評估
- `results/resolution_experiments.csv`：256 與 384 解析度實驗

安裝輕量分析套件後，可以重新產生模型彙整表與總覽圖：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
pip install pandas matplotlib
python src/summarize_published_results.py
```

macOS／Linux：

```bash
source .venv/bin/activate
pip install pandas matplotlib
python src/summarize_published_results.py
```

產生的檔案會存放在 `outputs/published_summary/`。

## 資料下載

### MVTec AD

本專案使用的是第一代 MVTec AD，不是 MVTec AD 2。由於 MVTec 原下載連結
目前可能無法正常取得檔案，本專案實際使用以下 Hugging Face 第三方鏡像：

- 完整 15 類：[ProgrammerGnome/MVTecAD](https://huggingface.co/datasets/ProgrammerGnome/MVTecAD)，檔名為 `mvtec_anomaly_detection.tar.xz`，約 5.26 GB
- 單一 `metal_nut` 類別：[MSherbinii/mvtec-ad-metal-nut](https://huggingface.co/datasets/MSherbinii/mvtec-ad-metal-nut)

如需執行三模型 × 15 類的完整程式，請下載第一個完整版本。安裝
`huggingface_hub` 後，可使用 Hugging Face 官方 CLI 下載，不需要把資料放進
Git 儲存庫：

```bash
pip install huggingface_hub
hf download ProgrammerGnome/MVTecAD mvtec_anomaly_detection.tar.xz \
  --repo-type dataset \
  --local-dir data/download
```

若只想確認 PatchCore 單類別程式能否執行，可下載較小的 `metal_nut` 鏡像：

```bash
hf download MSherbinii/mvtec-ad-metal-nut \
  --repo-type dataset \
  --local-dir data/metal_nut_download
```

以上是第三方資料鏡像，不是 MVTec 官方維護的 Hugging Face 儲存庫。資料的
來源、內容與授權仍應以 [MVTec AD 官方說明頁面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
以及下載資料內附的 `license.txt` 為準。Hugging Face 頁面顯示的授權標籤
不一定能取代原始資料集授權。

完整壓縮檔下載後，將它解壓縮，找到直接包含 15 個類別的資料夾，再將類別
資料夾放在：

```text
data/mvtec_ad/
├── bottle/
├── cable/
├── capsule/
├── ...
└── zipper/
```

每個類別應具有以下結構：

```text
<category>/
├── train/good/
├── test/good/
├── test/<defect_type>/
└── ground_truth/<defect_type>/
```

如果不想移動資料，也可以透過 `--dataset-root` 指向「直接包含 15 個類別」
的資料夾。

### ImageNette（只有 EfficientAD 需要）

EfficientAD 需要額外的 ImageNette 訓練圖片。來源與不同尺寸版本可參考
[fastai ImageNette 官方儲存庫](https://github.com/fastai/imagenette)。本專案設定
可使用[完整尺寸版本](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz)。

解壓縮後預設放置位置為：

```text
data/imagenette2/train/
```

也可以使用 `--imagenette-dir` 指定其他位置。PatchCore 與 FastFlow不需要
ImageNette。更完整的資料夾說明請見 `data/README.md`。

## 安裝環境

建議使用 Python 3.11。請先依照電腦的 CPU、顯示卡與 CUDA 環境，從
[PyTorch 官方安裝頁面](https://pytorch.org/get-started/locally/)選擇對應指令
安裝 PyTorch，再安裝其餘套件：

```bash
pip install -r requirements.txt
```

原始正式實驗使用 Anomalib 2.4.2；`requirements.txt` 已固定此版本，並將
`pandas` 限制於 3.0 以下，以避免資料切分相容性問題。

## 執行單一模型

以 PatchCore 執行 `metal_nut`：

```bash
python src/run_anomaly_model.py --model patchcore --category metal_nut
```

使用自訂資料與輸出路徑：

```bash
python src/run_anomaly_model.py \
  --model fastflow \
  --category bottle \
  --dataset-root /path/to/mvtec_ad \
  --results-root /path/to/output
```

執行 EfficientAD：

```bash
python src/run_anomaly_model.py \
  --model efficientad \
  --category metal_nut \
  --imagenette-dir /path/to/imagenette2/train
```

模型執行結果預設寫入 `outputs/benchmark_3x15/`，包含設定、環境資訊、
評估指標、逐張圖片預測、視覺化結果與 checkpoint。

## 批次執行

批次程式可以指定一個或多個模型、類別，亦可執行完整 3 × 15 實驗：

```bash
python src/batch_benchmark.py \
  --models patchcore fastflow efficientad \
  --categories all \
  --resume
```

正式執行前，可先加入 `--dry-run` 查看預計執行的命令。完整 45 組實驗的
時間與硬體成本很高，一般使用者可先執行單一 PatchCore 類別確認環境。

## 其他分析工具

建立本專案使用的旋轉及光照訓練資料變體：

```bash
python src/build_augmentation_datasets.py
```

將 Labelme polygon 與 MVTec 官方 mask 比較：

```bash
python src/evaluate_labelme_annotations.py
```

Labelme 範例資料的放置規則請見 `data/README.md`。

## 結果使用注意事項

- 本專案的數值以儲存庫內記錄的實驗協定為準。
- 執行時間與 latency 會隨硬體、CUDA 和套件版本而改變。
- `forward latency` 僅計算模型 forward，不包含圖片讀取、資料前處理與完整後處理。
- 若更改 validation 切分、threshold 或輸入解析度，結果不能直接與既有表格視為同一協定。
