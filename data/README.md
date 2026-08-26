# 資料放置說明

資料集不包含在本儲存庫中。MVTec 原下載連結目前可能無法正常取得檔案，
因此本專案使用 Hugging Face 上的第三方鏡像。

## MVTec AD

- 完整 15 類鏡像：https://huggingface.co/datasets/ProgrammerGnome/MVTecAD
- 完整壓縮檔名：`mvtec_anomaly_detection.tar.xz`（約 5.26 GB）
- 單一 `metal_nut` 鏡像：https://huggingface.co/datasets/MSherbinii/mvtec-ad-metal-nut
- 原始資料集說明：https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- 預設位置：`data/mvtec_ad/`

完整資料下載指令：

```bash
pip install huggingface_hub
hf download ProgrammerGnome/MVTecAD mvtec_anomaly_detection.tar.xz \
  --repo-type dataset \
  --local-dir data/download
```

下載後請解壓縮，並將直接包含 15 個類別的資料夾整理為下列結構：

```text
data/mvtec_ad/
└── bottle/
    ├── train/good/
    ├── test/good/
    ├── test/<defect_type>/
    └── ground_truth/<defect_type>/
```

Hugging Face 來源為第三方鏡像。資料內容及授權應以原始資料集內附的
`license.txt` 與 MVTec 說明為準。

## ImageNette

只有 EfficientAD 需要 ImageNette。

- 官方說明：https://github.com/fastai/imagenette
- 完整尺寸下載：https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
- 預設位置：`data/imagenette2/train/`

也可以在執行時使用 `--dataset-root` 與 `--imagenette-dir` 指定其他路徑。

## Labelme 評估範例

若要執行 `src/evaluate_labelme_annotations.py`，請準備：

```text
data/labelme_sample/
├── annotations/       # Labelme JSON
├── images/            # 原始圖片，檔名需與 JSON 對應
└── reference_masks/   # <image_id>_mask.png
```
