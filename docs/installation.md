# 安裝指南

<!-- 最後更新：2026-07-08 -->

## 環境需求

- Python >= 3.12
- 建議使用獨立的虛擬環境（uv 或 venv）
- 有 NVIDIA 顯卡可大幅加速推論

## uv vs venv — 我該選哪個？

**建議用 [uv](https://docs.astral.sh/uv/)。** 相依（含 CUDA 13.0 的 PyTorch）都寫在 `pyproject.toml` 裡，`uv sync` 一行就能裝好全部；venv + pip 則需自行分步安裝 PyTorch CUDA 版。

| | uv（建議） | venv（Python 內建） |
|---|---|---|
| 安裝門檻 | 需先安裝 uv 工具 | 不需額外安裝，Python 自帶 |
| 速度 | 安裝套件速度快 10–100 倍 | 一般 |
| PyTorch CUDA 版 | `pyproject.toml` 自動指向 cu130 index，`uv sync` 直接裝好 | 需自行從 cu130 index 手動安裝 |
| 指令風格 | `uv sync` / `uv run` | `pip install ...` |
| 適合對象 | 追求效率、想要一鍵安裝 | 初學者、不想裝額外工具 |

> **簡單來說**：優先用 **uv**（一行 `uv sync` 搞定含 CUDA 的所有相依）；如果不想裝 uv，才退回用 Python 內建的 **venv**。

---

- [使用 uv 安裝（建議）](./installation_uv.md)
- [使用 Python venv 安裝（fallback）](./installation_venv.md)

---

## 驗證結果

安裝完成後執行（uv 環境請在前面加上 `uv run`）：
```bash
python scripts/cuda_info.py
```

正常輸出應該像這樣：
```
torch version: 2.12.1+cu130
cuda available: True
cuda version: 13.0
cudnn version: 91300
```

確認重點：
- `torch version` 結尾要有 `+cuXXX`，**不是** `+cpu`
- `cuda available` 必須是 `True`

## GPU 沒有被使用？

如果推論時只有 1-2 FPS，代表沒有用到 GPU。以下是排查步驟：

### 確認 VRAM 有被佔用

在終端機執行：
```bash
nvidia-smi
```

![nvidia-smi](../asset/nvidia-smi.png)

- 右上角 `CUDA Version` 是你的 driver 支援的**最高** CUDA 版本
- 下方表格的 Memory Usage 在載入模型後應該有幾百 MB 以上的佔用
- 如果推論時 VRAM 完全沒增加，代表模型跑在 CPU 上

### 常見原因：torch 被降級為 CPU 版

安裝或升級某個套件時（尤其用了 `-U` flag），可能會把 `torch` 換成 CPU 版。

檢查方式(linux 或 git-bash)：
```bash
pip list | grep torch
```

如果版本號沒有 `+cuXXX` 後綴，就是 CPU 版。解決方式：

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

uv 版的話，直接重新同步即可（`pyproject.toml` 已指定 cu130 index）：
```bash
uv sync --reinstall-package torch --reinstall-package torchvision
```
