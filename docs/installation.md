# 安裝指南

<!-- 最後更新：2026-04-07 -->

## 環境需求

- Python >= 3.12
- 建議使用獨立的虛擬環境（venv 或 uv）
- 有 NVIDIA 顯卡可大幅加速推論

## venv vs uv — 我該選哪個？

| | venv（Python 內建） | uv |
|---|---|---|
| 安裝門檻 | 不需額外安裝，Python 自帶 | 需先安裝 uv 工具 |
| 速度 | 一般 | 安裝套件速度快 10–100 倍 |
| 適合對象 | 初學者、不想裝額外工具 | 熟悉終端操作、追求效率 |
| 指令風格 | `pip install ...` | `uv pip install ...` 或 `uv sync` |

> **簡單來說**：如果你剛接觸 Python，用 **venv** 就好；如果你想要更快的安裝體驗，或者專案已經在用 uv，就選 **uv**。

---

- [使用 Python venv 安裝](./installation_venv.md)
- [使用 uv 安裝](./installation_uv.md)

---

## 驗證結果

安裝完成後執行：
```bash
python tools/cuda_info.py
```

正常輸出應該像這樣：
```
torch version: 2.10.0+cu126
cuda available: True
cuda version: 12.6
cudnn version: 91002
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
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

uv版的話, 記得前面加上`uv`  
```bash
uv pip uninstall torch torchvision torchaudio -y
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
