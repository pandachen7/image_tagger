# 安裝指南 — 使用 Python venv（fallback）

<!-- 最後更新：2026-07-08 -->

> 回到 [安裝指南總覽](./installation.md)
>
> 建議優先使用 [uv](./installation_uv.md)（一行 `uv sync` 即可）；以下為不想安裝 uv 時的替代方案。
> pip 不會讀取 `pyproject.toml` 裡的 uv index 設定，所以 **PyTorch CUDA 版必須自行從 cu130 index 安裝**。

## 1. 建立虛擬環境

```bash
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (CMD)
.venv\Scripts\activate.bat
# Windows (Git Bash)
.venv/Scripts/activate.bat
# Linux / macOS
source .venv/bin/activate
```

## 2. 先安裝 PyTorch CUDA 版

> **重要**：PyTorch 的 CUDA 版必須**最先安裝**，否則後續套件可能自動裝入 CPU 版，導致推論極慢。

本專案預設使用 **CUDA 13.0（cu130）**：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

> 若你的 NVIDIA driver 較舊、不支援 CUDA 13.0，請到 [PyTorch 官網](https://pytorch.org/get-started/locally/) 查看可用版本，選擇**不超過 driver 支援上限**的 CUDA 版（例如 `.../whl/cu124`）。
> 如果之前已裝過 PyTorch，建議**全部移除**再重裝（特別注意 `torchvision` 容易被遺漏）。最保險的作法是開一個全新的虛擬環境。

## 3. 安裝其餘相依

其餘相依定義在 `pyproject.toml`，直接安裝本專案即可（torch / torchvision 已在上一步裝好，不會被覆蓋）：

```bash
pip install .
```

## 4. Linux 額外相依

如果 PyQt6 出現錯誤：
```bash
sudo apt-get install -y libxcb-cursor-dev
```

## 5. 啟動與驗證

```bash
python main.py               # 啟動程式
python scripts/cuda_info.py  # 驗證 CUDA
```

> 驗證結果的說明請見 [安裝指南總覽](./installation.md#驗證結果)
