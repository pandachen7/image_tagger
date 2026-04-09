# 安裝指南 — 使用 Python venv

<!-- 最後更新：2026-04-07 -->

> 回到 [安裝指南總覽](./installation.md)

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

## 2. 安裝 PyTorch CUDA 版

> **重要**：PyTorch 的 CUDA 版必須**最先安裝**，否則後續套件可能自動裝入 CPU 版，導致推論極慢。

先到 [PyTorch 官網](https://pytorch.org/get-started/locally/) 查看你的 NVIDIA driver 支援的 CUDA 版本，選擇**不超過該版本**的 PyTorch 即可。

```bash
# 範例：CUDA 12.4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> 如果之前已裝過 PyTorch，建議**全部移除**再重裝（特別注意 `torchvision` 容易被遺漏）。
> 最保險的作法是開一個全新的虛擬環境。

## 3. 安裝其他相依套件

```bash
pip install -r requirements.txt
```

## 4. Linux 額外相依

如果 PyQt6 出現錯誤：
```bash
sudo apt-get install -y libxcb-cursor-dev
```

## 5. 驗證安裝

```bash
python scripts/cuda_info.py
```

> 驗證結果的說明請見 [安裝指南總覽](./installation.md#驗證結果)
