# 安裝指南 — 使用 uv

<!-- 最後更新：2026-04-07 -->

> 回到 [安裝指南總覽](./installation.md)

## 1. 安裝 uv

如果尚未安裝 uv，請先安裝：

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> 安裝完成後重新開啟終端，確認 `uv --version` 有正確輸出。

## 2. 建立虛擬環境

```bash
uv venv --python 3.12
```

啟動方式與 venv 相同：
```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (CMD)
.venv\Scripts\activate.bat
# Windows (Git Bash) / Linux / macOS
source .venv/bin/activate
```

## 3. 安裝 PyTorch CUDA 版

> **重要**：PyTorch 的 CUDA 版必須**最先安裝**，否則後續套件可能自動裝入 CPU 版，導致推論極慢。

先到 [PyTorch 官網](https://pytorch.org/get-started/locally/) 查看你的 NVIDIA driver 支援的 CUDA 版本，選擇**不超過該版本**的 PyTorch 即可。

```bash
# 範例：CUDA 12.4
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 4. 安裝其他相依套件

```bash
uv pip install -r requirements.txt
```

## 5. Linux 額外相依

如果 PyQt6 出現錯誤：
```bash
sudo apt-get install -y libxcb-cursor-dev
```

## 6. 驗證安裝

```bash
python tools/cuda_info.py
```

> 驗證結果的說明請見 [安裝指南總覽](./installation.md#驗證結果)
