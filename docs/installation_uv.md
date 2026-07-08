# 安裝指南 — 使用 uv（建議）

<!-- 最後更新：2026-07-08 -->

> 回到 [安裝指南總覽](./installation.md)

所有相依（含 **CUDA 13.0** 的 PyTorch）都寫在 `pyproject.toml`，`uv sync` 會自動建立虛擬環境並裝好全部套件，不需手動分步安裝 PyTorch。

## 1. 安裝 uv

如果尚未安裝 uv，請先安裝：

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> 安裝完成後重新開啟終端，確認 `uv --version` 有正確輸出。

## 2. 安裝所有相依

在專案根目錄執行：

```bash
uv sync
```

`uv sync` 會：

- 依 `.python-version`（3.12）建立 `.venv`
- 從 PyTorch 官方 cu130 index 安裝含 CUDA 13.0 的 `torch` / `torchvision`
- 從 PyPI 安裝其餘相依（ultralytics、PyQt6、opencv-python…）

> PyTorch cu130 相依較大，首次安裝需要一些時間。
>
> cu130 是給 NVIDIA driver 支援 CUDA 13.0（含以上）的環境。若你的 driver 較舊，請到 [PyTorch 官網](https://pytorch.org/get-started/locally/) 查看可用的 CUDA 版本，並將 `pyproject.toml` 中 `[[tool.uv.index]]` 的 `url` 改成對應版本（例如 `.../whl/cu124`）後再 `uv sync`。

## 3. 啟動程式

```bash
uv run main.py
```

> `uv run` 會自動使用專案的 `.venv`，不需手動 activate。
> 如果偏好手動啟動，仍可 `.venv\Scripts\Activate.ps1`（Windows）/ `source .venv/bin/activate`（Linux）後直接 `python main.py`。

## 4. Linux 額外相依

如果 PyQt6 出現錯誤：
```bash
sudo apt-get install -y libxcb-cursor-dev
```

## 5. 驗證安裝

```bash
uv run scripts/cuda_info.py
```

> 驗證結果的說明請見 [安裝指南總覽](./installation.md#驗證結果)
