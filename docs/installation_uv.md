# 安裝指南 — 使用 uv（建議）

<!-- 最後更新：2026-08-05 -->

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

### 請保持 uv 為最新版

**本專案需要 uv >= 0.12**，且建議養成定期更新的習慣：

```bash
uv self update
```

> 若 uv 是用其他方式安裝的（如 pipx、Homebrew、scoop），`uv self update` 會失效，請改用原本的管道更新。

uv 改版很快（每一兩週就有新版），而 PyTorch 這類套件的 index 又常有邊緣狀況，**舊版 uv 的錯誤訊息經常無法反映真正的原因**。實際遇過的例子：

- PyTorch cu130 index 沒有發布 `torchvision` Windows wheel 的 `#sha256`（Linux wheel 則正常）。uv <= 0.7.x 遇到這種缺 hash 的 wheel，會退而拿下載檔去比對該版本「其他平台」的 hash，於是必然 mismatch，報出 `Hash mismatch for torchvision==...`，看起來像檔案損毀或 CDN 有問題，實際上檔案完全正常。升到 uv 0.12.1 後即正常安裝。

所以 **`uv sync` 出現無法理解的相依 / hash 錯誤時，第一步先更新 uv 再試一次**，通常比去改 `pyproject.toml` 有效。

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
