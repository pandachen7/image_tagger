# 開發規範

## 技術棧

- python
- opencv
- 使用uv來管理python pkg

### preferred python package

- ruamel.yaml
- orjson
- pathlib, 取代os的路徑功能

### python風格

- 不需要加__init__.py
- 不要刪除comment, 除非用另一個comment整理, 或者該comment不符計算內容
- 一般import都直接放檔案最上層即可. 如果需要, 可加上`from __future__ import annotations`好套用type hint
- 如果是重型pkg, 可透過`TYPE_CHECKING`與lazy import或在func裡import, 來加速init
- Enum的成員要大寫
- raise前用log.error()紀錄錯誤, 不要把後端的詳細錯誤直接raise導致傳給前端, 只能貼大概的錯誤類別
- 所有的try都務必加上except, 並且配合log.error()來紀錄可能的問題

### config

- 如果有修改`config**.py`, 一併修改對應的`cfg/**.yaml`

## 注意事項

- 如果有不同方案可供選擇的話, 請先與使用者確認細節
- 使用繁體中文zh-TW來回覆
- 新增或更新程式碼檔案時, 請在最上方加入或修改Comment簡單描述這個檔案的用途, 並且更新編輯時間
- class與func都要加上簡短的comment, param與return都要有type hint
