本專案可對影像畫框  
可用於object detection的影像訓練dataset  

![](./asset/system_gui.png)

功能一覽
- 滾輪可預覽上/下一個檔案
- 預設label種類名稱[設定檔](./config/label.yaml), 按下數字鍵可以直接切換label名稱, 或者按下小寫的`L`來設定指定名稱
  以減少輸入次數
- 關於影片撥放
  - 讀取影片檔案, 並且labeling
  - 使用空白鍵play/pause
  - 有影片播放進度條
  - 可調整速度
  - 按下滑鼠鍵會停止撥放
  - 如果自動儲存開啟, 則按下play後也會自動儲存, 儲存的檔名將會是"[原檔名]_frame[N], 並且圖片跟xml都會抽出

## 安裝相關
安裝pytorch的延伸package, 都一定要先裝pytorch的cuda版本; 不然幾乎都是先自動安裝cpu版的, 所以會跑很慢

windows可使用pip來安裝pytorch CUDA版, 詳細可看  
https://pytorch.org/get-started/locally/  
選一個比你電腦的CUDA版本還低的pytorch就行  
**NOTE: 安裝前把所有pytorch的package都清乾淨, 尤其是 torchvision 很容易被遺忘, 沒清乾淨直接裝的話版本配不上一定出錯**
```bash
# 範例CUDA 12.4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
之後安裝requirements即可
```bash
pip install -r requirements.txt
```

## 雜談
PyQt最棒的地方在於, 畫面上可以直接打中文

### 關於video播放
Google AI Gemini-2.0-pro 跟我都試過了, 沒有辦法把video widget的frame傳到畫布中編輯  
因此用傳統的方式來把opencv frame轉成pixmap  
