本專案可對影像畫框  
可用於object detection的影像訓練dataset  

![](./asset/system_gui.png)

Google AI Gemini-2.0-pro 跟我都試過了, 沒有辦法把video wiget的frame傳到畫布中編輯  
因此用傳統的方式來把frame轉成pixmap  
PyQt6真的是做了很多很多餘的事, 像是按下某元件, 所有按鍵都會只對該元件作用  
原本設定好的快捷鍵全失效  