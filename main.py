import torch  # torch必須比pyqt還早, 以免索引出錯

from src.object_tagger import main

# 注意：訓練時 DataLoader worker 在 Windows 用 spawn 啟動會重新 import main.py，
# module-level 的 print 會被印很多次，所以放進 __main__ guard。
if __name__ == "__main__":
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("cudnn version:", torch.backends.cudnn.version())
    main()
