import torch

from src.object_tagger import main

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())

if __name__ == "__main__":
    main()
