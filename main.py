import torch

from src.object_tagger import main

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
if __name__ == "__main__":
    main()
