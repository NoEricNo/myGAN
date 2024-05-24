import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

# this file is for testing if you can use GPU.
# Long story short:
# If you are using pyCharm alone, install CUDA 12.1 this is the latest compatible version with pyTorch.
# Then you uninstall pyTorch and install the version compatible with CUDA 12.1
# pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
# If you use Anaconda based PyCharm, you need to manage the Anaconda environment.