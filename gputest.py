import torch

print("✅ CUDA available:", torch.cuda.is_available())
print("🖥️ Device:", torch.cuda.get_device_name(0))