import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: no GPU detected — go to Runtime > Change runtime type > GPU")
else:
    print(torch.cuda.get_device_name(0))
