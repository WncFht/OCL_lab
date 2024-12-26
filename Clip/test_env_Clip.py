import torch
import clip
from PIL import Image

# Should print available models
print(clip.available_models())

# Should show your CUDA device if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")