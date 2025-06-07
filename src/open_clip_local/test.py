import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.open_clip_local.factory import create_model_and_transforms

if __name__ == "__main__":
    print(model.visual)  # Will give you exact config