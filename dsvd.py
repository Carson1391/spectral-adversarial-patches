from PIL import Image
import numpy as np
arr = (patch.detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
Image.fromarray(arr).save("proof.png")
print("saved proof.png")
