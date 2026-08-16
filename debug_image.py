import numpy as np
from PIL import Image
import sys

sys.path.insert(0, r'C:\Users\carso\Desktop\YODO\adversarial-robustness-toolbox\tests\estimators\object_detection')
from run_external_patch_benchmark import load_rgb, save_image

img = load_rgb(r'C:\Users\carso\Downloads\ChatGPT Image Jul 1, 2026, 04_33_21 AM (1).png', 416)
print(f"Shape: {img.shape}, dtype: {img.dtype}")
print(f"Min: {img.min()}, Max: {img.max()}, Mean per channel: {img.mean(axis=(1,2))}")

# Save using our function
save_image(img, r'C:\Users\carso\Desktop\YODO\debug_clean.png', boxes=None)

# Also save manually the correct way
correct = np.clip(img.transpose(1, 2, 0), 0, 255).astype(np.uint8)
Image.fromarray(correct).save(r'C:\Users\carso\Desktop\YODO\debug_manual.png')

# Load both back and compare
our_img = np.asarray(Image.open(r'C:\Users\carso\Desktop\YODO\debug_clean.png'))
manual_img = np.asarray(Image.open(r'C:\Users\carso\Desktop\YODO\debug_manual.png'))
print(f"Our save - shape: {our_img.shape}, mean per channel: {our_img.mean(axis=(0,1))}")
print(f"Manual save - shape: {manual_img.shape}, mean per channel: {manual_img.mean(axis=(0,1))}")
print(f"Images match: {np.array_equal(our_img, manual_img)}")
