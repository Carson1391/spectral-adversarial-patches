import torch, math, numpy as np
from ultralytics import YOLO

m = YOLO('YOLOv3/yolov3u.pt').to('cuda')
model = m.model

for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        w = module.weight
        if w.ndim == 4 and w.shape[0] == 80 and w.shape[1] < 1000:
            D = w.shape[1]
            w_flat = w.mean(dim=(2,3)).detach().cpu().numpy()  # (80, D)
            break

from numpy.linalg import eigh

# Build Laplacian
norms = np.linalg.norm(w_flat, axis=1, keepdims=True)
w_norm = w_flat / (norms + 1e-8)
A = w_norm @ w_norm.T
A_thresh = np.where(A > 0.3, A, 0)
np.fill_diagonal(A_thresh, 0)
deg = A_thresh.sum(axis=1)
L = np.diag(deg) - A_thresh
eigenvalues, eigenvectors = eigh(L)

# Person and meter eigenvector values across all 80 eigenvectors
person_vals = eigenvectors[0, :]   # (80,)
meter_vals = eigenvectors[12, :]   # (80,)

# Separation per eigenvector: |person_val - meter_val|
# Large separation = this eigenvector direction distinguishes person from meter
separation = np.abs(person_vals - meter_vals)

# Sort by eigenvalue (skip first trivial one)
idx = np.arange(1, 80)

# LOW-k (smallest eigenvalues = most flexible/confusable directions)
print('LOW-k eigenvectors (most confusable — attack here):')
low_k = idx[:5]
for k in low_k:
    sep = separation[k]
    print(f'  eig {k}: lambda={eigenvalues[k]:.2f}  person={person_vals[k]:.4f}  meter={meter_vals[k]:.4f}  separation={sep:.4f}')

print()

# Which eigenvectors have the LARGEST person-meter separation?
print('Top separation eigenvectors (where person vs meter differ most):')
sep_order = np.argsort(-separation[1:])+1
for k in sep_order[:5]:
    sep = separation[k]
    print(f'  eig {k}: lambda={eigenvalues[k]:.2f}  person={person_vals[k]:.4f}  meter={meter_vals[k]:.4f}  separation={sep:.4f}')

print()

# Which eigenvectors have the SMALLEST separation (person ≈ meter, most confusable)?
print('Lowest separation eigenvectors (person ≈ meter, easiest to flip):')
sep_asc = np.argsort(separation[1:])+1
for k in sep_asc[:5]:
    sep = separation[k]
    print(f'  eig {k}: lambda={eigenvalues[k]:.2f}  person={person_vals[k]:.4f}  meter={meter_vals[k]:.4f}  separation={sep:.4f}')

print()

# Now map eigenvectors back to feature dimensions
# Project w_person - w_meter onto the low-k eigenvectors to find which dims matter
# Actually: the eigenvectors are in class-space (80). To get feature-space (256) importance:
# For each eigenvector v_k, compute the weighted feature direction: sum_i v_k[i] * w_flat[i]
# This gives a 256-dim vector showing which feature dims that eigenvector emphasizes
diff_vec = w_flat[0] - w_flat[12]  # (256,) person - meter in feature space

# Project the person-meter difference onto the low-k eigenvector feature directions
print('Feature dimensions that matter most for person→meter flip')
print('(ranked by projection onto low-k eigenvector directions):')
print()

# For each of the low-k eigenvectors, compute feature-space direction
low_k_feature_importance = np.zeros(D)
for k in low_k:
    # Feature direction for this eigenvector
    feat_dir = eigenvectors[:, k] @ w_flat  # (256,)
    # Project person-meter difference onto this direction
    low_k_feature_importance += np.abs(diff_vec * feat_dir)

# Top feature dims from low-k analysis
top_dims = np.argsort(-low_k_feature_importance)[:20]
print('Top 20 dims (strongest in confusable directions):')
for d in top_dims:
    wp = w_flat[0, d]
    wm = w_flat[12, d]
    print(f'  dim {d}: w_person={wp:+.4f} w_meter={wm:+.4f} diff={wp-wm:+.4f} importance={low_k_feature_importance[d]:.6f}')

# Bottom feature dims (don't matter for the flip)
bottom_dims = np.argsort(low_k_feature_importance)[:10]
print()
print('Bottom 10 dims (irrelevant for person→meter flip):')
for d in bottom_dims:
    wp = w_flat[0, d]
    wm = w_flat[12, d]
    print(f'  dim {d}: w_person={wp:+.4f} w_meter={wm:+.4f} importance={low_k_feature_importance[d]:.6f}')

del m
torch.cuda.empty_cache()
