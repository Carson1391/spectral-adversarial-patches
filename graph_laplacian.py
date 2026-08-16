import torch, math, numpy as np
from ultralytics import YOLO

m = YOLO('YOLOv3/yolov3u.pt').to('cuda')
model = m.model

# Get class weights
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        w = module.weight
        if w.ndim == 4 and w.shape[0] == 80 and w.shape[1] < 1000:
            D = w.shape[1]
            w_flat = w.mean(dim=(2,3)).detach().cpu().numpy()  # (80, D)
            break

COCO = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',
        10:'fire hydrant',11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',
        20:'elephant',21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',
        30:'skis',31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',
        38:'tennis racket',39:'bottle',40:'wine glass',41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',
        48:'sandwich',49:'orange',50:'broccoli',51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'couch',
        58:'potted plant',59:'bed',60:'dining table',61:'toilet',62:'tv',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',
        68:'microwave',69:'oven',70:'toaster',71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',
        78:'hair drier',79:'toothbrush'}

# Build similarity graph (cosine similarity -> adjacency matrix)
from numpy.linalg import eigvalsh, eigh

# Normalize weight vectors
norms = np.linalg.norm(w_flat, axis=1, keepdims=True)
w_norm = w_flat / (norms + 1e-8)

# Adjacency matrix: cosine similarity
A = w_norm @ w_norm.T  # (80, 80)
# Threshold: only keep strong connections (sim > 0.3)
A_thresh = np.where(A > 0.3, A, 0)
np.fill_diagonal(A_thresh, 0)

# Degree matrix
deg = A_thresh.sum(axis=1)
D_mat = np.diag(deg)

# Graph Laplacian: L = D - A
L = D_mat - A_thresh

# Eigendecomposition (symmetric matrix)
eigenvalues, eigenvectors = eigh(L)

# Print results
print(f'Graph Laplacian Analysis — YOLOv3u (D={D})')
print(f'Adjacency threshold: cosine > 0.3')
print(f'Connected edges: {(A_thresh > 0).sum() // 2}')
print(f'Isolated nodes (degree=0): {(deg == 0).sum()}')
print()

# Fiedler vector (2nd smallest eigenvalue) shows the main community structure
print('Smallest eigenvalues (community structure):')
for i in range(min(10, len(eigenvalues))):
    if eigenvalues[i] < 0.01:
        print(f'  lambda[{i}] = {eigenvalues[i]:.6f} (connected component)')
    else:
        print(f'  lambda[{i}] = {eigenvalues[i]:.6f}')

# Fiedler vector
fiedler = eigenvectors[:, 1]  # 2nd smallest eigenvalue's eigenvector

# Sort classes by Fiedler value — this reveals community structure
order = np.argsort(fiedler)
print(f'\nClasses ordered by Fiedler vector (community structure):')
print(f'  {"Left cluster":<40} | {"Right cluster":<40}')
print(f'  {"-"*40} | {"-"*40}')
for i in range(40):
    left = f'{COCO[order[i]]} ({fiedler[order[i]]:.3f})'
    right_idx = order[79-i]
    right = f'{COCO[right_idx]} ({fiedler[right_idx]:.3f})'
    print(f'  {left:<40} | {right:<40}')

# Where do person and parking meter fall?
person_fiedler = fiedler[0]
meter_fiedler = fiedler[12]
print(f'\nPerson Fiedler position: {person_fiedler:.4f} (rank {np.argsort(fiedler)[0]+1}/80)')
print(f'Parking meter Fiedler position: {meter_fiedler:.4f} (rank {np.argsort(fiedler)[12]+1}/80)')

# Top neighbors of person and meter in the graph
print(f'\nPerson (0) graph neighbors (cosine > 0.3):')
person_neighbors = [(j, A[0, j]) for j in range(80) if A[0, j] > 0.3 and j != 0]
person_neighbors.sort(key=lambda x: -x[1])
for idx, sim in person_neighbors[:10]:
    print(f'  {COCO[idx]:<20} cos={sim:.4f}  fiedler={fiedler[idx]:.4f}')

print(f'\nParking meter (12) graph neighbors (cosine > 0.3):')
meter_neighbors = [(j, A[12, j]) for j in range(80) if A[12, j] > 0.3 and j != 12]
meter_neighbors.sort(key=lambda x: -x[1])
for idx, sim in meter_neighbors[:10]:
    print(f'  {COCO[idx]:<20} cos={sim:.4f}  fiedler={fiedler[idx]:.4f}')

# Shared neighbors (classes connected to BOTH person and meter)
person_n = set(j for j in range(80) if A[0, j] > 0.3 and j != 0)
meter_n = set(j for j in range(80) if A[12, j] > 0.3 and j != 12)
shared_n = person_n & meter_n
print(f'\nShared neighbors (connected to BOTH person AND meter): {len(shared_n)}')
for idx in shared_n:
    print(f'  {COCO[idx]:<20} cos_person={A[0,idx]:.4f}  cos_meter={A[12,idx]:.4f}')

# Direct person-meter connection
print(f'\nDirect person-meter cosine similarity: {A[0,12]:.4f}')
print(f'Direct person-meter L2 distance: {np.linalg.norm(w_flat[0] - w_flat[12]):.4f}')

# Spectral embedding (first 3 non-trivial eigenvectors)
print(f'\nSpectral embedding (3D) for key classes:')
emb = eigenvectors[:, 1:4]  # skip first (trivial)
for cls in [0, 12, 9, 11, 13, 56, 60, 14, 15, 16, 2, 5, 7]:
    print(f'  {COCO[cls]:<20} ({emb[cls,0]:.3f}, {emb[cls,1]:.3f}, {emb[cls,2]:.3f})')

del m
torch.cuda.empty_cache()
