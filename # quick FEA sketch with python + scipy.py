# quick FEA sketch with python + scipy
import numpy as np, matplotlib.pyplot as plt
from scipy.linalg import eigh

N = 61                      # grid points
h = 0.005                   # 5 cm plate
dx = 0.05/(N-1)
D  = 70e9*0.0005**3/12      # E*h^3/12  (Al)
rho= 2700*0.0005            # ρ*h
lap = (np.diag([-4]*(N))+np.diag([1]*(N-1),1)+np.diag([1]*(N-1),-1))/dx**2

# build 2-D biharmonic operator L⊗I + I⊗L
L = np.kron(lap, np.eye(N))+np.kron(np.eye(N), lap)
K = D*L@L                    # bending stiffness
M = rho*np.eye(N*N)          # mass matrix

w2, vec = eigh(K, M, subset_by_index=[150,150])   # mode ≈20 kHz
mode = vec[:,0].reshape(N,N)

plt.imshow(mode.real, cmap='bwr'); plt.axis('off')
plt.title("Predicted 20–22 kHz Chladni mode"); plt.show()