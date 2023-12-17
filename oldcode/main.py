# FINITE VOLUMNE METHODS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse.linalg as sp
from scipy.sparse import csr_matrix
from scipy.signal import convolve2d
import os
import pypardiso

### PARAMETERS ###

n = 100           # Grid points
a, b = 0, 1       # Box edges

h = 1./(n+2)      # Space interval
h2 = h*h 

k = h             # Time interval
save_interval = 0.1 # Frequency to save images
max_t = 10        # Time value before termination

e0 = 1.0           # Erosion Rate
d0 = 1.0           # Deposition Rate

c0 = 0.0          # Minimum sigma value
c1 = 1.0          # Maximum sigma value
omega = 6.5       # Erosive sharpness
phi_star = 0.70   # Erosion threshold

phi_min  = 0.75   # Minimum initial phi_s value
phi_max  = 0.80   # Maximum initial phi_s value
zeta = 0.01       # Width of correlation of initial heat field

n_blur_width = 25 # Cell width to have convolution template on
xi = 10           # Cell width of convolution std

max_decay = 0.9   # Threshold value on erosion (stop-gap measure)

source = np.zeros((n,n))

# Boundary terms
boundary = np.zeros((n+2,n+2))
boundary[0,1:n+1]   = 0 
boundary[n+1,1:n+1] = 0
boundary[1:n+1,0]   = 0
boundary[1:n+1,n+1] = 0

#boundary[48:52,n+1] = -3
boundary[0,:] = -3/25
boundary[n+1,:] = 3/25
#boundary[0,48:52] = -3
#boundary[n+1,48:52] = 3
#boundary[48:52,0] = 3

# Indexing dictionaries
ij2idx = {}
idx2ij = {}
for i in range(n+2):
    for j in range(n+2):
        ij2idx[(i,j)] = i * (n+2) + j
        idx2ij[i * (n+2) + j] = (i,j)

xs = np.linspace(a, b, n+2)
ys = np.linspace(a, b, n+2)

xs_mesh, ys_mesh = np.meshgrid(xs, ys)



### SIMULATION METHODS ###

def s():
    global t
    s = (t / max_t) * boundary.copy()
    s[1:n+1,1:n+1] = source
    s_lagrange = np.zeros((n+2)*(n+2)+1)
    s_lagrange[:-1] = s.ravel()
    return s_lagrange

def kappa(phi):
    return ((1-phi) ** 3) / (phi ** 2)

def dp2(p):
    dpdx = (p[:,2:] - p[:,:-2]) / (2*h)
    dpdy = (p[2:,:] - p[:-2,:]) / (2*h)
    dp = np.square(boundary)
    dp[:,1:-1] += np.square(dpdx)
    dp[1:-1,:] += np.square(dpdy)
    return dp

def gradp(p):
    return np.sqrt(p)

def sigma(phi):
    global omega
    other_phi = phi 

    xs = np.arange(2*n_blur_width+1)
    ys = np.arange(2*n_blur_width+1)
    xs, ys = np.meshgrid(xs, ys)
        
    dists = (n_blur_width - xs)**2 + (n_blur_width - ys)**2

    kernel = np.exp(-dists/(2*xi)) # Make local Gaussian kernel
    kernel /= np.sum(kernel)

    other_phi = convolve2d(other_phi, kernel, boundary="symm", mode="same")

    return c0 + (c1-c0) * (0.5 * np.tanh(omega * (other_phi - phi_star)) + 0.5)



# Return q = phi * u
def flux(phi, p):
    return -kappa(phi) * gradp(p)

def e(phi_s, phi_g, p):
    return e0 * phi_s * np.maximum(0, dp2(p) - sigma(phi_s))

def d(phi_s, phi_g, p):
    return d0 * phi_g * np.maximum(0, phi_s - phi_star)

def dphi_s_dt(phi_s, phi_g, p):
    #ret = d(phi_s, phi_g, p) - e(phi_s, phi_g, p)
    ret = - e(phi_s, phi_g, p)
    return np.nan_to_num(ret, nan=0.0)

def dphi_g_dt(phi_s, phi_g, p):
    #ret = e(phi_s, phi_g, p) - d(phi_s, phi_g, p) - apply_A(phi_g, p)
    ret = e(phi_s, phi_g, p)
    return np.nan_to_num(ret, nan=0.0)


def A(phi, do_lagrange=True):
    arr_data, arr_row, arr_col = [], [], []

    for (i,j) in ij2idx.keys():
        if i == 0:
            if j == 0:
                # Corner
                data = [-2, 1, 1]
                row = [0, 0, 1]
                col = [0, 1, 0]
            if 1 <= j and j <= n:
                # Edge
                flow = (kappa(phi[0,j]) + kappa(phi[1,j])) / 2
                data = [flow/h, -flow/h]
                row = [0, 1]
                col = [j, j]
            if j == n+1:
                # Corner
                data = [-2, 1, 1]
                row = [0, 0, 1]
                col = [n+1, n, n+1]
        if 1 <= i and i <= n:
            if j == 0:
                # Edge
                flow = (kappa(phi[i,0]) + kappa(phi[i,1])) / 2
                data = [flow/h, -flow/h]
                row = [i, i]
                col = [0, 1]
            if 1 <= j and j <= n:

                # Core
                flow_down  = (kappa(phi[i,j]) + kappa(phi[i-1,j])) / 2
                flow_up    = (kappa(phi[i,j]) + kappa(phi[i+1,j])) / 2
                flow_left  = (kappa(phi[i,j]) + kappa(phi[i,j-1])) / 2
                flow_right = (kappa(phi[i,j]) + kappa(phi[i,j+1])) / 2

                data = [-(flow_down + flow_up + flow_right + flow_left) / h2,\
                        flow_left / h2, \
                        flow_right / h2, \
                        flow_down / h2, \
                        flow_up / h2]

                row = [i, i, i, i-1, i+1]
                col = [j, j-1, j+1, j, j]

            if j == n+1:
                # Edge
                flow = (kappa(phi[i, n+1]) + kappa(phi[i, n])) / 2
                data = [flow/h, -flow/h]
                row = [i, i]
                col = [n+1, n]
        if i == n+1:
            if j == 0:
                # Corner
                data = [-2, 1, 1]
                row = [n+1, n+1, n]
                col = [0, 1, 0]
            if 1 <= j and j <= n:
                # Edge
                flow = (kappa(phi[n+1,j]) + kappa(phi[n,j])) / 2
                data = [flow/h, -flow/h]
                row = [n+1, n]
                col = [j, j]
            if j == n+1:
                # Corner
                data = [-2, 1, 1]
                row = [n+1, n+1, n]
                col = [n+1, n, n+1]

        # Add to array
        arr_data.extend(data)
        for ii, jj in zip(row, col):
            arr_row.append(ij2idx[i, j])
            arr_col.append(ij2idx[ii, jj])

        # Lagrange
        if do_lagrange:
            arr_data.append(1)
            arr_row.append(ij2idx[i,j])
            arr_col.append((n+2)*(n+2))

            arr_data.append(1)
            arr_row.append((n+2)*(n+2))
            arr_col.append(ij2idx[i,j])

    # Lagrange
    n2 = (n+2)*(n+2)
    if do_lagrange:
        n2 += 1
    row, col, data = np.array(arr_row), np.array(arr_col), np.array(arr_data)
    arr = csr_matrix((data, (row, col)), shape=(n2,n2))

    return arr

def apply_A(phi, x):
    return (A(phi, do_lagrange=False) * x.ravel()).reshape((n+2),(n+2))

def p(phi):
    return pypardiso.spsolve(A(phi), s())[:-1].reshape((n+2,n+2))
    #return sp.spsolve(A(phi), s())[:-1].reshape((n+2,n+2))



### ANIMATION ###

fig = plt.figure()

axs = np.empty((2,3), dtype=object)
axs[0,0] = fig.add_subplot(2, 3, 1)
axs[0,1] = fig.add_subplot(2, 3, 2)
axs[0,2] = fig.add_subplot(2, 3, 3, projection='3d')
axs[1,0] = fig.add_subplot(2, 3, 4)
axs[1,1] = fig.add_subplot(2, 3, 5)
axs[1,2] = fig.add_subplot(2, 3, 6)


itr_index = 0
files = os.listdir("img")
while "itr%d" % itr_index in files:
    itr_index += 1
itr_dir = "img/itr%d" % itr_index
os.mkdir(itr_dir)

def init():
    global phi_s, phi_g, phi_l, t

    t = 0

    # Set up the unmoving component
    phi_s = np.zeros((n+2,n+2))
    for itr in range(int(n*n)):
        xs = h * np.arange(n+2)
        ys = h * np.arange(n+2)
        xs, ys = np.meshgrid(xs, ys)

        mu = np.random.uniform(-h, 1+h, 2)
        dists = (mu[0] - xs)**2 + (mu[1] - ys)**2

        phi_s += np.sqrt(2*np.pi/zeta) * np.exp(-dists/(2 * zeta**2))
    phi_top = np.min(phi_s)
    phi_bot = np.max(phi_s)
    phi_s = phi_min + (phi_max - phi_min) * ((phi_s - phi_bot) / (phi_top - phi_bot))

    # Originally, nothing is moving
    phi_g = np.zeros((n+2,n+2))

    # Because everything sums to 1
    phi_l = 1 - phi_s - phi_g


def update(frame):
    global phi_s, phi_g, phi_l, t, itr_dir, itr_index

    for itr in range(5):
        axs[0,0].cla()
        axs[0,1].cla()
        axs[0,2].cla()
        axs[1,0].cla()
        axs[1,1].cla()
        axs[1,2].cla()
    
        phi_s1 = phi_s
        phi_g1 = phi_g
        phi_l1 = phi_l
        curr_p = p(phi_g1 + phi_l1)
        #print(curr_p)
        k_s1 = dphi_s_dt(phi_s1, phi_g1, curr_p)
        k_g1 = dphi_g_dt(phi_s1, phi_g1, curr_p)
        k_l1 = -(k_s1 + k_g1)
    
        phi_s2 = phi_s + k*k_s1/2
        phi_g2 = phi_g + k*k_g1/2
        phi_l2 = phi_l + k*k_l1/2
        curr_p = p(phi_g2 + phi_l2)
        #print(curr_p)
        k_s2 = dphi_s_dt(phi_g2, phi_s2, curr_p)
        k_g2 = dphi_g_dt(phi_g2, phi_s2, curr_p)
        k_l2 = -(k_s2 + k_g2)
    
        phi_s3 = phi_s + k*k_s2/2
        phi_g3 = phi_g + k*k_g2/2
        phi_l3 = phi_l + k*k_l2/2
        curr_p = p(phi_g3 + phi_l3)
        k_s3 = dphi_s_dt(phi_g3, phi_s3, curr_p)
        k_g3 = dphi_g_dt(phi_g3, phi_s3, curr_p)
        k_l3 = -(k_s3 + k_g3)
    
        phi_s4 = phi_s + k*k_s3
        phi_g4 = phi_g + k*k_g3
        phi_l4 = phi_l + k*k_l3
        curr_p = p(phi_g4 + phi_l4)
        k_s4 = dphi_s_dt(phi_g4, phi_s4, curr_p)
        k_g4 = dphi_g_dt(phi_g4, phi_s4, curr_p)
        k_l4 = -(k_s4 + k_g4)
    
        phi_s += k * (k_s1 + 2*k_s2 + 2*k_s3 + k_s4) / 6
        phi_g += k * (k_g1 + 2*k_g2 + 2*k_g3 + k_g4) / 6
        phi_l += k * (k_l1 + 2*k_l2 + 2*k_l3 + k_l4) / 6
    
        t += k
    
    # PLOTTING #
    fig.suptitle("t = %.5f" % t)
    
    axs[0,0].imshow(phi_s, vmin=0.2, vmax=0.9, cmap='binary')
    axs[0,0].set_title("$\phi$")
    
    axs[0,1].imshow(s()[:-1].reshape((n+2,n+2)))
    axs[0,1].set_title("s")
    
    axs[0,2].plot_surface(xs_mesh, ys_mesh, curr_p)
    #axs[0,2].imshow(curr_p)
    axs[0,2].set_title("$p$")
    
    #axs[1,0].imshow(kappa(phi_s) * np.sqrt(dp2(curr_p)))
    #axs[1,0].set_title(r"$|\kappa(\phi)\nabla p|$")
    axs[1,0].imshow(np.sqrt(dp2(curr_p)))
    axs[1,0].set_title(r"$|\nabla p|$")
    
    #axs[1,1].imshow((k_s1 + 2*k_s2 + 2*k_s3 + k_s4) / 6)
    #axs[1,1].set_title("$\partial_t \phi$")
    
    axs[1,2].imshow(sigma(phi_s))
    axs[1,2].set_title("$\sigma$")

    #print(np.min(k_s1), np.max(k_s1), t)
    #print(np.min(k_g1), np.max(k_g1))
    print()
    
    # Save plots
    if t - np.floor(t/save_interval)*save_interval < k:
        index = 0
        files = os.listdir(itr_dir)
        while "fig%d.png" % index in files:
            index += 1
        plt.savefig(itr_dir + "/fig%d.png" % index)
    
    # Restart after enough time (aesthetic)
    if t > max_t:
        itr_index += 1
        itr_dir = "img/itr%d" % itr_index
        os.mkdir(itr_dir)
        init()
    return fig,


ani = anim.FuncAnimation(fig, update, init_func=init, save_count=200)
ani.save(itr_dir + "/anim.mp4")
#plt.show() 














