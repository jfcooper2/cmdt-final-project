# FINITE VOLUME METHODS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse.linalg as sp
from scipy.sparse import csr_matrix
from scipy.signal import convolve2d
import json
import os
import pypardiso

#np.set_printoptions(precision=4)

### PARAMETERS ###

n = 100
a, b = 0, 1

h = 1./(n+2)
h2 = h*h

k_star = 0.5 * h
save_interval = 0.25
max_t = 5

e0 = 1.0
d0 = 1.0
gamma = 20.0

omega = 2 * np.pi
d_phi_star = 0.70
e_phi_star = 0.70

phi_s_min  = 0.80
phi_s_max  = 0.90
zeta = 0.0025 # 0.01

n_blur_width = 25
xi = 0.5 # 2

max_decay = 1000 # Totally a stop-gap measure (FIXED)

source = np.zeros((n,n)) 
#source[45:55,45:55] = 0.1

# Boundary terms
boundary = np.zeros((n+2,n+2))

# Edges
boundary[48:52,n+1] = -5
#boundary[0,:] = 1./5
#boundary[n+1,:] = 1./5
#boundary[0,:] = -1
boundary[:,0] = 1./5
#boundary[n+1,:] = 1
#boundary[:,n+1] = 2./5
#boundary[0,48:52] = -3
#boundary[n+1,48:52] = 3
#boundary[48:52,0] = 3

# Corners
boundary[0,0] = 0
boundary[n+1,0] = 0
boundary[0,n+1] = 0
boundary[n+1,n+1] = 0

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
    #s = boundary.copy()
    s[1:n+1,1:n+1] = source
    s_lagrange = np.zeros((n+2)*(n+2)+1)
    s_lagrange[:-1] = s.ravel()
    return s_lagrange

def kappa(phi):
    return (phi ** 3) / ((1 - phi) ** 2)

def dp2(p):
    """
    dpdx  = (p[:,2:] - p[:,1:-1]) / (h)
    dpdx *= (p[:,1:-1] - p[:,:-2]) / (h)
    dpdy  = (p[2:,:] - p[1:-1,:]) / (h)
    dpdy *= (p[1:-1,:] - p[:-2,:]) / (h)
    dp = np.square(boundary)
    dp[:,1:-1] += np.abs(dpdx)
    dp[1:-1,:] += np.abs(dpdy)
    """

    #"""
    dpdx  = (p[:,2:] - p[:,:-2]) / (2*h)
    dpdx *= (p[:,2:] - p[:,:-2]) / (2*h)
    dpdy  = (p[2:,:] - p[:-2,:]) / (2*h)
    dpdy *= (p[2:,:] - p[:-2,:]) / (2*h)
    dp = np.square(boundary)
    dp[:,1:-1] += dpdx
    dp[1:-1,:] += dpdy
    #"""
    return dp

def sigma(phi):
    global omega

    xs = np.arange(2*n_blur_width+1)
    ys = np.arange(2*n_blur_width+1)
    xs, ys = np.meshgrid(xs, ys)
        
    dists = (n_blur_width - xs)**2 + (n_blur_width - ys)**2

    kernel = np.exp(-dists/(2*xi)) # Make local Gaussian kernel
    kernel /= np.sum(kernel)

    other_phi = convolve2d(phi, kernel, boundary="symm", mode="same")

    return 0.5 * (np.tanh(omega * (other_phi - e_phi_star)) + 1)

def e(phi_s, phi_g, p):
    return phi_s * np.minimum(max_decay, e0 * np.maximum(0, dp2(p) / gamma - sigma(phi_s)))

def d(phi_s, phi_g, p):
    return phi_g * np.minimum(max_decay, d0 * np.maximum(0, phi_s - d_phi_star))

def dphi_s_dt(phi_s, phi_g, p):
    dd = d(phi_s, phi_g, p)
    ee = e(phi_s, phi_g, p)
    #print(np.min(dd), np.max(dd))
    #print(np.min(ee), np.max(ee))
    #print(np.min(dp2(p)), np.max(dp2(p)))
    #print()

    #ret = np.zeros_like(phi_s)
    #ret = - e(phi_s, phi_g, p)
    ret = d(phi_s, phi_g, p) - e(phi_s, phi_g, p)
    return np.nan_to_num(ret, nan=0.0)

def dphi_g_dt(phi_s, phi_g, p):
    phi = 1-phi_s # 1-psi_s = psi_g + psi_l
    #ret = np.zeros_like(phi_s)
    #ret = e(phi_s, phi_g, p)
    ret = e(phi_s, phi_g, p) - d(phi_s, phi_g, p)
    #ret = (e(phi_s, phi_g, p) - d(phi_s, phi_g, p)) / 2
    #ret = e(phi_s, phi_g, p) - d(phi_s, phi_g, p) - apply_A(kappa(phi), p)
    #ret = e(phi_s, phi_g, p) - d(phi_s, phi_g, p) + apply_A(kappa(phi), p) / 2
    #ret = e(phi_s, phi_g, p) - d(phi_s, phi_g, p) - apply_A(kappa(phi), p) / 2
    #ret = phi_g / phi * apply_A(kappa(phi), p)
    #ret = e(phi_s, phi_g, p) - d(phi_s, phi_g, p) - phi_g / phi * apply_A(kappa(phi), p)
    #ret = e(phi_s, phi_g, p) - d(phi_s, phi_g, p) + apply_A(phi_g / phi * kappa(phi), p) / 2
    #ret = e(phi_s, phi_g, p) - d(phi_s, phi_g, p) - apply_A(phi_g / phi * kappa(phi), p)
    return np.nan_to_num(ret, nan=0.0)



def A(kappa_phi, do_lagrange=True):
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
                #"""
                flow = (kappa_phi[0,j] + kappa_phi[1,j]) / 2
                data = [flow/h, -flow/h]
                row = [0, 1]
                col = [j, j]
                #"""

                """
                flow = kappa_phi[0,j]
                data = [-3*flow/(2*h), 4*flow/(2*h), -flow/(2*h)]
                row = [0, 1, 2]
                col = [j, j, j]
                """
            if j == n+1:
                # Corner
                data = [-2, 1, 1]
                row = [0, 0, 1]
                col = [n+1, n, n+1]
        if 1 <= i and i <= n:
            if j == 0:
                # Edge
                #"""
                flow = (kappa_phi[i,0] + kappa_phi[i,1]) / 2
                data = [flow/h, -flow/h]
                row = [i, i]
                col = [0, 1]
                #"""

                """
                flow = kappa_phi[i,0]
                data = [-3*flow/(2*h), 4*flow/(2*h), -flow/(2*h)]
                row = [i, i, i]
                col = [0, 1, 2]
                """
            if 1 <= j and j <= n:

                # Core
                #"""
                flow_down  = (kappa_phi[i,j] + kappa_phi[i-1,j]) / 2
                flow_up    = (kappa_phi[i,j] + kappa_phi[i+1,j]) / 2
                flow_left  = (kappa_phi[i,j] + kappa_phi[i,j-1]) / 2
                flow_right = (kappa_phi[i,j] + kappa_phi[i,j+1]) / 2

                data = [-(flow_down + flow_up + flow_right + flow_left) / h2,\
                        flow_left / h2, \
                        flow_right / h2, \
                        flow_down / h2, \
                        flow_up / h2]
                #"""

                """
                data = [(-4*kappa_phi[i,j]) / h2, \
                        (kappa_phi[i,j] + 0.25 * kappa_phi[i, j-1] - 0.25 * kappa_phi[i, j+1]) / h2, \
                        (kappa_phi[i,j] + 0.25 * kappa_phi[i, j+1] - 0.25 * kappa_phi[i, j-1]) / h2, \
                        (kappa_phi[i,j] + 0.25 * kappa_phi[i-1, j] - 0.25 * kappa_phi[i+1, j]) / h2, \
                        (kappa_phi[i,j] + 0.25 * kappa_phi[i+1, j] - 0.25 * kappa_phi[i-1, j]) / h2]
                """

                row = [i, i, i, i-1, i+1]
                col = [j, j-1, j+1, j, j]

            if j == n+1:
                # Edge
                #"""
                flow = (kappa_phi[i, n+1] + kappa_phi[i, n]) / 2
                data = [flow/h, -flow/h]
                row = [i, i]
                col = [n+1, n]
                #"""

                """
                flow = kappa_phi[i,n+1]
                data = [-3*flow/(2*h), 4*flow/(2*h), -flow/(2*h)]
                row = [i, i, i]
                col = [n+1, n, n-1]
                """
        if i == n+1:
            if j == 0:
                # Corner
                data = [-2, 1, 1]
                row = [n+1, n+1, n]
                col = [0, 1, 0]
            if 1 <= j and j <= n:
                # Edge
                #"""
                flow = (kappa_phi[n+1,j] + kappa_phi[n,j]) / 2
                data = [flow/h, -flow/h]
                row = [n+1, n]
                col = [j, j]
                #"""

                """
                flow = kappa_phi[n+1,j]
                data = [-3*flow/(2*h), 4*flow/(2*h), -flow/(2*h)]
                row = [n+1, n, n-1]
                col = [j, j, j]
                """
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

    n2 = (n+2) * (n+2)
    if do_lagrange:
        n2 += 1
    row, col, data = np.array(arr_row), np.array(arr_col), np.array(arr_data)
    arr = csr_matrix((data, (row, col)), shape=(n2,n2))

    return arr

def apply_A(kappa_phi, p):
    return (A(kappa_phi, do_lagrange=False) * p.ravel()).reshape((n+2,n+2))

def p(phi):
    return pypardiso.spsolve(A(kappa(phi)), s())[:-1].reshape((n+2,n+2))



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


params = {}
params['desc'] = "Adaptive"
params['n'] = n
params['a'] = a
params['b'] = b
params['k'] = k_star
params['max_t'] = max_t
params['e0'] = e0
params['d0'] = d0
params['gamma'] = gamma
params['omega'] = omega
params['d_phi_star'] = d_phi_star
params['e_phi_star'] = e_phi_star
params['phi_s_min'] = phi_s_min
params['phi_s_max'] = phi_s_max
params['zeta'] = zeta
params['n_blur_width'] = n_blur_width
params['xi'] = xi
params['max_decay'] = max_decay
with open(itr_dir + "/params.json", "w") as outfile:
    json.dump(params, outfile)


def init():
    global phi_s, phi_g, phi_l, t

    t = 0

    phi = np.zeros((n+2,n+2))
    for itr in range(int(n*n)):
        xs = h * np.arange(n+2)
        ys = h * np.arange(n+2)
        xs, ys = np.meshgrid(xs, ys)

        mu = np.random.uniform(-2*zeta, 1+2*zeta, 2)
        dists = (mu[0] - xs)**2 + (mu[1] - ys)**2

        phi += np.sqrt(2*np.pi/zeta) * np.exp(-dists/(2 * zeta**2))
    phi_top = np.min(phi)
    phi_bot = np.max(phi)

    phi_s = phi_s_min + (phi_s_max - phi_s_min) * ((phi - phi_bot) / (phi_top - phi_bot))
    #phi_s[:(n//2)] -= 0.10

    phi_g = np.zeros_like(phi_s)
    #phi_g = (1-phi_s)/2
    phi_l = 1-phi_s-phi_g

    print(np.min(phi_s), np.max(phi_s))


def update(frame):
    global phi_s, phi_g, phi_l, t, itr_dir, itr_index

    for itr in range(5):
        axs[0,0].cla()
        axs[0,1].cla()
        axs[0,2].cla()
        axs[1,0].cla()
        axs[1,1].cla()
        axs[1,2].cla()
    
        # RK4 Updates
        phi_s1 = phi_s
        phi_g1 = phi_g
        phi_l1 = phi_l
        curr_p = p(phi_g1 + phi_l1)
        k_s1 = dphi_s_dt(phi_s1, phi_g1, curr_p)
        k_g1 = dphi_g_dt(phi_s1, phi_g1, curr_p)
        k_l1 = -k_s1-k_g1
        k = min([k_star, 0.2 * np.min(np.abs(phi_s1 / k_s1)), 0.2 * np.min(np.abs(phi_g1 / k_g1)), 0.2 * np.min(np.abs(phi_l1 / k_l1))])
        #k = min([k_star, 0.01 / np.max(np.abs(k_s1)), 0.01 / np.max(np.abs(k_g1)), 0.01 / np.max(np.abs(k_l1))])
        if k == k_star: print("At fastest")
        else: print("---------------------------------------------------")

        phi_s2 = phi_s + k*k_s1/2
        phi_g2 = phi_g + k*k_g1/2
        phi_l2 = phi_l + k*k_l1/2
        curr_p = p(phi_g2 + phi_l2)
        k_s2 = dphi_s_dt(phi_s2, phi_g2, curr_p)
        k_g2 = dphi_g_dt(phi_s2, phi_g2, curr_p)
        k_l2 = -k_s2-k_g2

        phi_s3 = phi_s + k*k_s2/2
        phi_g3 = phi_g + k*k_g2/2
        phi_l3 = phi_l + k*k_l2/2
        curr_p = p(phi_g3 + phi_l3)
        k_s3 = dphi_s_dt(phi_s3, phi_g3, curr_p)
        k_g3 = dphi_g_dt(phi_s3, phi_g3, curr_p)
        k_l3 = -k_s3-k_g3

        phi_s4 = phi_s + k*k_s3
        phi_g4 = phi_g + k*k_g3
        phi_l4 = phi_l + k*k_l3
        curr_p = p(phi_g4 + phi_l4)
        k_s4 = dphi_s_dt(phi_s4, phi_g4, curr_p)
        k_g4 = dphi_g_dt(phi_s4, phi_g4, curr_p)
        k_l4 = -k_s4-k_g4

        # Change the state of the system
        dphi_s = (k_s1 + 2 * k_s2 + 2 * k_s3 + k_s4) / 6
        dphi_g = (k_g1 + 2 * k_g2 + 2 * k_g3 + k_g4) / 6
        dphi_l = (k_l1 + 2 * k_l2 + 2 * k_l3 + k_l4) / 6

        phi_s += k * dphi_s
        phi_g += k * dphi_g
        phi_l += k * dphi_l

        print("System sum:", np.sum(phi_s + phi_g + phi_l - 1))
        print("Diff sum:", np.sum(dphi_s + dphi_g + dphi_l))
    
        t += k
    
        # PLOTTING #
        fig.suptitle("t = %.5f" % t)
    
        axs[0,0].imshow(phi_s, vmin=0.2, vmax=0.9, cmap='binary')
        axs[0,0].set_title("$\phi_s$")
    
        axs[0,1].imshow(s()[:-1].reshape((n+2,n+2)))
        axs[0,1].set_title("s")
    
        axs[0,2].plot_surface(xs_mesh, ys_mesh, curr_p)
        #axs[0,2].imshow(curr_p)
        axs[0,2].set_title("$p$")
    
        axs[1,0].imshow(np.sqrt(dp2(curr_p)))
        axs[1,0].set_title(r"$|\nabla p|$")
        #axs[1,0].imshow(kappa(phi_s) * np.sqrt(dp2(curr_p)))
        #axs[1,0].set_title(r"$|\kappa(\phi_s)\nabla p|$")
    
        axs[1,1].imshow(dphi_s)
        axs[1,1].set_title("$\partial_t \phi_s$")
    
        axs[1,2].imshow(sigma(phi_s))
        axs[1,2].set_title("$\sigma$")
    
        print(t)
        print("phi_s [%.5f, %.5f]" % (np.min(phi_s), np.max(phi_s)))
        print("phi_g [%.5f, %.5f]" % (np.min(phi_g), np.max(phi_g)))
        print("phi_l [%.5f, %.5f]" % (np.min(phi_l), np.max(phi_l)))
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


ani = anim.FuncAnimation(fig, update, init_func=init, save_count=400)
ani.save(itr_dir + "/anim.mp4")
#plt.show() 















