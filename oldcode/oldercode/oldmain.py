import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse.linalg as sp
from scipy.sparse import csr_matrix, coo_matrix
from scipy.signal import convolve2d

do_lagrange = False
do_red_inte = True # idk what this is actually called

# Model params
n = 100                 # Grid points per dimension
a, b = 0, 1             # Range of simulation

omega = 6.5             # Ground stiffness
phi_star = 0.70         # Erosive concentration thres
phi_zero = 0.75         # Initial concentration value
max_t =  5              # Time before sim restart
zeta = 2                # Concentration of noise field
sigma_phi = 0.20        # Amplitude of noise field
xi = 1                  # Std of correlation kernel

red_inte_eps = 1e-5     # Regularization for pressure
erosion_thres =  0.0    # Parameter to threshold the max term in dphi
erosion_mul = 1.0       # Parameter in maximum (multiplicative)
n_blur_width = 8        # Cell width to do correlation with

# Von Neumann BCs
boundary = np.zeros((n+2, n+2))

boundary[-1, 48:52] = 0.5 # Top
boundary[ n, 48:52] = 0 # Bottom
boundary[ 48:52,-1] = -0.5 # Left
boundary[ 48:52, n] = -0.5 # Right
"""
boundary[-1, :] = -1
boundary[n, :] = 1
"""

# Internal Fluid Source
sources = np.zeros((n,n))
sources[:,:] = 0.0
#sources[40:60,40:60] = 1

# Derived params
h = (b-a) / (n+2)
k = 0.1*h

# Plotting coordinates
xs = np.linspace(0+h, 1-h, n)
ys = np.linspace(0+h, 1-h, n)
xs_mesh, ys_mesh = np.meshgrid(xs, ys)

# Coordinate conversions
ij2idx = {} # Convert i,j coords to vector indices
idx2ij = {} # Convert vector indices to i,j coords
for i in range(-1, n+2):
    for j in range(-1, n+2):
        index = j+i*(n+2)
        if j == -1:
            index += n+2
        if i == -1:
            index += (n+2)*(n+2)
        ij2idx[(i,j)] = index
        idx2ij[index] = (i,j)

def kappa(phi):
    #return np.power(phi, 2) / np.power(1-phi, 3)
    return np.power(1-phi, 3) / np.power(phi, 2)

def psi(phi):
    return 0.5 * (np.tanh(omega*(phi-phi_star)) + 1)
    #return phi
    #return 0.5 * (np.tanh(omega*(phi-phi_star)) + 1)

def get_dp2(p):
    idx_forward  = np.arange(n) + 1
    idx_current  = np.arange(n)
    idx_backward = np.arange(n) - 1

    # Forward difference * Backward difference
    """
    dpdx  = (p[:, idx_forward] - p[:, idx_current])  / (h)
    dpdx *= (p[:, idx_current] - p[:, idx_backward]) / (h)
    dpdy  = (p[idx_forward, :] - p[idx_current, :])  / (h)
    dpdy *= (p[idx_current, :] - p[idx_backward, :]) / (h)
    """

    # Centered difference squared
    #"""
    dpdx  = (p[:, idx_forward] - p[:, idx_backward]) / (2*h)
    dpdx *= (p[:, idx_forward] - p[:, idx_backward]) / (2*h)
    dpdy  = (p[idx_forward, :] - p[idx_backward, :]) / (2*h)
    dpdy *= (p[idx_forward, :] - p[idx_backward, :]) / (2*h)
    #"""

    dp2 = np.square(boundary) # Boundary gradients are given
    dp2[:n,:n] += dpdx[:n,:]  # Core gradients are calculated
    dp2[:n,:n] += dpdy[:,:n]  # Core gradients are calculated
    #dp2[:,:n] += dpdx  # Core gradients are calculated
    #dp2[:n,:] += dpdy  # Core gradients are calculated

    return dp2

def dphidt(phi, p):
    idx_forward  = np.arange(n) + 1
    idx_current  = np.arange(n)
    idx_backward = np.arange(n) - 1

    dp2 = get_dp2(p)

    other_phi = phi

    # Blurring
    #"""
    xs = np.arange(2*n_blur_width+1)
    ys = np.arange(2*n_blur_width+1)
    xs, ys = np.meshgrid(xs, ys)
        
    dists = (n_blur_width - xs)**2 + (n_blur_width - ys)**2

    kernel = np.exp(-dists/(2*xi)) # Make local Gaussian kernel
    kernel /= np.sum(kernel)

    other_phi = convolve2d(other_phi, kernel, boundary="symm", mode="same")
    #"""

    return phi * np.maximum(0, dp2 - erosion_mul * psi(other_phi) - erosion_thres)

def get_source(t):
    source = t * boundary.copy() # Boundary conditions

    # Core source terms
    source[:n,:n] = sources

    source = source.ravel()  # Make into vector

    # Lagrange Multiplier
    if do_lagrange:
        # Make source larger for the lambda term
        source_ret = np.zeros((n+2)**2+1) 
        source_ret[:-1] = source
        source = source_ret

    return source

def get_p_arr(g):
    row = []
    col = []
    data = []

    for i in range(n):
        # Core
        for j in range(n):
            index = ij2idx[i,j] # This is like having n+2 be sqrt(-1) in \mathbb{C}

            if True: # For indentation
                stencil = np.zeros((3,3))
                
                # Centered difference non-diffusive component
                #"""
                stencil[-1, 0] =   4*g[i,j] - g[i+1,j] + g[i-1,j]
                stencil[ 0,-1] =   4*g[i,j] - g[i,j+1] + g[i,j-1]
                stencil[ 0, 0] = -16*g[i,j]
                stencil[ 0, 1] =   4*g[i,j] + g[i,j+1] - g[i,j-1]
                stencil[ 1, 0] =   4*g[i,j] + g[i+1,j] - g[i-1,j]
                stencil /= 4*h*h
                #"""

                # Flux across each diffusive boundary
                """
                g_l = (g[i,j] + g[i,j-1]) / 2
                g_r = (g[i,j] + g[i,j+1]) / 2
                g_u = (g[i,j] + g[i-1,j]) / 2
                g_d = (g[i,j] + g[i+1,j]) / 2
                stencil[-1, 0] =  g_u # u
                stencil[ 0,-1] =  g_l # l
                stencil[ 0, 0] = -(g_u+g_d+g_l+g_r)
                stencil[ 0, 1] =  g_r # r
                stencil[ 1, 0] =  g_d # d
                stencil /= h*h
                """

                # Regularize pressure degree of freedom
                if do_red_inte:
                    stencil[ 0, 0] += red_inte_eps * np.random.random()
    
                data.append(stencil[-1, 0])
                row.append(ij2idx[i,j])
                col.append(ij2idx[i-1,j])

                data.append(stencil[0, -1])
                row.append(ij2idx[i,j])
                col.append(ij2idx[i,j-1])
    
                data.append(stencil[0, 0])
                row.append(ij2idx[i,j])
                col.append(ij2idx[i,j])
        
                data.append(stencil[0, 1])
                row.append(ij2idx[i,j])
                col.append(ij2idx[i,j+1])
            
                data.append(stencil[1, 0])
                row.append(ij2idx[i,j])
                col.append(ij2idx[i+1,j])
    
                # Lagrange Multiplier
                if do_lagrange:
                    data.append(1)
                    row.append(index)
                    col.append((n+2)*(n+2))
                    data.append(1)
                    row.append((n+2)*(n+2))
                    col.append(index)

        # Left boundary
        flux = g[i,n]
        flux = 1
        data.append(-1*flux/h)
        row.append(ij2idx[i,n]) 
        col.append(ij2idx[i,n-1])
        data.append(1*flux/h)
        row.append(ij2idx[i,n]) 
        col.append(ij2idx[i,n])

        # Right boundary
        flux = g[i,-1]
        flux = 1
        data.append(-1*flux/h)
        row.append(ij2idx[i,-1]) 
        col.append(ij2idx[i,0])
        data.append(1*flux/h)
        row.append(ij2idx[i,-1]) 
        col.append(ij2idx[i,-1])

    # Top and Bottom
    for j in range(n):

        # Top boundary
        flux = g[n,j]
        flux = 1
        data.append(-1*flux/h)
        row.append(ij2idx[n,j]) 
        col.append(ij2idx[n-1,j])
        data.append(1*flux/h)
        row.append(ij2idx[n,j]) 
        col.append(ij2idx[n,j])

        # Bottom boundary
        flux = g[-1,j]
        flux = 1
        data.append(-1*flux/h)
        row.append(ij2idx[-1,j]) 
        col.append(ij2idx[0,j])
        data.append(1*flux/h)
        row.append(ij2idx[-1,j]) 
        col.append(ij2idx[-1,j])

    # Corners (don't really matter. Just for non-singular A
    data.append(1)
    row.append(ij2idx[n,n])
    col.append(ij2idx[n,n])
    
    data.append(1)
    row.append(ij2idx[n,-1])
    col.append(ij2idx[n,-1])
    
    data.append(1)
    row.append(ij2idx[-1,n])
    col.append(ij2idx[-1,n])
    
    data.append(1)
    row.append(ij2idx[-1,-1])
    col.append(ij2idx[-1,-1])

    n2 = (n+2)**2
    if do_lagrange: n2 += 1 # For the lambda var

    row, col, data = np.array(row), np.array(col), np.array(data)
    arr = csr_matrix((data, (row, col)), shape=(n2,n2))

    return arr


fig = plt.figure()

axs = np.array([[None, None, None], [None, None, None]])
axs[0,0] = fig.add_subplot(2, 3, 1)
axs[0,1] = fig.add_subplot(2, 3, 2)
axs[0,2] = fig.add_subplot(2, 3, 3, projection='3d')
axs[1,0] = fig.add_subplot(2, 3, 4)
axs[1,1] = fig.add_subplot(2, 3, 5)


def init():
    global phi, t
    t = 0

    # Totally Random Field
    """
    phi = phi_zero + min([sigma_phi/n, 0.9-phi_zero]) * np.random.random((n+2,n+2))
    """

    # Random Gaussian Spots
    #"""
    phi = np.zeros((n+2, n+2))
    #for itr in range(int(n*n/10)):
    for itr in range(int(n*n)):
        xs = np.arange(n+2)
        xs[-1] -= n+2
        ys = np.arange(n+2)
        ys[-1] -= n+2
        xs, ys = np.meshgrid(xs, ys)
        
        mu = np.random.uniform(-1, n+1, 2)
        dists = (mu[0] - xs)**2 + (mu[1] - ys)**2

        phi += np.sqrt(2*np.pi/zeta) * np.exp(-dists/(2 * zeta**2))
    phi_min = np.min(phi)
    phi_max = np.max(phi)
    phi = phi_zero + sigma_phi * ((phi - phi_min) / (phi_max - phi_min))
    #"""

    return fig,

xs_grid, ys_grid = np.meshgrid(xs, ys)

def update(frame):
    global phi, t
    # Show current frame
    axs[0,0].cla()
    axs[0,1].cla()
    axs[0,2].cla()
    axs[1,0].cla()
    axs[1,1].cla()

    # Time update
    t += k

    # Pressure update
    p_arr = get_p_arr(kappa(phi))
    p = sp.spsolve(p_arr, get_source(t))
    if do_lagrange: p = p[:-1] # Remove lambda term
    p.shape = (n+2,n+2)

    # Mass concentration update (RK4??)
    """
    k1 = dphidt(phi, p)
    k2 = dphidt(phi - (k/2) * k1, p)
    k3 = dphidt(phi - (k/2) * k2, p)
    k4 = dphidt(phi - k * k3, p)
    phi -= (k/6) * (k1 + 2*k2 + 2*k3 + k4)
    """

    # Mass concentration update (RK1)
    #"""
    phi -= k * dphidt(phi, p)
    #"""
    
    fig.suptitle("t = %.5f" % t)

    axs[0,0].imshow(phi, vmin=0.2, vmax=0.9, cmap='binary')
    axs[0,0].set_title(r"$\phi$")

    axs[0,1].imshow(get_source(t).reshape((n+2,n+2)))
    axs[0,1].set_title("s")

    axs[0,2].plot_surface(xs_grid, ys_grid, p[:n,:n])
    axs[0,2].set_title("p")
    axs[0,2].axis('off')

    axs[1,0].imshow(get_dp2(p)[:n,:n])
    axs[1,0].set_title(r"$|\nabla p|^2$")

    axs[1,1].imshow(dphidt(phi, p))
    axs[1,1].set_title(r"$\partial_t p$")

    #axs[1,0].imshow(p[:n,:n])
    #ax.imshow(p[1:n,:n]-p[:n-1,:n])
    #ax.imshow(dphi.reshape(n+2,n+2), vmin=0, vmax=1)
    #ax.imshow(p_arr.toarray()*4)
    #ax.imshow(kappa(phi)[:n,:n])
    #ax.imshow(psi(phi))
    #ax.imshow(p[:n,:n])

    dp2 = get_dp2(p)
    print(np.min(dp2[:n,:n]), np.max(dp2[:n,:n]))
    #print(np.max(p[:n,:n])-np.min(p[:n,:n]),np.max(p[:n,:n]),np.min(p[:n,:n]))

    # Restart after enough time (aesthetic)
    if t > max_t:
        init()
    return fig,

ani = anim.FuncAnimation(fig, update, init_func=init)
plt.show() 





