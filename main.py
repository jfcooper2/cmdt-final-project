import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

# Number of interior cells for each dimension
n = 60
# Space step
h = 1 / (n+1)
h2 = h*h
# Time step
k = 0.1*h

# Full duration of simulation
maxt = 0.1
# Number of iterations
frames = int(np.ceil(maxt / k))

# xs - xcoords of cells
# ys - ycoords of cells
xs = np.linspace(0, 1, n+2)
ys = np.linspace(0, 1, n+2)
xs_mesh, ys_mesh = np.meshgrid(xs, ys)
xs = xs_mesh.ravel()
ys = ys_mesh.ravel()

# Sigmoid translation terms
phi_star = 0.7
omega = 6.5

boundary = np.zeros((n+2,n+2))
for i in range(-1,n+1):
    # Everything moves outward
    boundary[-1, i] = 1
    boundary[ n, i] = 1
    boundary[ i,-1] = 1
    boundary[ i, n] = 1
boundary *= 0.1

# Solid material density
phi = 0.8 * np.ones((n+2, n+2)) + 0.02 * np.random.random(size=(n+2, n+2))
# Pressure field
p = np.zeros((n+2, n+2))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def source(t):
    #s = 0.01 * np.ones((n+2,n+2))
    #s = 0.01 * np.ones((n,n))
    s = np.zeros((n,n))
    low = int(n/3)
    high = int(2*n/3)
    s[low:high,low:high] = 0.1
    return np.array(s)

times = []
imgs = []
t = 0
while t < maxt:
    t += k

    print("TIME", t)

    # Permeability
    kappa = np.divide(np.power(1-phi, 3), np.power(phi, 2))
    print("Max kappa", np.max(np.abs(kappa)))
    dkappadx = np.zeros_like(kappa)
    dkappady = np.zeros_like(kappa)
    for i in range(n):
        for j in range(n):
            dkappadx[i,j] += (kappa[  i,j+1] - kappa[  i,j-1])/(2*h)
            dkappady[i,j] += (kappa[i+1,  j] - kappa[i-1,  j])/(2*h)

    #"""
    # Solve the steady-state fluid eq
    s = source(t).ravel()
    laplacian = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            curr_i = i*n+j
            laplacian[curr_i][curr_i] += 4*kappa[i,j]
            if i != 0:   laplacian[curr_i][(i-1)*n+    j] -= kappa[i,j]
            if i != n-1: laplacian[curr_i][(i+1)*n+    j] -= kappa[i,j]
            if j != 0:   laplacian[curr_i][    i*n+(j-1)] -= kappa[i,j]
            if j != n-1: laplacian[curr_i][    i*n+(j+1)] -= kappa[i,j]

            if i != 0:   laplacian[curr_i][(i-1)*n+    j] -= dkappady[i,j]/(2*h2)
            if i != n-1: laplacian[curr_i][(i+1)*n+    j] += dkappady[i,j]/(2*h2)
            if j != 0:   laplacian[curr_i][    i*n+(j-1)] -= dkappadx[i,j]/(2*h2)
            if j != n-1: laplacian[curr_i][    i*n+(j+1)] += dkappadx[i,j]/(2*h2)

    laplacian_inv = np.linalg.inv(laplacian)
    p[:n,:n] = np.dot(laplacian_inv, s).reshape((n,n))
    for i in range(n):
        # Max because negative pressures don't make sense
        # TODO: Using inner kappas, not those on the boundary
        #p[  i, -1] = np.max([0, p[  i,  0] - (boundary[  i, -1] / kappa[  i,  0])])
        #p[  i,  n] = np.max([0, p[  i,n-1] - (boundary[  i,  n] / kappa[  i,n-1])])
        #p[ -1,  i] = np.max([0, p[  0,  i] - (boundary[ -1,  i] / kappa[  0,  i])])
        #p[  n,  i] = np.max([0, p[n-i,  i] - (boundary[  n,  i] / kappa[n-i,  i])])
        p[  i, -1] = p[  i,  0] - (boundary[  i, -1] / kappa[ i,-1]) * h
        p[  i,  n] = p[  i,n-1] - (boundary[  i,  n] / kappa[ i, n]) * h
        p[ -1,  i] = p[  0,  i] - (boundary[ -1,  i] / kappa[-1, i]) * h
        p[  n,  i] = p[n-1,  i] - (boundary[  n,  i] / kappa[ n, i]) * h
        #p[  i, -1] = p[  i,  0] - (boundary[  i, -1])
        #p[  i,  n] = p[  i,n-1] - (boundary[  i,  n])
        #p[ -1,  i] = p[  0,  i] - (boundary[ -1,  i])
        #p[  n,  i] = p[n-i,  i] - (boundary[  n,  i])
    #p[-1,-1] = p[  0,  0] - (boundary[-1,-1] / kappa[  0,  0])
    #p[-1, n] = p[  0,n-1] - (boundary[-1, n] / kappa[  0,n-1])
    #p[ n,-1] = p[n-1,  0] - (boundary[ n,-1] / kappa[n-1,  0])
    #p[ n, n] = p[n-1,n-1] - (boundary[ n, n] / kappa[n-1,n-1])
    #"""


    # Update the phi stuff
    dp2 = np.zeros_like(p)
    #"""
    for i in range(n):
        for j in range(n):
            dpdx = (p[  i,j+1] - p[  i,j-1])/2
            dpdy = (p[i+1,  j] - p[i-1,  j])/2
            dp2[i,j] = (dpdx*dpdx + dpdy*dpdy) / h2

    # Boundary flux of p is given by the boundary
    for i in range(n):
        # TODO: Using inner kappas, not those on the boundary
        #dp2[ i,-1] = (boundary[ i,-1]) ** 2
        #dp2[ i, n] = (boundary[ i, n]) ** 2
        #dp2[-1, i] = (boundary[-1, i]) ** 2
        #dp2[ n, i] = (boundary[ n, i]) ** 2
        dp2[ i,-1] = (boundary[ i,-1] / kappa[ i,-1]) ** 2
        dp2[ i, n] = (boundary[ i, n] / kappa[ i, n]) ** 2
        dp2[-1, i] = (boundary[-1, i] / kappa[-1, i]) ** 2
        dp2[ n, i] = (boundary[ n, i] / kappa[ n, i]) ** 2
    #dp2[-1,-1] = 0.5 * (boundary[-1,-1] / kappa[  0,  0]) ** 2
    #dp2[-1, n] = 0.5 * (boundary[-1, n] / kappa[  0,n-1]) ** 2
    #dp2[ n,-1] = 0.5 * (boundary[ n,-1] / kappa[n-1,  0]) ** 2
    #dp2[ n, n] = 0.5 * (boundary[ n, n] / kappa[n-1,n-1]) ** 2
    #"""

    psi = sigmoid(omega * (phi - phi_star))

    dphidt = np.multiply(phi, np.maximum(0, dp2 - psi))
    #dphidt = np.multiply(phi, dp2 - psi)
    phi -= k * dphidt


    times.append(t)
    #imgs.append(phi[:n,:n].reshape((n,n)))
    #imgs.append(kappa.reshape((n+2,n+2)))
    #imgs.append(kappa[:n,:n].reshape((n,n)))
    #imgs.append(dphidt[:n,:n].reshape((n,n)))
    #imgs.append(p[:n,:n].reshape((n,n)))
    imgs.append(dp2[:n,:n].reshape((n,n)))
    #imgs.append(dphidt[:n,:n].reshape((n,n)))

#plt.imshow(phi.reshape((n,n)))
#plt.colorbar()
#plt.show()

fig, ax = plt.subplots()

def update(frame):
    global imgs

    ax.cla()
    ax.set_title("%.5f" % times[frame])
    ax.imshow(imgs[frame])
    #plt.colorbar()

    return fig,

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(imgs), interval=100)

plt.show()
