import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

n = 30
h = 1 / (n+1)
h2 = h*h
k = 0.5*h
maxt = 1
frames = int(np.ceil(maxt / k))

xs = np.linspace(h, 1-h, n)
ys = np.linspace(h, 1-h, n)
xs_mesh, ys_mesh = np.meshgrid(xs, ys)
xs_mesh = xs_mesh.ravel()
ys_mesh = ys_mesh.ravel()

phi_star = 0.5
omega = 1

phi = 0.5 * np.ones_like(xs_mesh) + 0.2 * np.random.random(size=n*n)
p = np.zeros_like(xs_mesh)

#der_center = 1./(2*h) * (np.eye(n, k=1) - np.eye(n, k=-1))
#der_forward = 1./h * (np.eye(n, k=1) - np.eye(n))
#der_backward = 1./h * (np.eye(n) - np.eye(n, k=-1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def source(t):
    #s = []
    #for i in range(n):
    #    for j in range(n):
    #        s.append(0.1)
    s = 0.01 * np.ones(n*n)
    return np.array(s)

times = []
imgs = []
t = 0
while t < maxt:
    print("TIME", t)

    #"""
    # Update the phi stuff
    dp2 = np.zeros_like(xs_mesh)
    for i in range(n):
        for j in range(n):
            curr_i = i*n+j
            dpdx, dpdy = 0,0
            if i != 0: dpdy -= p[(i-1)*n+j]/2
            if i != n-1: dpdy += p[(i+1)*n+j]/2
            if j != 0: dpdx -= p[i*n+(j-1)]/2
            if j != n-1: dpdx += p[i*n+(j+1)]/2
            dp2[curr_i] = (dpdx*dpdx + dpdy*dpdy) / h2

    psi = sigmoid(omega * (phi - phi_star))

    dphidt = np.multiply(phi, np.maximum(0, dp2 - psi))
    print(np.max(dp2), np.max(psi))
    phi -= k * dphidt
    #"""

    #"""
    # Permeability
    kappa = np.divide(np.power(1-phi, 3), np.power(phi, 2))
    dkappady = np.zeros_like(kappa)
    dkappadx = np.zeros_like(kappa)
    for i in range(n):
        for j in range(n):
            curr_i = i*n+j
            if i != 0: dkappady[curr_i] -= kappa[(i-1)*n+j]/2
            if i != n-1: dkappady[curr_i] += kappa[(i+1)*n+j]/2
            if j != 0: dkappadx[curr_i] -= kappa[i*n+(j-1)]/2
            if j != n-1: dkappadx[curr_i] += kappa[i*n+(j+1)]/2

    # Solve the steady-state fluid eq
    s = source(t)
    laplacian = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            curr_i = i*n+j
            laplacian[curr_i][curr_i] += 4*kappa[curr_i]
            if i != 0: laplacian[curr_i][(i-1)*n+j] -= kappa[curr_i]
            if i != n-1: laplacian[curr_i][(i+1)*n+j] -= kappa[curr_i]
            if j != 0: laplacian[curr_i][i*n+(j-1)] -= kappa[curr_i]
            if j != n-1: laplacian[curr_i][i*n+(j+1)] -= kappa[curr_i]

            if i != 0: laplacian[curr_i][(i-1)*n+j] -= dkappady[curr_i]/2
            if i != n-1: laplacian[curr_i][(i+1)*n+j] += dkappady[curr_i]/2
            if j != 0: laplacian[curr_i][i*n+(j-1)] -= dkappadx[curr_i]/2
            if j != n-1: laplacian[curr_i][i*n+(j+1)] += dkappadx[curr_i]/2

    laplacian /= h2
    p = np.dot(np.linalg.inv(laplacian), s)
    #"""

    t += k

    times.append(t)
    imgs.append(phi.reshape((n,n)))
    #imgs.append(dp2.reshape((n,n)))

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
