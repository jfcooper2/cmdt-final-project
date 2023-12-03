import numpy as np
import scipy.sparse.linalg as sp
from scipy.sparse import csr_matrix, coo_matrix
from scipy.signal import convolve2d

def get_derivative(arr, h, axis="x", kind="centered2", order=1):
    der = None

    if order == 1: # f'
        if kind == "centered2":
            if axis == "x":
                der = (arr[1:-1, 2:] - arr[1:-1, :-2]) / (2*h)
            if axis == "y":
                der = (arr[2:, 1:-1] - arr[:-2, 1:-1]) / (2*h)

        if kind == "forward1":
            if axis == "x":
                der = (arr[1:-1, 2:] - arr[1:-1, 1:-1]) / h
            if axis == "y":
                der = (arr[2:, 1:-1] - arr[1:-1, 1:-1]) / h

        if kind == "backward1":
            if axis == "x":
                der = (arr[1:-1, 1:-1] - arr[1:-1, :-2]) / h
            if axis == "y":
                der = (arr[1:-1, 1:-1] - arr[:-2, 1:-1]) / h

        if kind == "forward2":
            if axis == "x":
                der = (arr[2:-2, 4:] - 4*arr[2:-2, 3:-1] + 3*arr[2:-2, 2:-2]) / h
            if axis == "x":
                der = (arr[4:, 2:-2] - 4*arr[3:-1, 2:-2] + 3*arr[2:-2, 2:-2]) / h

        if kind == "backward2":
            if axis == "x":
                der = (arr[2:-2, :-4] - 4*arr[2:-2, 1:-3] + 3*arr[2:-2, 2:-2]) / h
            if axis == "x":
                der = (arr[:-4, 2:-2] - 4*arr[1:-3, 2:-2] + 3*arr[2:-2, 2:-2]) / h

        if kind == "centered4": # I think?
            if axis == "x":
                der = (-arr[2:-2, 4:] + 4*arr[2:-2, 3:-1] - 4*arr[2:-2, 1:-3] + arr[2:-2, :-4]) / (2*h)
            if axis == "y":
                der = (-arr[4:, 2:-2] + 4*arr[3:-1, 2:-2] - 4*arr[1:-3, 2:-2] + arr[:-4, 2:-2]) / (2*h)

    if order == 2: # f''
        if kind == "centered2":
            if axis == "x":
                der = (arr[1:-1, 2:] - 2*arr[1:-1, 1:-1] + arr[1:-1, :-2]) / (h*h)
            if axis == "y":
                der = (arr[2:, 1:-1] - 2*arr[1:-1, 1:-1] + arr[:-2, 1:-1]) / (h*h)
        
        if kind == "laplacian":
            der = (arr[1:-1, 2:] + arr[1:-1, :-2] + arr[2:, 1:-1] + arr[:-2, 1:-1] - 4*arr[1:-1, 1:-1]) / (h*h)

    return der


def get_derivative_array(n, h, axis="x", kind="centered2", order=1, boundary=None, boundary_kind="neumann"):
    row = []
    col = []
    data = []

    if order == 1:
        if kind == "centered2":
            if axis == "x":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + (i+1) * (n+4))

                        data.append(-1/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + (i-1) * (n+4))

            if axis == "y":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/(2*h))
                        row.append(j + i * (n+4))
                        col.append((j+1) + i * (n+4))

                        data.append(-1/(2*h))
                        row.append(j + i * (n+4))
                        col.append((j-1) + i * (n+4))

        if kind == "forward1":
            if axis == "x":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/h)
                        row.append(j + i * (n+4))
                        col.append(j + (i+1) * (n+4))

                        data.append(-1/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

            if axis == "y":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/h)
                        row.append(j + i * (n+4))
                        col.append((j+1) + i * (n+4))

                        data.append(-1/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

        if kind == "backward1":
            if axis == "x":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/h)
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

                        data.append(-1/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + (i-1) * (n+4))

            if axis == "y":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/h)
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

                        data.append(-1/(2*h))
                        row.append(j + i * (n+4))
                        col.append((j-1) + i * (n+4))

        if kind == "forward2":
            if axis == "x":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/h)
                        row.append(j + i * (n+4))
                        col.append(j + (i+2) * (n+4))

                        data.append(-4/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + (i+1) * (n+4))

                        data.append(3/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

            if axis == "y":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/h)
                        row.append(j + i * (n+4))
                        col.append((j+2) + i * (n+4))

                        data.append(-4/(2*h))
                        row.append(j + i * (n+4))
                        col.append((j+1) + i * (n+4))

                        data.append(3/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

        if kind == "backward2":
            if axis == "x":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/h)
                        row.append(j + i * (n+4))
                        col.append(j + (i-2) * (n+4))

                        data.append(-4/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + (i-1) * (n+4))

                        data.append(3/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

            if axis == "y":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/h)
                        row.append(j + i * (n+4))
                        col.append((j-2) + i * (n+4))

                        data.append(-4/(2*h))
                        row.append(j + i * (n+4))
                        col.append((j-1) + i * (n+4))

                        data.append(3/(2*h))
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

    if order == 2: # f''
        if kind == "centered2":
            if axis == "x":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/(h*h))
                        row.append(j + i * (n+4))
                        col.append(j + (i-1) * (n+4))

                        data.append(-2/(h*h))
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

                        data.append(1/(h*h))
                        row.append(j + i * (n+4))
                        col.append(j + (i+1) * (n+4))

            if axis == "y":
                for i in range(2, n+2):
                    for j in range(2, n+2):
                        data.append(1/(h*h))
                        row.append(j + i * (n+4))
                        col.append((j-1) + i * (n+4))

                        data.append(-2/(h*h))
                        row.append(j + i * (n+4))
                        col.append(j + i * (n+4))

                        data.append(1/(h*h))
                        row.append(j + i * (n+4))
                        col.append((j+1) + i * (n+4))
        
        if kind == "laplacian":
            if i in range(2, n+2):
                if j in range(2, n+2):
                    data.append(1/(h*h))
                    row.append(j + i * (n+4))
                    col.append(j + (i-1) * (n+4))

                    data.append(1/(h*h))
                    row.append(j + i * (n+4))
                    col.append(j + (i+1) * (n+4))

                    data.append(1/(h*h))
                    row.append(j + i * (n+4))
                    col.append((j-1) + i * (n+4))

                    data.append(1/(h*h))
                    row.append(j + i * (n+4))
                    col.append((j+1) + i * (n+4))

                    data.append(-4/(h*h))
                    row.append(j + i * (n+4))
                    col.append(j + i * (n+4))

        if kind == "div_grad_centered2":
            if i in range(2, n+2):
                if j in range(2, n+2):
                    data.append(1/(h*h))
                    row.append(j + i * (n+4))
                    col.append(j + (i-1) * (n+4))

                    data.append(1/(h*h))
                    row.append(j + i * (n+4))
                    col.append(j + (i+1) * (n+4))

                    data.append(1/(h*h))
                    row.append(j + i * (n+4))
                    col.append((j-1) + i * (n+4))

                    data.append(1/(h*h))
                    row.append(j + i * (n+4))
                    col.append((j+1) + i * (n+4))

                    data.append(-4/(h*h))
                    row.append(j + i * (n+4))
                    col.append(j + i * (n+4))


    if True:
        # Inner Boundaries
        if boundary_kind == "dirichlet":
            for i in range(2, n+2):
                data.append(1)
                row.append(i + (n+4))
                col.append(i + (n+4))
    
                data.append(1)
                row.append(i + (n+2) * (n+4))
                col.append(i + (n+2) * (n+4))
    
                data.append(1)
                row.append(i * (n+4) + 1)
                col.append(i * (n+4) + 1)
    
                data.append(1)
                row.append(i * (n+4) + (n+2))
                col.append(i * (n+4) + (n+2))

        if boundary_kind == "neumann":
            for i in range(2, n+2):
                data.append(1/h)
                row.append(i + (n+4))
                col.append(i + 1 * (n+4))

                data.append(-1/h)
                row.append(i + (n+4))
                col.append(i + 2 * (n+4))


                data.append(1/h)
                row.append(i + (n+2) * (n+4))
                col.append(i + (n+2) * (n+4))
    
                data.append(-1/h)
                row.append(i + (n+2) * (n+4))
                col.append(i + (n+1) * (n+4))
    

                data.append(1/h)
                row.append(i * (n+4) + 1)
                col.append(i * (n+4) + 1)
    
                data.append(-1/h)
                row.append(i * (n+4) + 1)
                col.append(i * (n+4) + 2)
    

                data.append(1/h)
                row.append(i * (n+4) + (n+2))
                col.append(i * (n+4) + (n+2))

                data.append(-1/h)
                row.append(i * (n+4) + (n+2))
                col.append(i * (n+4) + (n+1))


        # Outer Boundaries
        for i in range(1, n+3):
            data.append(1)
            row.append(i)
            col.append(i)

            data.append(1)
            row.append(i + (n+3) * (n+4))
            col.append(i + (n+3) * (n+4))

            data.append(1)
            row.append(i * (n+4))
            col.append(i * (n+4))

            data.append(1)
            row.append(i * (n+4) + (n+3))
            col.append(i * (n+4) + (n+3))

        # Inner Corners
        data.append(1)
        row.append(1)
        col.append(1)
    
        data.append(1)
        row.append(n+2)
        col.append(n+2)

        data.append(1)
        row.append((n+4)*(n+2) + 1)
        col.append((n+4)*(n+2) + 1)
    
        data.append(1)
        row.append((n+4)*(n+2) + n+2)
        col.append((n+4)*(n+2) + n+2)

        # Corners
        data.append(1)
        row.append(0)
        col.append(0)
    
        data.append(1)
        row.append(n+3)
        col.append(n+3)

        data.append(1)
        row.append((n+4)*(n+3))
        col.append((n+4)*(n+3))
    
        data.append(1)
        row.append((n+4)*(n+3) + n+3)
        col.append((n+4)*(n+3) + n+3)


    row, col, data = np.array(row), np.array(col), np.array(data)
    arr = csr_matrix((data, (row, col)), shape=(n2,n2))

    return arr
