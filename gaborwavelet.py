import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as patches

def gaborWavelet(variance_square, xi, x, m):
    return gaussWindow(1/(2*variance_square), (x-m)/variance_square) * np.exp(-x*1j*xi)


def gaussWindow(alpha, x):
    return np.sqrt(alpha/np.pi) * np.exp(-alpha*np.power(x,2))


def gaborKernel(variance_square, xi, yi, size = 7):
    mean = int(size/2)
    x = np.arange(0, size)
    gw_x = gaborWavelet(size*variance_square, xi/size, x, mean)
    gw_y = gaborWavelet(size*variance_square, yi/size, x, mean)
    return np.outer(gw_x,gw_y)


def horizontalGaborKernel(variance_square, xi, size = 7):
    mean = int(size/2)
    x = np.arange(0, size)
    gw = gaborWavelet(size*variance_square, xi/size, x, mean)
    window = gaussWindow(1/(2*variance_square*size), (x-mean)/(variance_square*size))

    return np.outer(gw, window)
    

def getFlips(kernel):
    return [kernel, np.flip(kernel, 0), np.flip(kernel, 1), np.flip(np.flip(kernel, 0), 1)]


def getBasicKernels(size : int = 7):
    number_of_kernels = 32
    stddev = np.sqrt(2/number_of_kernels)
    gk = normalizeKernel(np.imag(gaborKernel(0.25, 2*np.pi, 2*np.pi, size)), -1,1) * stddev
    gk_x = normalizeKernel(np.imag(gaborKernel(0.25, 4*np.pi, 2*np.pi, size)), -1,1) * stddev
    gk_y = normalizeKernel(np.imag(gaborKernel(0.25, 2*np.pi, 4*np.pi, size)), -1,1) * stddev
    ck = normalizeKernel(circularKernel(1.5, size), -1,1,) * stddev
    hg = normalizeKernel(np.imag(horizontalGaborKernel(0.25, 2*np.pi, size)), -1,1) * stddev
    
    normal_kernels = []
    # I just want to have kernel number which is a power of 2
    number_of_normal_kernels = 14
    for i in range(number_of_normal_kernels):
        normal_kernels.append(np.random.rand(size,size))

    return np.concatenate([getFlips(gk), getFlips(gk_x), getFlips(gk_y), [ck], [-ck], [hg], 
            [hg.transpose()], [np.flip(hg,0)], [np.flip(hg.transpose(), 1)], normal_kernels], axis=0)


def orthonormalInit(kernel_shape : np.ndarray):
    """
        This function returns an initial weight matrix for a layer. The result will be orthonormal 
        vectors (kernels) as described at 
        https://hjweide.github.io/orthogonal-initialization-in-convolutional-layers

        The distribution will be sqrt(2/d4) as in the Glorot initialization method.

        Input:
            kernel_shape: expected to be [d1, d2, d3, d4], where kernel size is d1*d2, d3 is the input
                channel size (ignored here), and d4 is the number of neurons (kernels) inside the layer.
    """
    random_matrix = np.random.rand(kernel_shape[3], kernel_shape[0] * kernel_shape[1] * kernel_shape[2])


    u,_,vt = np.linalg.svd(random_matrix, full_matrices=False)
    print("svd finished, vt shape: %s, u shape: %s" % (str(np.shape(vt)), str(np.shape(u))))
    if np.shape(random_matrix)[0] > np.shape(random_matrix)[1]:
        w = np.reshape(u, [kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]])
    else:
        w = np.reshape(vt, [kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]])


    # the 0.9 multiplier is there because of using leaky relu, so variance for x < 0
    # is not 0, but 0.1 * variance(x>0)
    stdev = np.sqrt((0.9*2)/kernel_shape[2])
    w *= stdev

    return w.astype(np.float32)




def circularKernel(r, size = 7):
    res = np.ndarray([size, size])
    mean = int(size/2)
    for i in range(size):
        for j in range(size):
            x = - ( np.sqrt( np.power(i-mean, 2) + np.power(j-mean, 2) ) - r )
            res.itemset((i,j), x)
    return res


def plot2DWavelet(xi, R, norming = False):
    gw = gaborWavelet(1.5,4,R,np.pi/2, np.pi)
    surf = np.outer(gw, gw)

    if norming:
        surf = surf / np.linalg.norm(surf)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    RR1, RR2 = np.meshgrid(R, R)

    plotsurf = ax.plot_surface(RR1,RR2,np.imag(surf), cmap=cm.coolwarm)

    ax.set_zlim(-1, 1)

    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(plotsurf, shrink=0.5, aspect=5)

    plt.show()


def plot2DKernels(kernels, square_width = 1.0, square_height = 1.0, normalize = False):
    numOfKernels = np.shape(kernels)[0]
    numOfRows = int(np.ceil(np.sqrt(numOfKernels)))
    numOfCols = int(np.ceil(numOfKernels / numOfRows))

    f, axs = plt.subplots(numOfCols, numOfRows, figsize=(numOfRows, numOfCols))

    # axs is a 2d array
    for ax1 in axs:
        for ax2 in ax1:
            ax2.axis('off')

    for i in range(numOfCols):
        for j in range(numOfRows):
            index = i*numOfRows + j

            if index < numOfKernels:
                plot2DKernel(axs[i,j], kernels[index], normalize = normalize)
            else:
                break


def plotWavelet(xi, R):
    gw = gaborWavelet(1.5,4,R,np.pi/2, np.pi)
    plt.plot(R, gw)
    plt.show()


def normalizeKernel(kernel, a, b):
    """
        It will squish kernel's value into the [a,b] interval. Kernel must have variance > 0.
    """
    return ( (b-a) * ( kernel - np.min(kernel) ) /  (np.max(kernel) - np.min(kernel) ) ) + a


def plot2DKernel(ax, kernel, square_width = 1.0, square_height = 1.0, normalize = False):
    """
        Kernel is assumed to be 2D. If normalize is False, the kernel must have 
        numbers between -1 and 1.
    """
    width,height = np.shape(kernel)
    if normalize:
        kernel = normalizeKernel(kernel, -1, 1)

    for i in range(0, width):
        for j in range(0, height):
            color = ( kernel[i,j] + 1 ) / 2
            ax.add_patch(patches.Rectangle((i,j), square_width, square_height, color = str(color)))

    ax.set_xlim(0, width)
    ax.set_ylim(0,height)
