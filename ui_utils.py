import numpy as np
from scipy import interpolate


def curve_line(p1, p2, amp):
    npts = 100  # number of points to sample
    y = np.array([0, .5, .75, .75, .5, 0])  # describe your shape in 1d like this

    # get the adder. This will be used to raise the z coords
    x = np.arange(y.size)
    xnew = np.linspace(x[0], x[-1], npts)  # sample the x coord
    tck = interpolate.splrep(x, y, s=0)
    adder = interpolate.splev(xnew, tck, der=0) * amp
    adder[0] = adder[-1] = 0
    adder = adder.reshape((-1, 1))

    # get a line between points
    shape3 = np.vstack([np.linspace(p1[dim], p2[dim], npts) for dim in range(3)]).T

    # raise the z coordinate
    shape3[:, -1] = shape3[:, -1] + adder[:, -1]

    x, y, z = (shape3[:, dim] for dim in range(3))
    return x, y, z
