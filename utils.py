'''
    Custom Replay buffer for MBPO
'''
# pylint: disable=R0902, R0913
import numpy as np

#zgl
import matplotlib.pyplot as plt
from scipy import sparse
import osqp

# plot cricle
def obj(u, ustar):
    y = np.linalg.norm(u - ustar)
    return y


def circle(x, y, r):
    th = np.linspace(0, 2 * np.pi, 100)
    xunit = r * np.cos(th) + x
    yunit = r * np.sin(th) + y
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x, y, color="darkred", linewidth=2)

def dphidx(s):
    x0 = 0.5
    y0 = 1.5
    x = s[0]
    y = s[1]
    theta = s[2]
    first_chain = -2 * ((x0-x)*np.sin(theta) - (y0-y)*np.cos(theta))
    dpds = [first_chain*(-np.sin(theta)),
            first_chain*np.cos(theta),
            first_chain*((x0-x)*np.cos(theta) + (y0-y)*np.sin(theta))]
    return dpds

def gx(s):
    # dot s = g(s) * u
    theta = s[2]
    g = [np.cos(theta), 0,
         np.sin(theta), 0,
         0, 1]
    return g


def quadprog(H, f, A=None, b=None,
             initvals=None, verbose=True):
    qp_P = sparse.csc_matrix(H)
    qp_f = np.array(f)
    qp_l = -np.inf * np.ones(len(b))
    qp_A = sparse.csc_matrix(A)
    qp_u = np.array(b)
    model = osqp.OSQP()
    model.setup(P=qp_P, q=qp_f,
                A=qp_A, l=qp_l, u=qp_u, verbose=verbose)
    if initvals is not None:
        model.warm_start(x=initvals)
    results = model.solve()
    return results.x, results.info.status