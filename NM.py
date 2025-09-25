import torch
from core.geometry import ElectrodeMesh
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from fenics import *
import numpy as np
import ufl
# NM.py
from fenics import *
import ufl
import numpy as np

def run_fem_helmholtz(mesh,
                      coords,
                      eps_val=1.0,
                      k_val=1.0,
                      lb = (0, 0),   
                      ru = (1.0, 1.0):
    """
    Solve on Ω (assumed to be [0,1]×[0,1]):
        ε Δu + k^2 u = f(x,y),   u=0 on ∂Ω,
    using the forcing from your paper (Eq. 2) directly in physical coordinates.
    Returns (coords, u_vals) sampled at 'coords'.
    """
    # unwrap to a dolfin Mesh if needed
    fenics_mesh = mesh.mesh if hasattr(mesh, "mesh") else mesh
    assert isinstance(fenics_mesh, Mesh), "Provide a dolfin Mesh or an object with .mesh"

    V  = FunctionSpace(fenics_mesh, 'CG', 2)
    u  = TrialFunction(V)
    v  = TestFunction(V)

    eps = Constant(eps_val)
    k2  = Constant(k_val**2)

    # physical coordinates (NO normalization)
    X = SpatialCoordinate(fenics_mesh)
    x, y = X[0], X[1]

    # f(x,y) as given (Eq. 2)
    f = ( 2*ufl.pi*ufl.cos(ufl.pi*y)*ufl.sin(ufl.pi*x)
        + 2*ufl.pi*ufl.cos(ufl.pi*x)*ufl.sin(ufl.pi*y)
        + (x + y)*ufl.sin(ufl.pi*x)*ufl.sin(ufl.pi*y)
        - 2*(ufl.pi**2)*(x + y)*ufl.sin(ufl.pi*x)*ufl.sin(ufl.pi*y) )

    a = eps*dot(grad(u), grad(v))*dx + k2*u*v*dx
    L = f*v*dx

    bc = DirichletBC(V, Constant(0.0), 'on_boundary')

    uh = Function(V)
    solve(a == L, uh, bc)

    coords = np.asarray(coords, dtype=np.float64)
    u_vals = np.array([uh(Point(float(xi), float(yi))) for (xi, yi) in coords], dtype=np.float64)
    return coords, u_vals







