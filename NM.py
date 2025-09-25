import torch
from core.geometry import ElectrodeMesh
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from fenics import *
import numpy as np

def run_fem_helmholtz(mesh,
                      coords,
                      eps_val=1.0,
                      k_val=1.0,
                      lb=(0.0, 0.0),
                      ru=(1.0, 1.0)):
    """
    Solve on Ω=[lb_x,ru_x]×[lb_y,ru_y]:
        ε Δu + k^2 u = f(x̂,ŷ),  u=0 on ∂Ω,
    where (x̂,ŷ) are normalized to [0,1]^2 to match your Eq.(2) forcing.
    Returns (coords, u_vals) sampled at 'coords'.
    """
    # 1) ensure we have a dolfin Mesh (same pattern as your electro-thermal FEM)
    fenics_mesh = mesh.mesh if hasattr(mesh, "mesh") else mesh
    assert isinstance(fenics_mesh, Mesh), (
        "run_fem_helmholtz: 'mesh' must be a dolfin Mesh or an object with .mesh (dolfin Mesh)."
    )

    # 2) function space and variational forms
    V = FunctionSpace(fenics_mesh, 'CG', 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    eps = Constant(eps_val)
    k2  = Constant(k_val**2)

    # normalize coordinates to [0,1]^2 for the forcing
    lb_x, lb_y = float(lb[0]), float(lb[1])
    ru_x, ru_y = float(ru[0]), float(ru[1])
    Lx = ru_x - lb_x
    Ly = ru_y - lb_y

    X    = SpatialCoordinate(fenics_mesh)
    xhat = (X[0] - lb_x) / Lx
    yhat = (X[1] - lb_y) / Ly

    f = ( 2*ufl.pi*ufl.cos(ufl.pi*yhat)*ufl.sin(ufl.pi*xhat)
        + 2*ufl.pi*ufl.cos(ufl.pi*xhat)*ufl.sin(ufl.pi*yhat)
        + (xhat + yhat)*ufl.sin(ufl.pi*xhat)*ufl.sin(ufl.pi*yhat)
        - 2*(ufl.pi**2)*(xhat + yhat)*ufl.sin(ufl.pi*xhat)*ufl.sin(ufl.pi*yhat) )

    a = eps*dot(grad(u), grad(v))*dx + k2*u*v*dx
    L = f*v*dx

    bc = DirichletBC(V, Constant(0.0), 'on_boundary')

    uh = Function(V)
    solve(a == L, uh, bc)

    # 3) sample at requested coordinates
    coords = np.asarray(coords, dtype=np.float64)
    u_vals = np.array([uh(Point(float(xi), float(yi))) for (xi, yi) in coords], dtype=np.float64)

    return coords, u_vals


