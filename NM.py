# NM.py
from fenics import (
    FunctionSpace, TrialFunction, TestFunction, Function,
    Constant, DirichletBC, solve, dx, dot, grad, Point
)
from ufl import SpatialCoordinate, pi, sin, cos
import numpy as np

def run_fem_helmholtz(mesh,
                      coords=None,
                      eps_val: float = 1.0,
                      k_val: float = 1.0):
    """
    Solve on [0,1]^2 with zero Dirichlet on ∂Ω:
        eps * Δu + k^2 u = f(x,y)
    Weak form:
        ∫ eps ∇u·∇v dx + ∫ k^2 u v dx = ∫ f v dx
    """

    V = FunctionSpace(mesh, 'P', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    eps = Constant(eps_val)
    k2  = Constant(k_val**2)

    # f(x,y) from your Eq. (2)
    x = SpatialCoordinate(mesh)
    f = (2*pi*cos(pi*x[1])*sin(pi*x[0])
         + 2*pi*cos(pi*x[0])*sin(pi*x[1])
         + (x[0] + x[1]) * sin(pi*x[0]) * sin(pi*x[1])
         - 2*(pi**2)*(x[0] + x[1]) * sin(pi*x[0]) * sin(pi*x[1]))

    # Bilinear/linear
    a = eps*dot(grad(u), grad(v))*dx + k2*(u*v)*dx
    L = f*v*dx

    # Homogeneous Dirichlet everywhere
    bc = DirichletBC(V, Constant(0.0), 'on_boundary')

    uh = Function(V)
    solve(a == L, uh, bc)

    # Sample at requested nodes
    if coords is None:
        coords_out = mesh.coordinates()
    else:
        coords_out = np.array(coords, dtype=float)

    u_samples = np.array([uh(Point(float(xi), float(yi))) for (xi, yi) in coords_out], dtype=float)
    return coords_out, u_samples
