# FEM_Helmholtz.py
from fenics import *
from ufl import SpatialCoordinate, pi, sin, cos
import numpy as np

def run_fem_helmholtz(mesh,
                      coords,
                      eps_val=1.0,
                      k_val=1.0,
                      lb=(0.0, 0.0),
                      ru=(1.0, 1.0)):
    """
    Solve on a rectangular domain Ω = [lb_x,ru_x] × [lb_y,ru_y]:
        eps * Δu + k^2 u = f(x̂, ŷ),   u=0 on ∂Ω
    where (x̂, ŷ) are the *normalized* coordinates mapped to [0,1]^2:
        x̂ = (x - lb_x)/(ru_x - lb_x), same for ŷ.
    This lets you use your original analytical f on [0,1]^2 regardless of mesh size.

    Returns:
      coords     : same (N×2) array you passed in
      u_vals     : FEM solution sampled at those coords (N,)
    """
    lb_x, lb_y = lb
    ru_x, ru_y = ru
    Lx = float(ru_x - lb_x)
    Ly = float(ru_y - lb_y)

    V = FunctionSpace(mesh, 'CG', 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Physical constants
    eps = Constant(eps_val)
    k2  = Constant(k_val**2)

    # Spatial coordinates and normalization to [0,1]^2 for the forcing term
    X = SpatialCoordinate(mesh)
    xhat = (X[0] - lb_x)/Lx
    yhat = (X[1] - lb_y)/Ly

    # Your Eq. (2) forcing defined on [0,1]^2 using (xhat,yhat)
    f = (2*pi*cos(pi*yhat)*sin(pi*xhat)
         + 2*pi*cos(pi*xhat)*sin(pi*yhat)
         + (xhat + yhat)*sin(pi*xhat)*sin(pi*yhat)
         - 2*(pi**2)*(xhat + yhat)*sin(pi*xhat)*sin(pi*yhat))

    # Weak form: ∫ eps ∇u·∇v dx + ∫ k^2 u v dx = ∫ f v dx
    a = eps*dot(grad(u), grad(v))*dx + k2*(u*v)*dx
    L = f*v*dx

    # Homogeneous Dirichlet BC on the whole boundary
    bc = DirichletBC(V, Constant(0.0), 'on_boundary')

    uh = Function(V)
    solve(a == L, uh, bc)

    # Sample solution at requested coordinates
    coords = np.asarray(coords, dtype=float)
    u_vals = np.array([uh(Point(float(xi), float(yi))) for (xi, yi) in coords], dtype=float)

    return coords, u_vals
