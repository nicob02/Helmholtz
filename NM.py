from petsc4py import PETSc
from fenics import (NonlinearProblem, NewtonSolver, FiniteElement,
                    MixedElement, assemble, FunctionSpace, TestFunctions, Function,
                    interpolate, Expression, split, dot, inner, grad, dx, DirichletBC,
                    Constant, exp, ln, derivative, PETScKrylovSolver,
                    PETScFactory, near, PETScOptions, assign, File, plot, SpatialCoordinate)
import numpy as np
import sys
import matplotlib.pyplot as plt
from core.geometry import ElectrodeMesh
from fenics import (
    Point, FunctionSpace, TrialFunction, TestFunction, Function,
    Constant, DirichletBC, solve, dx, dot, grad
)
from ufl import conditional, le, SpatialCoordinate

from fenics import (
    Point, FunctionSpace, TrialFunction, TestFunction, Function,
    Constant, DirichletBC, solve, dx, dot, grad, conditional, le,
    interpolate, SpatialCoordinate
)

from fenics import (
    Point, FunctionSpace, TrialFunction, TestFunction,
    Function, Constant, DirichletBC, solve, dx,
    dot, grad, conditional, le, SpatialCoordinate
)
import numpy as np
def run_fem_helmholtz(mesh,
                      coords=None,
                      eps_val: float = 1.0,
                      k_val: float = 1.0):
    """
    Solve the strong-form Helmholtz with source on [0,1]^2:
        eps * Δu + k^2 u = f(x,y),   u|_{∂Ω} = 0.
    Weak form:
        Find u in H_0^1(Ω) s.t.
          ∫ eps ∇u·∇v dx + ∫ k^2 u v dx = ∫ f v dx

    Parameters
    ----------
    mesh : fenics.Mesh
        Mesh of [0,1]^2 (e.g., UnitSquareMesh(...)).
    coords : np.ndarray or None
        Locations (N,2) where the FEM solution will be sampled (graph.pos).
        If None, returns nodal values at mesh vertices (in the same order as mesh.coordinates()).
    eps_val : float
        Diffusion coefficient ε.
    k_val : float
        Wave number k.

    Returns
    -------
    coords_out : np.ndarray, shape (N,2)
    u_samples : np.ndarray, shape (N,)
    """

    V = FunctionSpace(mesh, 'P', 1)

    # Unknown & test
    u = TrialFunction(V)
    v = TestFunction(V)

    # Constants
    eps = Constant(eps_val)
    k2  = Constant(k_val**2)

    # Source term f(x,y) matching your Eq. (2):
    # f(x,y) = 2πcos(πy) sin(πx) + 2πcos(πx) sin(πy)
    #          + (x+y) sin(πx) sin(πy) - 2π^2 (x+y) sin(πx) sin(πy)
    x = SpatialCoordinate(mesh)
    pi = ufl.pi
    f = (2*pi*ufl.cos(pi*x[1]) * ufl.sin(pi*x[0])
         + 2*pi*ufl.cos(pi*x[0]) * ufl.sin(pi*x[1])
         + (x[0]+x[1])*ufl.sin(pi*x[0])*ufl.sin(pi*x[1])
         - 2*(pi**2)*(x[0]+x[1])*ufl.sin(pi*x[0])*ufl.sin(pi*x[1]))

    # Bilinear and linear forms
    a = eps*dot(grad(u), grad(v))*dx + k2*inner(u, v)*dx
    L = inner(f, v)*dx

    # Homogeneous Dirichlet on the whole boundary (u = 0 on ∂Ω)
    bc = DirichletBC(V, Constant(0.0), 'on_boundary')

    # Solve
    uh = Function(V)
    solve(a == L, uh, bc)

    # Sample solution at requested coordinates
    if coords is None:
        # return values at mesh vertices
        coords_out = mesh.coordinates()
        u_samples = np.array([uh(Point(float(xi), float(yi))) for xi, yi in coords_out], dtype=float)
    else:
        coords_out = np.array(coords, dtype=float)
        u_samples = np.array([uh(Point(float(xi), float(yi))) for (xi, yi) in coords_out], dtype=float)

    return coords_out, u_samples



