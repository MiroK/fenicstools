from dolfin import *
from fenicstools import divergence_matrix, clement_matrix
import numpy

# Exact solution
u_exact = Expression(('sin(pi*x[0])', 'cos(2*pi*x[1])'))
divu_exact = Expression('pi*(cos(pi*x[0]) - 2*sin(2*pi*x[1]))')


def demo_clement_divergence(mesh):
    CG1_vector = VectorFunctionSpace(mesh, 'CG', 1)
    u = interpolate(u_exact, CG1_vector)

    DG0 = FunctionSpace(mesh, 'DG', 0)
    divu_DG0 = Function(DG0)
    DIV_M = divergence_matrix(u)
    divu_DG0.vector()[:] = DIV_M*u.vector()

    # plot(divu_DG0)
    # plot(divu_exact, mesh)

    CG1 = FunctionSpace(mesh, 'CG', 1)
    C = clement_matrix(CG1)
    divu_CG1 = Function(CG1)
    divu_CG1.vector()[:] = C*divu_DG0.vector()

    # plot(divu_CG1)
    # interactive()

    h = mesh.hmin()
    error = errornorm(divu_exact, divu_CG1)

    return h, error

# -----------------------------------------------------------------------------

# Create Meshes
Ns = [20, 40, 60, 80, 160, 240]
meshes = [UnitSquareMesh(N, N) for N in Ns]

# Run convergence test
h_, e_ = demo_clement_divergence(meshes[0])
for mesh in meshes[1:]:
    h, e = demo_clement_divergence(mesh)
    rate = numpy.log(e/e_)/numpy.log(h/h_)
    h_, e_ = h, e
    print h_, e_, rate
