'''
Run a convergence test with curl computed by Stokes theorem. Perform the
test on variety of meshes in 2d to see how the results depend on the mesh
quality.
'''

from dolfin import *
from fenicstools import stokes_curl
import numpy

u_exact = Expression(('sin(pi*x[0]*x[1])', 'cos(pi*x[0]*x[1])'))
curlu_exact = Expression('pi*(-sin(pi*x[0]*x[1])*x[1]-cos(pi*x[0]*x[1])*x[0])')


def demo_2d_curl(mesh):
    # Compute the curl with Stokes theorem
    cr_curlu = stokes_curl(u_exact, mesh=mesh)

    # Represent the exact curl on DG
    DG = FunctionSpace(mesh, 'DG', 0)
    curlu = interpolate(curlu_exact, DG)

    # Compute the l^oo and l^1 norms
    curlu.vector().axpy(-1, cr_curlu.vector())
    error_Loo = curlu.vector().norm('linf')/DG.dim()
    error_L2 = curlu.vector().norm('l2')/DG.dim()
    h = mesh.hmin()

    # Create global measure of h and errors
    comm = mpi_comm_world()
    h = MPI.min(comm, mesh.hmin())
    error_Loo = MPI.max(comm, error_Loo)
    error_L2 = MPI.max(comm, error_L2)

    return h, error_Loo, error_L2

# -----------------------------------------------------------------------------

# Create various meshes
rectangle0 = Rectangle(-1, -1, 1, 1)
rectangle1 = Rectangle(-0.25, -0.25, 0.25, 0.25)
circle = Circle(0, 0, 0.25)

domain0 = rectangle0
domain1 = rectangle0 - rectangle1
domain2 = rectangle0 - circle

domains = [domain0, domain1, domain2]
names = ['Rectangle', 'RectangleHole', 'CircleHole']
Ns = [20, 40, 60, 80, 160, 240]
mesh_types = {name: [Mesh(domain, N) for N in Ns]
              for name, domain in zip(names, domains)}
mesh_types['UnitSquareMesh'] = [UnitSquareMesh(N, N) for N in Ns]

for mesh_type in mesh_types:
    print mesh_type
    meshes = mesh_types[mesh_type]
    print '     h    |   rate_Loo  |  rate_L2 '
    mesh = meshes[0]
    h_, eLoo_, eL2_ = demo_2d_curl(mesh)
    for mesh in meshes[1:]:
        h, eLoo, eL2 = demo_2d_curl(mesh)
        rate_Loo = numpy.log(eLoo/eLoo_)/numpy.log(h/h_)
        rate_L2 = numpy.log(eL2/eL2_)/numpy.log(h/h_)
        h_, eLoo_, eL2_ = h, eLoo, eL2
        print '  %.4f  |     %.2f    |   %.2f   ' % (h_, rate_Loo, rate_L2)
    print
