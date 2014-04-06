'''
Run a convergence test with curl computed by Stokes theorem in 3d.
'''

from dolfin import *
from StokesCurl import stokes_curl
import numpy

u_exact = Expression(('sin(pi*x[0]*x[0]*x[1]*x[2])',
                      'sin(pi*x[0]*x[1]*x[1]*x[2])',
                      'sin(pi*x[0]*x[1]*x[2]*x[2])'))

curl_code = \
    ('cos(pi*x[0]*x[1]*x[2]*x[2])*pi*x[0]*x[2]*x[2]\
        - cos(pi*x[0]*x[1]*x[1]*x[2])*pi*x[0]*x[1]*x[1]',
     'cos(pi*x[0]*x[0]*x[1]*x[2])*pi*x[0]*x[0]*x[1]\
        - cos(pi*x[0]*x[1]*x[2]*x[2])*pi*x[1]*x[2]*x[2]',
     'cos(pi*x[0]*x[1]*x[1]*x[2])*pi*x[1]*x[1]*x[2]\
        - cos(pi*x[0]*x[0]*x[1]*x[2])*pi*x[0]*x[0]*x[2]')
curlu_exact = Expression(curl_code)


def demo_3d_curl(mesh):
    # Compute the curl with Stokes theorem
    cr_curlu = stokes_curl(u_exact, mesh=mesh)

    # Represent the exact curl on DG
    DG = VectorFunctionSpace(mesh, 'DG', 0)
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

# Create meshes
meshes = [UnitCubeMesh(N, N, N) for N in [4, 8, 12, 16, 24, 32]]

print '     h    |   rate_Loo  |  rate_L2 '
mesh = meshes[0]
h_, eLoo_, eL2_ = demo_3d_curl(mesh)
for mesh in meshes[1:]:
    h, eLoo, eL2 = demo_3d_curl(mesh)
    rate_Loo = numpy.log(eLoo/eLoo_)/numpy.log(h/h_)
    rate_L2 = numpy.log(eL2/eL2_)/numpy.log(h/h_)
    h_, eLoo_, eL2_ = h, eLoo, eL2
    print '  %.4f  |     %.2f    |   %.2f   ' % (h_, rate_Loo, rate_L2)
print
