__author__ = "Miroslav Kuchta <mirok@math.uio.no>"
__date__ = "2014-04-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

import inspect
from dolfin import VectorFunctionSpace, FunctionSpace, Function, interpolate,\
    compile_extension_module, GenericFunction
from os.path import abspath, join

folder = abspath(join(inspect.getfile(inspect.currentframe()), '../fem'))
code = open(join(folder, 'cr_curl.cpp'), 'r').read()
compiled_cr_module = compile_extension_module(code=code)


def stokes_curl(u, mesh=None):
    '''
    This function uses Stokes theorem to compute curl of vector u inside cell
    by integrating facet terms over facets of the cell. In 2d, the facet term
    is dot(u, dot(R.T, n)), where R is a rotation matrix R=((0, 1), (-1, 0)),
    and the resulting curl(u) is a scalar. In 3d, the facet term is
    cross(n, u) and the result is a vector. Integration method leads to curl
    being exact of linear u. Curl is always represented in DG0 function space.
    '''

    # Require u to be GenericFunction
    assert isinstance(u, GenericFunction)

    # Require u to be vector
    rank = u.value_rank()
    assert rank == 1

    # For now, there is no support for manifolds
    if mesh is None:
        _mesh = u.function_space().mesh()
    else:
        _mesh = mesh
    tdim = _mesh.topology().dim()
    gdim = _mesh.geometry().dim()
    assert tdim == gdim

    for i in range(rank):
        assert u.value_dimension(i) == gdim

    # Undefined for 1D
    assert gdim > 1

    CR = VectorFunctionSpace(_mesh, 'CR', 1)
    # Based of gdim choose proper space u curl
    if gdim == 2:
        DG = FunctionSpace(_mesh, 'DG', 0)
    else:
        DG = VectorFunctionSpace(_mesh, 'DG', 0)

    curlu = Function(DG)
    _u = interpolate(u, CR)
    _u.update()

    # Compute the curl
    compiled_cr_module.cr_curl(curlu, _u)

    return curlu
