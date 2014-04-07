import inspect
from os.path import join, abspath
from dolfin import compile_extension_module, Function, FunctionSpace, assemble,\
    TrialFunction, TestFunction, dx, Matrix, inner

fem_folder = abspath(join(inspect.getfile(inspect.currentframe()), '../fem'))
code = open(join(fem_folder, 'gradient_weight.cpp'), 'r').read()
compiled_module = compile_extension_module(code=code)


def clement_matrix(V):
    '''
    Compute clement interpolation matrix for interpolation between finite
    element spaces V and DG0. The spaces are assumed to be scalar and degrees
    of freedom of V are point evaluations.
    '''

    assert V.ufl_element().value_shape() == ()
    assert V.ufl_element().family() in ['Lagrange', 'Crouzeix-Raviart']

    mesh = V.mesh()

    DG0 = FunctionSpace(mesh, 'DG', 0)
    weights = Function(DG0)

    I = Matrix()
    dg0 = TrialFunction(DG0)
    v = TestFunction(V)
    assemble(inner(dg0, v)*dx, tensor=I)

    compiled_module.compute_DG0_to_CG_weight_matrix(I, weights, False)

    return I
