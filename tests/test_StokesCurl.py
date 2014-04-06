import nose

from dolfin import VectorFunctionSpace, UnitSquareMesh, UnitCubeMesh,\
    interpolate, Expression

from fenicstools import *


def test_StokesCurl():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = interpolate(Expression(('-x[1]', 'x[0]')), V)
    curlu = stokes_curl(u)
    CURLU = curlu.vector().array()
    point_0 = all(abs(CURLU - 2.) < 1E-13)
    nose.tools.assert_equal(point_0, True)

    mesh = UnitCubeMesh(4, 4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = interpolate(Expression(('x[2]-x[1]', 'x[0]-x[2]', 'x[1]-x[0]')), V)
    curlu = stokes_curl(u)
    CURLU = curlu.vector().array()
    point_0 = all(abs(CURLU - 2.) < 1E-13)
    nose.tools.assert_equal(point_0, True)
