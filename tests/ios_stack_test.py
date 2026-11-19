import math
import sys

import pytest


def test_it_runs_on_ios():
    assert sys.platform == "ios"


def test_netgen_meshes_a_square():
    from netgen.geom2d import unit_square
    assert len(unit_square.GenerateMesh(maxh=0.1).Elements2D()) > 100


def test_ngsolve_runs_threads():
    import ngsolve as ngs
    with ngs.TaskManager():
        assert ngs.GetNumThreads() > 1


def test_ngsolve_solves_poisson():
    import ngsolve as ngs
    from netgen.geom2d import unit_square
    exact = ngs.sin(math.pi * ngs.x) * ngs.sin(math.pi * ngs.y)
    with ngs.TaskManager():
        mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.05))
        fes = ngs.H1(mesh, order=3, dirichlet=".*")
        u, v = fes.TnT()
        a = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx).Assemble()
        f = ngs.LinearForm(2 * math.pi ** 2 * exact * v * ngs.dx).Assemble()
        gfu = ngs.GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
        error = math.sqrt(ngs.Integrate((gfu - exact) ** 2, mesh))
    assert error < 1e-4


def test_planex_meshes_through_netgen():
    import ngsolve as ngs
    import planex as px
    plate = px.QuickSketch().Rectangle(40, 20).Face() - px.Circle((10, 10), 4).Face()
    mesh = ngs.Mesh(px.OCCGeometry(plate).GenerateMesh(maxh=2))
    mesh.Curve(3)
    assert ngs.Integrate(1, mesh) == pytest.approx(800 - 16 * math.pi, rel=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
