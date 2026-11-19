import math
import sys

import pytest

import planex as px


def test_it_runs_on_ios():
    print(sys.platform, sys.version, px.__version__)
    assert sys.platform == "ios"


def test_a_plate_with_a_hole_has_its_exact_area():
    plate = px.QuickSketch().Rectangle(40, 20).Face()
    hole = px.Circle((10, 10), 4).Face()
    assert (plate - hole).mass == pytest.approx(800 - 16 * math.pi, rel=1e-12)


def test_a_constraint_sketch_solves():
    sk = px.Sketch((0, 0))
    sk.Line(10, 0).Line(0, 5, px.Perpendicular).Line(-10, 0, px.Perpendicular).Close(px.Perpendicular)
    assert sk.Face().mass == pytest.approx(50, rel=1e-12)
