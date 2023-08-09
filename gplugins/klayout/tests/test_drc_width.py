from __future__ import annotations

import gdsfactory as gf

from gplugins.klayout.drc.check_width import check_width


def test_wmin_failing(layer: tuple[int, int] = (1, 0)) -> None:
    w = 50
    min_width = 50 + 10  # component edges are smaller than min_width
    c = gf.components.rectangle(size=(w, w), layer=layer)
    gdspath = c.write_gds()

    # r = check_width(gdspath, min_width=min_width, layer=layer)
    # print(check_width(gdspath, min_width=min_width, layer=layer))
    assert check_width(gdspath, min_width=min_width, layer=layer) == 2
    assert check_width(c, min_width=min_width, layer=layer) == 2


def test_wmin_passing(layer: tuple[int, int] = (1, 0)) -> None:
    w = 50
    min_width = 50 - 10  # component edges are bigger than the min_width
    c = gf.components.rectangle(size=(w, w), layer=layer)
    gdspath = c.write_gds()

    # print(check_width(c, min_width=min_width, layer=layer))
    # assert check_width(gdspath, min_width=min_width, layer=layer) is None
    # assert check_width(c, min_width=min_width, layer=layer) is None
    assert check_width(gdspath, min_width=min_width, layer=layer) == 0
    assert check_width(c, min_width=min_width, layer=layer) == 0


if __name__ == "__main__":
    # test_wmin_failing()
    test_wmin_passing()
