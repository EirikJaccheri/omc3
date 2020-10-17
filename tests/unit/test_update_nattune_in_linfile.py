from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest
import tfs

from omc3.harpy.constants import COL_NATTUNE, COL_NATAMP, COL_TUNE, COL_AMP
from omc3.scripts.update_nattune_in_linfile import main as update_nattune, PLANES

INPUT_DIR = Path(__file__).parent.parent / "inputs"
RENAME_SUFFIX = '_mytest'


def runclean(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        _clean_output_files()  # comment for debugging single tests
    return wrapper


@pytest.mark.basic
@runclean
def test_all_planes_update():
    update_nattune(
        files=[str(_get_input_file())],
        interval=[0.26, 0.33],  # actual tunes are in that interval
        rename_suffix=RENAME_SUFFIX)
    assert len(list(INPUT_DIR.glob(f'*{RENAME_SUFFIX}*'))) == 2
    for plane in PLANES:
        new = tfs.read(INPUT_DIR / f'spec_test.sdds{RENAME_SUFFIX}.lin{plane.lower()}')
        assert np.allclose(new[f'{COL_NATTUNE}{plane}'], new[f'{COL_TUNE}{plane}'], atol=1e-7)
        assert np.allclose(new[f'{COL_NATAMP}{plane}'], new[f'{COL_AMP}{plane}'], atol=1e-5)


@pytest.mark.basic
@runclean
def test_error_in_interval():
    with pytest.raises(ValueError):
        update_nattune(
            files=[str(_get_input_file())],
            interval=[0., 0.],  # nothing here
            rename_suffix=RENAME_SUFFIX,
        )


@pytest.mark.extended
@runclean
def test_single_plane_update():
    update_nattune(
        files=[str(_get_input_file())],
        interval=[0.26, 0.33],
        rename_suffix=RENAME_SUFFIX,
        planes=["X"],
    )
    assert len(list(INPUT_DIR.glob(f'*{RENAME_SUFFIX}*'))) == 1
    assert (INPUT_DIR / f'spec_test.sdds{RENAME_SUFFIX}.linx').exists()


@pytest.mark.extended
@runclean
def test_keep_not_found():
    update_nattune(
        files=[str(_get_input_file())],
        interval=[0., 0.],  # nothing here
        rename_suffix=RENAME_SUFFIX,
        not_found_action='ignore'
    )
    for plane in PLANES:
        old = tfs.read(INPUT_DIR / f'spec_test.sdds.lin{plane.lower()}')
        new = tfs.read(INPUT_DIR / f'spec_test.sdds{RENAME_SUFFIX}.lin{plane.lower()}')
        assert np.allclose(old[f'{COL_NATTUNE}{plane}'], new[f'{COL_NATTUNE}{plane}'], atol=1e-17)
        assert np.allclose(old[f'{COL_NATAMP}{plane}'], new[f'{COL_NATAMP}{plane}'], atol=1e-17)


@pytest.mark.extended
@runclean
def test_remove_not_found():
    update_nattune(
        files=[str(_get_input_file())],
        interval=[0., 0.],  # nothing here
        rename_suffix=RENAME_SUFFIX,
        not_found_action='remove'
    )
    for plane in PLANES:
        new = tfs.read(INPUT_DIR / f'spec_test.sdds{RENAME_SUFFIX}.lin{plane.lower()}')
        assert len(new.index) == 0


@pytest.mark.extended
@runclean
def test_remove_some_not_found():
    update_nattune(
        files=[str(_get_input_file())],
        interval=[0.2631, 0.265],  # some here
        rename_suffix=RENAME_SUFFIX,
        not_found_action='remove'
    )

    newx = tfs.read(INPUT_DIR / f'spec_test.sdds{RENAME_SUFFIX}.linx')
    assert len(newx.index) == 2  # specific to this test-set

    newy = tfs.read(INPUT_DIR / f'spec_test.sdds{RENAME_SUFFIX}.liny')
    assert len(newy.index) == 3  # specific to this test-set


# Helper -----------------------------------------------------------------------


def _get_input_file() -> Path:
    return INPUT_DIR / "spec_test.sdds"


def _clean_output_files():
    for out_file in INPUT_DIR.glob(f'*{RENAME_SUFFIX}.lin*'):
        with suppress(IOError):
            out_file.unlink()

    for ini_file in Path.cwd().glob('*update_nattune*.ini'):
        with suppress(IOError):
            ini_file.unlink()
