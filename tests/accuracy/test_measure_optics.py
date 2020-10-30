import itertools
from pathlib import Path
from shutil import rmtree
from typing import Union

import pytest
import tfs

from omc3.hole_in_one import _optics_entrypoint  # <- Protected member of module. Make public?
from omc3.model import manager
from omc3.optics_measurements import measure_optics
from omc3.utils import stats
from omc3.utils.contexts import timeit
from tests.accuracy.twiss_to_lin import optics_measurement_test_files

LIMITS = {"P": 1e-4, "B": 3e-3, "D": 1e-2, "A": 6e-3}
DEFAULT_LIMIT = 5e-3
BASE_PATH = Path(__file__).parent.parent / "results"
MODEL_DIR = Path(__file__).parent.parent / "inputs" / "models"


def _drop_item(idx_drop: int, lst: list) -> list:
    """ Returns new list where idx_drop in old list is no longer present."""
    return [item for idx, item in enumerate(lst) if idx != idx_drop]


def _create_input(motion, beam=1):
    dpps = [0, 0, 0, -4e-4, -4e-4, 4e-4, 4e-4, 5e-5, -3e-5, -2e-5]
    print(f"\nInput creation: {dpps}")
    opt_dict = dict(accel="lhc", year="2018", ats=True, beam=beam, files=[""],
                    model_dir=str(MODEL_DIR / f"25cm_beam{beam}"),
                    outputdir=str(BASE_PATH))
    optics_opt, rest = _optics_entrypoint(opt_dict)
    optics_opt.accelerator = manager.get_accelerator(rest)
    lins = optics_measurement_test_files(opt_dict["model_dir"], dpps, motion,
                                         beam_direction=(1 if beam == 1 else -1))
    return lins, optics_opt


MEASURE_OPTICS_SETTINGS = {
    "compensation": ["model", "equation", "none"],
    "coupling_method": [2],
    "range_of_bpm": [11],
    "three_bpm_method": [False],
    "second_order_disp": [False],
}
MEASURE_OPTICS_INPUT = list(itertools.product(*MEASURE_OPTICS_SETTINGS.values()))

PRE_CREATED_INPUT = dict(free=_create_input(motion="free"), driven=_create_input(motion="driven"))
INPUT_OPPOSITE_DIRECTION = _create_input(motion="driven", beam=2)


@pytest.mark.basic
def test_single_file():
    test_single_file(*MEASURE_OPTICS_INPUT[0])


@pytest.mark.basic
def test_3_onmom_files():
    test_3_onmom_files(*MEASURE_OPTICS_INPUT[1])


@pytest.mark.extended
@pytest.mark.parametrize(
    "compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp",
    _drop_item(0, MEASURE_OPTICS_INPUT)
)
def test_single_file(compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    _test_prototype(slice(0, 1),
                    outputdir=str(BASE_PATH / "single"),
                    compensation=compensation,
                    coupling_method=coupling_method,
                    range_of_bpms=range_of_bpms,
                    three_bpm_method=three_bpm_method,
                    second_order_disp=second_order_disp,
                    )


@pytest.mark.extended
@pytest.mark.parametrize(
    "compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp",
    _drop_item(1, MEASURE_OPTICS_INPUT)
)
def test_3_onmom_files(compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    _test_prototype(slice(None, 3),
                    outputdir=str(BASE_PATH / "onmom"),
                    compensation=compensation,
                    coupling_method=coupling_method,
                    range_of_bpms=range_of_bpms,
                    three_bpm_method=three_bpm_method,
                    second_order_disp=second_order_disp,
                    )


@pytest.mark.extended
@pytest.mark.parametrize(
    "compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp",
    MEASURE_OPTICS_INPUT
)
def test_3_pseudo_onmom_files(compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    _test_prototype(slice(-3, None),
                    outputdir=str(BASE_PATH / "pseudo_onmom"),
                    compensation=compensation,
                    coupling_method=coupling_method,
                    range_of_bpms=range_of_bpms,
                    three_bpm_method=three_bpm_method,
                    second_order_disp=second_order_disp,
                    )


@pytest.mark.extended
@pytest.mark.parametrize(
    "compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp",
    MEASURE_OPTICS_INPUT
)
def test_offmom_files(compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    _test_prototype(slice(None, 7),
                    chromatic_beating=True,
                    outputdir=str(BASE_PATH / "offmom"),
                    compensation=compensation,
                    coupling_method=coupling_method,
                    range_of_bpms=range_of_bpms,
                    three_bpm_method=three_bpm_method,
                    second_order_disp=second_order_disp,
                    )


# ----- Helpers ----- #


def _test_prototype(lin_slice, **kwargs) -> None:
    lins, optics_opt = PRE_CREATED_INPUT["free" if kwargs['compensation'] == 'none' else "driven"]
    optics_opt.update(kwargs)
    inputs = measure_optics.InputFiles(lins[lin_slice], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt, kwargs.get('limits', LIMITS))


def _run_evaluate_and_clean_up(inputs, optics_opt, limits) -> None:
    with timeit(lambda spanned: print(f"\nTotal time for optics measurements: {spanned}")):
        measure_optics.measure_optics(inputs, optics_opt)
    evaluate_accuracy(optics_opt.outputdir, limits)
    _clean_up(optics_opt.outputdir)


def evaluate_accuracy(meas_path: Union[Path, str], limits: dict) -> None:
    for filepath in [f for f in Path(meas_path).iterdir() if (f.is_file() and ".tfs" in str(f))]:
        a = tfs.read(filepath)
        cols = [column for column in a.columns.to_numpy() if column.startswith('DELTA')]
        if str(filepath) == "normalised_dispersion_x.tfs":
            cols.remove("DELTADX")
        for col in cols:
            rms = stats.weighted_rms(a.loc[:, col].to_numpy(), errors=a.loc[:, f"ERR{col}"].to_numpy())
            if col[5] in limits.keys():
                assert rms < limits[col[5]], "\nFile: {:25}  Column: {:15}   RMS: {:.6f}".format(
                    str(filepath), col, rms)
            else:
                assert rms < DEFAULT_LIMIT, "\nFile: {:25}  Column: {:15}   RMS: {:.6f}".format(
                    str(filepath), col, rms)
            print(f"\nFile: {str(filepath):25}  Column: {col[5:]:15}   RMS:    {rms:.6f}")


def _clean_up(path_dir: Union[Path, str]) -> None:
    if Path(path_dir).is_dir():
        rmtree(path_dir, ignore_errors=True)
