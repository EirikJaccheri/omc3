from pathlib import Path
from shutil import rmtree

import pytest

from omc3.model.constants import MODIFIERS_MADX
from omc3.model_creator import create_instance_and_model
from omc3.utils import iotools

OMC3_DIR = Path(__file__).parent.parent.parent / "omc3"
INPUT_DIR = Path(__file__).parent.parent / "inputs"

BASE_OUTPUT = INPUT_DIR.parent / "model"
COMP_MODEL = INPUT_DIR / "models" / "25cm_beam1"
PS_MODEL = OMC3_DIR / "model" / "accelerators" / "ps" / "2018" / "strength"


def _create_input_lhc():
    iotools.create_dirs(BASE_OUTPUT)
    iotools.copy_item(COMP_MODEL / "opticsfile.24_ctpps2", BASE_OUTPUT/ "strengths.madx")
    iotools.write_string_into_new_file(BASE_OUTPUT / MODIFIERS_MADX,
                                       f"call, file='{BASE_OUTPUT / 'strengths.madx'}';\n")
    iotools.write_string_into_new_file(BASE_OUTPUT / "corrections.madx", "\n")
    iotools.write_string_into_new_file(BASE_OUTPUT / "extracted_mqts.str", "\n")


def _create_input_ps():
    iotools.create_dirs(BASE_OUTPUT)
    iotools.copy_item(PS_MODEL / "elements.str", BASE_OUTPUT / "elements.str")
    iotools.copy_item(PS_MODEL / "PS_LE_LHC_low_chroma.str", BASE_OUTPUT / "strengths.madx")
    iotools.write_string_into_new_file(BASE_OUTPUT / MODIFIERS_MADX,
                                       f"call, file='{BASE_OUTPUT / 'elements.str'}';\n"
                                       f"call, file='{BASE_OUTPUT / 'strengths.madx'}';\n")


@pytest.mark.basic
def test_booster_creation_nominal():
    iotools.create_dirs(BASE_OUTPUT)
    iotools.write_string_into_new_file(BASE_OUTPUT / MODIFIERS_MADX, "\n")
    opt_dict = dict(type="nominal", accel="psbooster", ring=1,
                    nat_tunes=[4.21, 4.27], drv_tunes=[0.205, 0.274], driven_excitation="acd",
                    dpp=0.0, energy=0.16, modifiers=str(BASE_OUTPUT / MODIFIERS_MADX),
                    fullresponse=True, outputdir=str(BASE_OUTPUT),
                    writeto=str(BASE_OUTPUT / "job.twiss.madx"),
                    logfile=str(BASE_OUTPUT / "madx_log.txt"))
    create_instance_and_model(opt_dict)
    _clean_up(BASE_OUTPUT)


@pytest.mark.basic
def test_ps_creation_nominal():
    _create_input_ps()
    opt_dict = dict(type="nominal", accel="ps", nat_tunes=[6.32, 6.29], drv_tunes=[0.325, 0.284],
                    driven_excitation="acd",
                    dpp=0.0, energy=1.4, modifiers=str(BASE_OUTPUT / MODIFIERS_MADX),
                    fullresponse=True, outputdir=str(BASE_OUTPUT),
                    writeto=str(BASE_OUTPUT / "job.twiss.madx"),
                    logfile=str(BASE_OUTPUT / "madx_log.txt"))
    create_instance_and_model(opt_dict)
    _clean_up(BASE_OUTPUT)


@pytest.mark.basic
def test_lhc_creation_nominal():
    _create_input_lhc()
    opt_dict = dict(type="nominal", accel="lhc", year="2018", ats=True, beam=1,
                    nat_tunes=[0.31, 0.32], drv_tunes=[0.298, 0.335], driven_excitation="acd",
                    dpp=0.0, energy=6.5, modifiers=str(BASE_OUTPUT / MODIFIERS_MADX),
                    fullresponse=True, outputdir=str(BASE_OUTPUT),
                    writeto=str(BASE_OUTPUT / "job.twiss.madx"),
                    logfile=str(BASE_OUTPUT / "madx_log.txt"))
    create_instance_and_model(opt_dict)
    _clean_up(BASE_OUTPUT)


@pytest.mark.basic
def test_lhc_creation_best_knowledge():
    _create_input_lhc()
    opt_dict = dict(type="best_knowledge", accel="lhc", year="2018", ats=True, beam=1,
                    nat_tunes=[0.31, 0.32], dpp=0.0, energy=6.5,
                    modifiers=str(BASE_OUTPUT / MODIFIERS_MADX), outputdir=str(BASE_OUTPUT),
                    writeto=str(BASE_OUTPUT / "job.twiss_best_knowledge.madx"),
                    logfile=str(BASE_OUTPUT / "madx_log_best_knowledge.txt"))
    create_instance_and_model(opt_dict)
    _clean_up(BASE_OUTPUT)


def _clean_up(path_dir):
    if Path(path_dir).is_dir():
        rmtree(path_dir, ignore_errors=True)
