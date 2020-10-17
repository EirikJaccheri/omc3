from pathlib import Path

import pytest

from omc3 import madx_wrapper
from omc3.utils.contexts import silence

OMC3_DIR = Path(__file__).parent.parent.parent / "omc3"
LIB = OMC3_DIR / "model" / "madx_macros"
LHC_MACROS = LIB / "lhc.macros.madx"
GENERAL_MACROS = LIB / "general.macros.madx"


@pytest.mark.basic
def test_with_macro(tmp_path):
    """ Checks:
         - Output_file is created.
    """
    content = f"call, file='{LHC_MACROS}';\ncall, file='{GENERAL_MACROS}';\n"
    outfile = tmp_path / "job.with_macro.madx"

    with silence():
        madx_wrapper.run_string(content, output_file=outfile)
    assert outfile.exists()
    out_lines = outfile.read_text()
    assert out_lines == content


@pytest.mark.basic
def test_with_nonexistent_file(tmp_path):
    """ Checks:
         - Madx crashes when tries to call a non-existent file
         - Logfile is created
         - Error message is read from log
    """
    call_file = "does_not_exist.madx"
    content = f"call, file ='{call_file}';"

    log_file = tmp_path / "tmp_log.log"
    with pytest.raises(madx_wrapper.MadxError) as e:
        madx_wrapper.run_string(content, log_file=log_file, cwd=tmp_path)
    assert log_file.is_file()
    assert call_file in str(e.value)
