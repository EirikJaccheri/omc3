import shutil
from pathlib import Path

from omc3.model.constants import MACROS_DIR, GENERAL_MACROS
from omc3.utils import logging_tools
from omc3.utils.iotools import read_all_lines_in_textfile, create_dirs

LOGGER = logging_tools.get_logger(__name__)


class SegmentCreator(object):
    @classmethod
    def prepare_run(cls, instance, output_path) -> None:
        output_path = Path(output_path)
        macros_path = output_path / MACROS_DIR
        create_dirs(macros_path)
        lib_path = Path(__file__).parent.parent / "madx_macros"
        shutil.copy(str(lib_path / GENERAL_MACROS), str(macros_path / GENERAL_MACROS))

    @classmethod
    def get_madx_script(cls, instance, output_path) -> str:
        output_path = Path(output_path)
        madx_template = read_all_lines_in_textfile(instance.get_file("segment.madx"))
        replace_dict = {
            "MAIN_SEQ": instance.load_main_seq_madx(),  # LHC only
            "OPTICS_PATH": instance.modifiers,  # all
            "NUM_BEAM": instance.beam,  # LHC only
            "PATH": output_path,  # all
            "OUTPUT": output_path,  # Booster only
            "LABEL": instance.label,  # all
            "BETAKIND": instance.kind,  # all
            "STARTFROM": instance.start.name,  # all
            "ENDAT": instance.end.name,  # all
            "RING": instance.ring,  # Booster only
            "KINETICENERGY": instance.energy,  # PS only
            "FILES_DIR": instance.get_dir(),  # Booster and PS
            "NAT_TUNE_X": instance.nat_tunes[0],  # Booster and PS
            "NAT_TUNE_Y": instance.nat_tunes[1],  # Booster and PS
        }
        return madx_template % replace_dict
