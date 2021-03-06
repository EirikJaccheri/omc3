"""
PS Booster
----------

Accelerator-Class for the ``PSB`` machine.


Model Creation Keyword Args:
    *--Optional--*

    - **dpp** *(float)*:

        Deltap/p to use.

        default: ``0.0``


    - **driven_excitation** *(str)*:

        Denotes driven excitation by `AC-dipole` (acd) or by `ADT` (adt)

        choices: ``('acd', 'adt')``


    - **drv_tunes** *(float)*:

        Driven tunes without integer part.


    - **energy** *(float)*:

        Energy in **Tev**.


    - **fullresponse**:

        If True, outputs also fullresponse madx file.

        action: ``store_true``


    - **model_dir** *(str)*:

        Path to model directory; loads tunes and excitation from model!


    - **modifiers** *(str)*:

        Path to the optics file to use (modifiers file).


    - **nat_tunes** *(float)*:

        Natural tunes without integer part.


    - **ring** *(int)*:

        Ring to use.

        choices: ``(1, 2, 3, 4)``


    - **xing**:

        If True, x-ing angles will be applied to model

        action: ``store_true``
"""
import logging
import os

from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import (Accelerator,
                                                 AcceleratorDefinitionError)
from omc3.model.constants import PLANE_TO_HV
LOGGER = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


class Psbooster(Accelerator):
    """Parent Class for Psbooster-types."""
    NAME = "psbooster"

    @staticmethod
    def get_parameters():
        params = super(Psbooster, Psbooster).get_parameters()
        params.add_parameter(name="ring", type=int, choices=(1, 2, 3, 4), help="Ring to use.")
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.ring = opt.ring

    @property
    def ring(self):
        if self._ring is None:
            raise AcceleratorDefinitionError("The accelerator definition is incomplete, ring "
                                             "has to be specified (--ring option missing?).")
        return self._ring

    @ring.setter
    def ring(self, value):
        if value not in (1, 2, 3, 4):
            raise AcceleratorDefinitionError("Ring parameter has to be one of (1, 2, 3, 4)")
        self._ring = value

    def verify_object(self):
        _ = self.ring

    @classmethod
    def get_dir(cls):
        return os.path.join(CURRENT_DIR, cls.NAME)

    @classmethod
    def get_file(cls, filename):
        return os.path.join(CURRENT_DIR, cls.NAME, filename)

    def get_exciter_bpm(self, plane, bpms):
        if not self.excitation:
            return None
        bpms_to_find = [f"BR{self.ring}.BPM3L3", f"BR{self.ring}.BPM4L3"]
        found_bpms = [bpm for bpm in bpms_to_find if bpm in bpms]
        if not len(found_bpms):
            raise KeyError
        return (list(bpms).index(found_bpms[0]), found_bpms[0]), f"{PLANE_TO_HV[plane]}ACMAP"


class _PsboosterSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None
