from pathlib import Path

import matplotlib
import pytest

from omc3.amplitude_detuning_analysis import analyse_with_bbq_corrections
from omc3.plotting.plot_amplitude_detuning import main as pltampdet
from omc3.plotting.plot_bbq import main as pltbbq

# Forcing non-interactive Agg backend so rendering is done similarly across platforms during tests
matplotlib.use("Agg")

INPUT_DIR = Path(__file__).parent.parent / "inputs"
AMPDET_INPUT = INPUT_DIR / "amplitude_detuning"


@pytest.mark.basic
def test_bbq_plot(tmp_path):
    fig = pltbbq(
        input=str(AMPDET_INPUT / "bbq_ampdet.tfs"), output=str(tmp_path / "bbq.pdf"),
    )
    assert fig is not None
    assert len(list(tmp_path.glob("*.pdf"))) == 1


@pytest.mark.basic
def test_ampdet_plot(tmp_path):
    fig = pltampdet(
        kicks=[str(AMPDET_INPUT / "kick_ampdet_xy.tfs")],
        labels=["Beam 1 Vertical"],
        plane="Y",
        correct_acd=True,
        output=str(tmp_path / "ampdet.pdf"),
    )
    assert len(fig) == 4
    assert len(list(tmp_path.glob("*.pdf"))) == 4


@pytest.mark.extended
@pytest.mark.parametrize("method", ["cut", "minmax", "outliers"])
def test_amplitude_detuning_full(method, tmp_path):
    setup = dict(
        beam=1,
        kick=str(AMPDET_INPUT),
        plane="Y",
        label="B1Vkicks",
        bbq_in=str(AMPDET_INPUT / "bbq_ampdet.tfs"),
        detuning_order=1,
        output=str(tmp_path),
        window_length=100 if method != "outliers" else 50,
        tunes=[0.2838, 0.3104],
        tune_cut=0.001,
        tunes_minmax=[0.2828, 0.2848, 0.3094, 0.3114],
        fine_window=50,
        fine_cut=4e-4,
        outlier_limit=1e-4,
        bbq_filtering_method=method,
    )
    kick_df, bbq_df = analyse_with_bbq_corrections(**setup)

    assert len(list(tmp_path.glob("*.tfs"))) == 2
    assert len([k for k, v in kick_df.headers.items() if k.startswith("ODR") and v != 0]) == 16


@pytest.mark.extended
def test_no_bbq_input(tmp_path):
    setup = dict(
        beam=1,
        kick=str(AMPDET_INPUT),
        plane="Y",
        label="B1Vkicks",
        detuning_order=1,
        output=str(tmp_path),
    )
    kick_df, bbq_df = analyse_with_bbq_corrections(**setup)

    assert bbq_df is None
    assert len(list(tmp_path.glob("*.tfs"))) == 1
    assert len([k for k, v in kick_df.headers.items() if k.startswith("ODR") and v != 0]) == 8
