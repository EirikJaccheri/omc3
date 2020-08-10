from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from matplotlib import pyplot as plt


class FigureContainer:
    """ Container for attaching additional information to one figure. """
    def __init__(self, id_: str, path: Path, axes_ids: Iterable[str]) -> None:
        self.fig, axs = plt.subplots(nrows=len(axes_ids))
        self.fig.canvas.set_window_title(id_)

        if len(axes_ids) == 1:
            axs = [axs]

        self.axes_ids = axes_ids  # to keep order
        self.axes = {ax_id: ax for ax_id, ax in zip(axes_ids, axs)}
        self.xlabels = {ax_id: None for ax_id in axes_ids}
        self.ylabels = {ax_id: None for ax_id in axes_ids}
        self.data = {ax_id: {} for ax_id in axes_ids}
        self.path = path

    def __getitem__(self, ax_id):
        return self.axes[ax_id], self.data[ax_id], self.xlabels[ax_id], self.ylabels[ax_id]


class FigureCollector:
    """ Class to collect figure containers and manage data adding. """
    def __init__(self) -> None:
        self.fig_dict = OrderedDict()   # dictionary of matplotlib figures, for output
        self.figs = OrderedDict()       # dictionary of FigureContainers, used internally

    def add_data_for_id(self, figure_id: str, label: str, data: dict,
                        x_label: str, y_label: str,
                        path: Path = None, axes_id: str = '',
                        axes_ids: Iterable[str] = ('',)) -> None:
        """ Add the data at the appropriate figure container. """
        try:
            figure_cont = self.figs[figure_id]
        except KeyError:
            figure_cont = FigureContainer(figure_id, path, axes_ids)

            self.figs[figure_id] = figure_cont
            self.fig_dict[figure_id] = figure_cont.fig

        figure_cont.ylabels[axes_id] = y_label
        figure_cont.xlabels[axes_id] = x_label
        figure_cont.data[axes_id][label] = data
