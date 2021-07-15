# MIT License
#
# Copyright (c) 2021 Michael Lutz (73613634+michaellutz2609@users.noreply.github.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import math
from datetime import datetime
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, join
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from img_processing.analyse.analysis_io import Analysis_IO
from img_processing.analyse.object_classifier import classify_moons_02
from img_processing.analyse.ser_analyser import SerAnalyzer
from img_processing.ser.ser_file_io import SerReader

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


context = {
    "file_name": "../data/ser/2021-06-07/03_34_00_pipp.ser",
    "step": 1,
    "fwhm": 22.0,
    "factor": 60.0,
    "aperture_factor_inner": 1.0,
    "aperture_factor_outer": 1.3
}


def get_default_tk(context):
    ''' Creates the default toolkit for interaction with the Analyser.

    :param context:
    :return: SerAnalyzer
    '''
    reader: SerReader = SerReader(file_name=context["file_name"], step=context["step"])
    analyzer: SerAnalyzer = SerAnalyzer(reader, context=context, classifier=classify_moons_02)
    io_mgr: Analysis_IO = Analysis_IO(context)
    analyzer.set_io_mgr(io_mgr)
    return analyzer


def visualize_and_plot():
    """
    Scenario: visualize and plot
    :return:
    """
    analyzer = get_default_tk(context)
    analyzer.load_analysis()
    # analyzer.load_images()
    # src_obj_ts_table = analyzer.sources_objects_timestamps
    # frm_ts_table = analyzer.frames_timestamps
    # obj_ap_ts_table = analyzer.objects_apertures_timestamps({analyzer.fwhm, 1.3 * analyzer.fwhm})
    # obj_ap_ts_table = analyzer.objects_apertures_timestamps({context["fwhm"] * 1.0, context["fwhm"] * 1.3,
    #                                                          context["fwhm"] * 1.8, context["fwhm"] * 2.2,
    #                                                          context["fwhm"] * 2.6, context["fwhm"] * 3.0,
    #                                                          context["fwhm"] * 3.4, context["fwhm"] * 3.8})
    obj_ap_ts_by_frm_table = analyzer.objects_apertures_timestamps_by_frame({context["fwhm"] * 1.0,
                                                                             context["fwhm"] * 1.3,
                                                                             context["fwhm"] * 1.8,
                                                                             context["fwhm"] * 2.2,
                                                                             context["fwhm"] * 2.6,
                                                                             context["fwhm"] * 3.0,
                                                                             context["fwhm"] * 3.4,
                                                                             context["fwhm"] * 3.8})

    # analyzer._io_mgr.save_aperture_sum_map(analyzer.aperture_sum_map({context["fwhm"] * 3.8}))

    fig_rows, fig_cols = 1, 1
    fig: Figure = plt.figure(figsize=(16, 9), dpi=300)
    ax: Axes = fig.add_subplot(fig_rows, fig_cols, 1)
    ax.grid()
    ax.plot_date(x_datetime(obj_ap_ts_by_frm_table), ap_sum_ratio(obj_ap_ts_by_frm_table, 3.8), "r+", xdate=True)
    plt.show()


def report_and_plot():
    """
    Scenario: do some reporting, or fine-tuning of analysis --> finding the right parameters
    :return:
    """
    analyzer = get_default_tk(context)
    analyzer.load_analysis()
    analyzer.frames[89].report()


def do_analysis():
    """
    Scenario: do analysis
    :return:
    """
    analyzer = get_default_tk(context)
    analyzer.run_default_analysis()
    # do an additional aperture measurement
    analyzer.aperture_sum_map({context["fwhm"] * 1.8, context["fwhm"] * 2.2})
    analyzer.save_tables()


def plot_by_group_lambda_xy_bin(ax: Axes, data_table: Table, x: Callable, y: Callable, group_by: str = 'object',
                            groups_dict: dict = {'a': 'b+', 'b': 'gx'}, moving_avg_window: int = 1,
                            date_scale: bool = True, x_range=(0, 0)):
    series_by_object = data_table.group_by(group_by)
    kernel = np.ones(moving_avg_window) / moving_avg_window
    if x_range != (0, 0):
        ax.set_xlim(*x_range)
    for group in groups_dict.keys():
        mask = series_by_object.groups.keys[group_by] == group
        series = series_by_object.groups[mask]
        timestamps = series["timestamp"]
        timestamps_bin = np.trunc(timestamps * 1E7)
        series_grouped = series.group_by(timestamps_bin)
        series_bin = series_grouped.groups.aggregate(np.mean)
        if date_scale:
            ax.plot_date(convolve_datetime64_series(x(series_bin), kernel),
                         np.convolve(y(series_bin), kernel, mode="valid"),
                         groups_dict[group], label=group, xdate=True)
        else:
            ax.plot(convolve_datetime64_series(x(series_bin), kernel),
                    np.convolve(y(series_bin), kernel, mode="valid"),
                    groups_dict[group], label=group)


def plot_by_group_lambda_xy(ax: Axes, data_table: Table, x: Callable, y: Callable, group_by: str = 'object',
                            groups_dict: dict = {'a': 'b+', 'b': 'gx'}, moving_avg_window: int = 1,
                            date_scale: bool = True, x_range=(0, 0), label_dict: dict = {'a': 'Europe', 'b': 'Io'}):
    series_by_object = data_table.group_by(group_by)
    kernel = np.ones(moving_avg_window) / moving_avg_window
    if x_range != (0, 0):
        ax.set_xlim(*x_range)
    for group in groups_dict.keys():
        mask = series_by_object.groups.keys[group_by] == group
        series = series_by_object.groups[mask]
        if date_scale:
            ax.plot_date(convolve_datetime64_series(x(series), kernel),
                         np.convolve(y(series), kernel, mode="valid"),
                         groups_dict[group], xdate=True, label=label_dict[group])
        else:
            ax.plot(convolve_datetime64_series(x(series), kernel),
                    np.convolve(y(series), kernel, mode="valid"),
                    groups_dict[group], label=group)


def plot_by_joined_group_lambda_xy(ax: Axes, joined_data_table: Table, x: Callable, y: Callable,
                                   moving_avg_window: int = 1,
                                   label="Undefined"):
    kernel = np.ones(moving_avg_window) / moving_avg_window
    # ax.plot_date(convolve_datetime64_series(x(joined_data_table), kernel),
    #              np.convolve(y(joined_data_table), kernel, mode="valid"),
    #              fmt="b+", label=label, xdate=True)
    # ax.set_ylim(0.10, 0.90)
    ax.plot_date(convolve_datetime64_series(x(joined_data_table), kernel),
                 np.convolve(y(joined_data_table), kernel, mode="valid"),
                 fmt=".-", label=label, xdate=True)


def convolve_datetime64_series(datetimes64: np.ndarray, kernel: np.ndarray):
    min = np.min(datetimes64)
    return np.convolve(datetimes64 - min, kernel, mode="valid") + min


x_datetime = lambda table: np.array(table['datetime'], dtype=np.datetime64).astype(datetime)
x_timestamp = lambda table: np.array(table['timestamp'])

y_distance_xy = lambda table: np.sqrt(x_pos(table) ** 2 + y_pos(table) ** 2)
y_aperture_sum = lambda table, factor: np.array(table[key_ap_factor(factor)])

# Creates parameterized access key for aperture_sum
key_ap_factor = lambda factor: f'aperture_sum_{context["fwhm"] * factor}'
# Creates parameterized access key for aperture_sum and object
key_ap_factor_obj = lambda factor, obj: f'{key_ap_factor(factor)}_{obj}'

# Computes local background (average sum / pixel) of circular ring between inner (radius_1=fwhm*fac_1)
# and outer circle (radius_2=fwhm*fac_2)
local_background = lambda table, fac_1, fac_2: np.array((table[key_ap_factor(fac_2)] - table[key_ap_factor(fac_1)]) /
                                                        context["fwhm"] ** 2 / math.pi / (fac_2 ** 2 - fac_1 ** 2))

# Computes absolute aperture sum subtracted by local background (based on circular ring fac_1/fac_2)
ap_sum_bg_subtracted = lambda table, fac_1, fac_2: np.array(table[key_ap_factor(fac_1)] - (table[key_ap_factor(
    fac_2)] - table[key_ap_factor(fac_1)]) * fac_1 ** 2 / (fac_2 ** 2 - fac_1 ** 2))

# Computes ratio of background subtracted aperture sums (requires folded data by frame)
ap_sum_bg_subtracted_ratio = lambda table, fac_1, fac_2: np.array((table[key_ap_factor_obj(fac_1, "a")] - (table[key_ap_factor_obj(fac_2, "a")] - table[key_ap_factor_obj(fac_1, "a")]) * fac_1 ** 2 / (fac_2 ** 2 - fac_1 ** 2)) / (table[key_ap_factor_obj(fac_1, "b")] - (table[key_ap_factor_obj(fac_2, "b")] - table[key_ap_factor_obj(fac_1, "b")]) * fac_1 ** 2 / (fac_2 ** 2 - fac_1 ** 2)))
ap_sum_ratio = lambda table, fac: np.array(table[key_ap_factor_obj(fac, "a")] / table[key_ap_factor_obj(fac, "b")])

x_pos = lambda table: np.array(table['xcentroid'])
y_pos = lambda table: np.array(table['ycentroid'])
y_distance_zero = lambda table: np.sqrt(x_pos(table) ** 2 + y_pos(table) ** 2)

if __name__ == '__main__':
    visualize_and_plot()
