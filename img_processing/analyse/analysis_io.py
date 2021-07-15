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

import json
import logging
from enum import Enum
from pathlib import Path

from astropy.table import Table

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisDataType(Enum):
    timestamps = "timestamps"
    frames = "frames"
    sources = "sources"
    objects = "objects"
    aperture_sum = "aperture_sum"


class Analysis_IO:

    def __init__(self, context: dict):
        prefix: str = f"step-{context['step']}_fwhm-{context['fwhm']}-factor-{context['factor']}"
        self._file_path: Path = Path(context["file_name"])
        file_name: str = self._file_path.parts[-1]
        self._base_name: str = file_name.split('.')[0]
        self._analysis_base_dir: Path = Path(f"{self._file_path.parent}/{self._base_name}")
        if not self._analysis_base_dir.exists():
            self._analysis_base_dir.mkdir()
        self._prefix = prefix

    def _create_file_name(self, data_type: AnalysisDataType):
        return f"{self._analysis_base_dir}/{self._prefix}.{data_type.name}.ecsv"

    def save_table(self, table: Table, data_type: AnalysisDataType):
        logger.info(f"Saving data table for type={data_type.name}")
        table.write(self._create_file_name(data_type))

    def load_table(self, data_type: AnalysisDataType) -> Table:
        logger.info(f"Loading data table for type={data_type.name}")
        return Table.read(self._create_file_name(data_type))

    def save_aperture_sum_map(self, aperture_sum_map: dict[float, Table]):
        for radius, aperture_sum_table in aperture_sum_map.items():
            logger.info(f"Saving aperture sums data file for radius={radius}")
            aperture_sum_table.write(
                f"{self._analysis_base_dir}/{self._prefix}_radius-{radius}.{AnalysisDataType.aperture_sum.name}.ecsv")

    def load_aperture_sum_map(self) -> dict[float, Table]:
        rv: dict[float, Table] = {}
        for ap_file in self._analysis_base_dir.glob(
                f"{self._prefix}_radius-*.{AnalysisDataType.aperture_sum.name}.ecsv"):
            ap_table: Table = Table.read(ap_file)
            rv[ap_table.meta["radius"]] = ap_table
            logger.info(f"Loading aperture sums for radius={ap_table.meta['radius']}")
        return rv

    def save_meta(self, meta_dict: dict):
        with open(f"{self._analysis_base_dir}/{self._base_name}.json", "w") as fp:
            json.dump(meta_dict, fp, sort_keys=True, indent=4)

    def load_meta(self) -> dict:
        rv: dict = {}
        with open(f"{self._analysis_base_dir}/{self._base_name}.json", "r") as fp:
            rv = json.load(fp)
        return rv
