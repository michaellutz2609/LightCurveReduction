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
from typing import Callable, Dict, Any

import astropy.table as apt
from astropy.table import Table, join

from img_processing.analyse.analysis_io import AnalysisDataType, Analysis_IO
from img_processing.analyse.frame_analyser import StarImageFrame
from img_processing.analyse.object_classifier import no_classifier
from img_processing.ser.ser_file_io import SerReader

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class SerAnalyzer:
    """
    A simple testbed to read in images from a ser-"Movie" (which is just a series of frames of pixels). Idea is to
    analyse the single frames in terms of overall luminosity, sharpness (focus)
    and some other results from AstroPy's DAOStarFinder.

    Implement processing options: use streaming for high number of images to process, use batch for in-memory and
    number (tables) processing.

    Idea: Minimize data size by cropping relevant frame areas (identified by positions), write to smaller file.
    Idea: Quickly step through full size frames, identify positions, best guess for area, ...
    Idea: refactor OO-style: base element is a frame with some characteristics, sources, apertures, identities, ...
    Idea: then implement some persistence mechanism for serializing / deserializing
    """
    _io_mgr: Analysis_IO
    _frames: list[StarImageFrame]

    _sources_objects_timestamps: Table
    _frames_timestamps: Table
    _objects_apertures_map: dict[str, Table]
    _objects_apertures_timestamps_map: dict[str, Table]
    _objects_apertures_timestamps_by_frame_map: dict[str, Table]

    def __init__(self, reader: SerReader, context: dict, classifier: Callable = no_classifier):
        '''
        fwhm and factor are constant for each instance of SerAnalyzer, being the basis for the analysis
        :param reader:
        :param fwhm:
        :param factor:
        :param aperture_factor:
        :param aperture_factor_outer:
        :param classifier:
        '''
        self._reader = reader
        self._context = context
        self._fwhm: int = context["fwhm"]
        self._factor: float = context["factor"]
        self._frames_table = Table(
            names=[StarImageFrame.FRAME_PK_NAME, 'np.sum(img)', 'np.median(img)', 'mad_std(img)'],
            dtype=[StarImageFrame.FRAME_PK_DTYPE, int, float, float])
        self._sources_table = Table()
        self._objects_table = Table(names=StarImageFrame.OBJECTS_TABLE_NAMES, dtype=StarImageFrame.OBJECTS_TABLE_DTYPE)
        self._aperture_sum_map: dict[float, Table] = {}

        self._io_mgr: Analysis_IO
        self._frames: list[StarImageFrame] = []

        self._sources_objects_timestamps = None
        self._frames_timestamps = None
        self._objects_apertures_map = dict()
        self._objects_apertures_timestamps_map = dict()
        self._objects_apertures_timestamps_by_frame_map = dict()

        # fig = plt.figure(figsize=(16, 9))
        # fig.subplots_adjust(hspace=0.05, wspace=0.05)
        # plt.show()

    # 4 steps:
    # a) meta data + timestamps_table, created upon initialization of SerReader
    # e) simple processing: img_sum, mad, etc.
    # b) find star sources
    # c) do photometry (list of aperture-sizes)
    # d) identify sources, i.e. name sources
    def run_default_analysis(self):
        sources_list: list[Table] = []
        objects_list: list[Table] = []
        aperture_sum_map_list: list[dict[float, Table]] = []

        while self._reader.has_next():
            # ax = fig.add_subplot(3, 4, 1)
            img = self._reader.next_frame()
            frame = StarImageFrame(self._reader.current_frame_id, img, self._reader.colorId, self._fwhm, self._factor)

            self._frames_table.add_row(
                [self._reader.current_frame_id, frame.img_sum, frame.img_median, frame.bkg_sigma])
            self._frames.append(frame)

            frame.run_analysis()

            sources_list.append(frame.sources())
            objects_list.append(frame.objects())
            aperture_sum_map_list.append(frame.aperture_sum_all_map())

        self._sources_table = apt.vstack(sources_list)
        self._objects_table = apt.vstack(objects_list)
        self._aggregate_aperture_sum_maps(aperture_sum_map_list)

    def load_analysis(self):
        self.reader.set_meta_dict(self._io_mgr.load_meta())
        self.reader.set_timestamps_table(self._io_mgr.load_table(AnalysisDataType.timestamps))
        self._frames_table = self._io_mgr.load_table(AnalysisDataType.frames)
        self._sources_table = self._io_mgr.load_table(AnalysisDataType.sources)
        self._objects_table = self._io_mgr.load_table(AnalysisDataType.objects)
        self._aperture_sum_map = self._io_mgr.load_aperture_sum_map()
        # while self._reader.has_next():
        #     img = self._reader.next_frame()
        for frame_row in self._frames_table:
            frame = StarImageFrame(frame_row[StarImageFrame.FRAME_PK_NAME], None,
                                   self._reader.colorId, self._fwhm, self._factor)
            frame._bkg_sigma = frame_row["mad_std(img)"]
            frame._median = frame_row["np.median(img)"]
            frame._sum = frame_row["np.sum(img)"]
            self._frames.append(frame)

            mask = (self._sources_table[StarImageFrame.FRAME_PK_NAME] == frame.frame_id)
            frame._sources = self._sources_table[mask]
            mask = (self._objects_table[StarImageFrame.FRAME_PK_NAME] == frame.frame_id)
            frame._objects = self._objects_table[mask]
            for radius, ap_table in self._aperture_sum_map.items():
                mask = (ap_table[StarImageFrame.FRAME_PK_NAME] == frame.frame_id)
                frame._photometry_map[radius] = ap_table[mask]

    def load_images(self):
        # requires `open` first
        self.reader.open()
        for frame in self.img_frames:
            frame._img = self.reader.next_frame()

    def _aggregate_aperture_sum_maps(self, aperture_sum_map_list):
        aperture_sum_list_map: dict[float, list[Table]] = {}
        for ap_sums_map in aperture_sum_map_list:
            for radius, ap_table in ap_sums_map.items():
                if radius in aperture_sum_list_map:
                    aperture_sum_list_map[radius].append(ap_table)
                else:
                    aperture_sum_list_map[radius] = [ap_table, ]
        for radius, ap_table_list in aperture_sum_list_map.items():
            ap_table: Table = apt.vstack(ap_table_list)
            # add some meta data to the column "aperture_sum"
            self._aperture_sum_map[radius] = ap_table

    def save_tables(self):
        self._io_mgr.save_meta(self.reader.meta_dict())
        self._io_mgr.save_table(self.reader.timestamps_table, AnalysisDataType.timestamps)
        self._io_mgr.save_table(self.frames, AnalysisDataType.frames)
        self._io_mgr.save_table(self.sources, AnalysisDataType.sources)
        self._io_mgr.save_table(self.objects, AnalysisDataType.objects)
        self._io_mgr.save_aperture_sum_map(self._aperture_sum_map)

    def save_meta(self):
        self._io_mgr.save_meta(self.reader.meta_dict())

    @property
    def img_frames(self) -> list[StarImageFrame]:
        return self._frames

    @property
    def frames(self) -> Table:
        return self._frames_table

    @property
    def sources(self) -> Table:
        return self._sources_table

    @property
    def objects(self) -> Table:
        return self._objects_table

    def aperture_sum_map(self, aperture_radii: set[float]) -> dict[float, Table]:
        rv: dict = {}
        # Create set of missing radii (to be computed)
        missing_radii = aperture_radii - self._aperture_sum_map.keys()
        if len(missing_radii) > 0:
            aperture_sum_map_list: list[dict[float, Table]] = []
            for frame in self._frames:
                aperture_sum_map_list.append(frame.aperture_sum_map(missing_radii))
            self._aggregate_aperture_sum_maps(aperture_sum_map_list)
        rv = {k: v for k, v in self._aperture_sum_map.items() if k in aperture_radii}
        return rv

    def set_io_mgr(self, io_mgr: Analysis_IO):
        self._io_mgr = io_mgr

    @property
    def sources_objects_timestamps(self) -> Table:
        """
        Creates a data view for further analysis or plotting by joining
        - sources,
        - identified objects,
        - and timestamps.
        :return: Table object
        """
        if self._sources_objects_timestamps is None:
            tmp_table = join(self.sources, self.objects,
                             keys=[StarImageFrame.FRAME_PK_NAME, StarImageFrame.SEQ_ID_NAME])
            self._sources_objects_timestamps = self._join_timestamps(tmp_table)
        return self._sources_objects_timestamps

    @property
    def frames_timestamps(self) -> Table:
        """
        Creates a data view for further analysis or plotting by joining
        - frames,
        - and timestamps.
        :return: Table object
        """
        if self._frames_timestamps is None:
            self._frames_timestamps = self._join_timestamps(self.frames)
        return self._frames_timestamps

    def objects_apertures(self, aperture_radii: set[float]) -> Table:
        """
        Creates a data view for further analysis or plotting by joining
        - objects,
        - and apertures.
        :param aperture_radii:
        :rtype: object
        :return: Table object
        """
        key, rv = SerAnalyzer._from_cache(self._objects_apertures_map, aperture_radii)
        if rv is not None:
            return rv

        rv: Table = self.objects
        # ToDo: introduce constants for standard table columns
        col_name = "aperture_sum"
        first_join = True
        for radius, ap_table in self.aperture_sum_map(aperture_radii).items():
            # ToDo: speed up performance by omitting xcenter/ycenter after first join (construct new table with
            #  specific columns first)? By default, join operates on all columns
            #  with the same name.
            if first_join:
                tmp_ap_table = ap_table
                first_join = False
            else:
                tmp_ap_table: Table = Table([ap_table[StarImageFrame.FRAME_PK_NAME],
                                             ap_table[StarImageFrame.SEQ_ID_NAME],
                                             ap_table[col_name]])
            rv = join(rv, tmp_ap_table, keys=[StarImageFrame.FRAME_PK_NAME, StarImageFrame.SEQ_ID_NAME])
            rv.rename_column(col_name, f"{col_name}_{radius}")
        self._objects_apertures_map[key] = rv
        return rv

    def objects_apertures_timestamps(self, aperture_radii: set[float]) -> Table:
        """
        Creates a data view for further analysis or plotting by joining
        - objects,
        - apertures,
        - and timestamps.
        :param aperture_radii:
        :rtype: object
        :return: Table object
        """
        # ToDo: introduce generic cache mechanism (i.e. a dict of dicts)
        key, rv = SerAnalyzer._from_cache(self._objects_apertures_timestamps_map, aperture_radii)
        if rv is not None:
            return rv

        rv = self._join_timestamps(self.objects_apertures(aperture_radii))
        self._objects_apertures_timestamps_map[key] = rv
        return rv

    @staticmethod
    def _from_cache(cache: dict[str, Table], aperture_radii: set[float]) -> (str, Table):
        key: str = "-".join(map(str, aperture_radii))
        if cache is not None and key in cache:
            return key, cache[key]
        return key, None

    def _join_timestamps(self, table: Table) -> Table:
        return join(table, self.reader.timestamps_table, keys=[StarImageFrame.FRAME_PK_NAME, ])

    def _objects_by_frame(self, table: Table) -> Table:
        """
        | frame_id | id_a | object_a | id_b | object_b |
        ________________________________________________
        | 13       | 1    | a        | 2    | b        |
        ToDo: introduce generic mechanism linked to the classifier method (object names, more than 2 sources)
        ToDo: Error?: More than one object per frame identified as a (or b)? --> Ensure object is unique per frame
        Useful for analysis of dropped frames, single/multiple detection and so on...
        To be used for relative photometry analysis flux(a)/flux(b)
        :return:
        """
        object_names = ["a", "b"]
        series_by_object = table.group_by(StarImageFrame.OBJ_NAME)
        mask_a = series_by_object.groups.keys[StarImageFrame.OBJ_NAME] == object_names[0]
        series_a = series_by_object.groups[mask_a]
        mask_b = series_by_object.groups.keys[StarImageFrame.OBJ_NAME] == object_names[1]
        series_b = series_by_object.groups[mask_b]
        rv = join(series_a, series_b, keys=[StarImageFrame.FRAME_PK_NAME], table_names=object_names)
        return rv

    def objects_apertures_timestamps_by_frame(self, aperture_radii: set[float]) -> Table:
        """
        :param aperture_radii:
        :return:
        """
        key, rv = SerAnalyzer._from_cache(self._objects_apertures_timestamps_by_frame_map, aperture_radii)
        if rv is not None:
            return rv

        rv = self._join_timestamps(self._objects_by_frame(self.objects_apertures_timestamps(aperture_radii)))
        self._objects_apertures_timestamps_by_frame_map[key] = rv
        return rv

    @property
    def reader(self):
        return self._reader

    @property
    def fwhm(self):
        return self._fwhm
