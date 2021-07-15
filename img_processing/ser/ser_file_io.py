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

import datetime
import logging
import os
from os import stat_result
from typing import BinaryIO

import numpy as np
from astropy.table import Table
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, demosaicing_CFA_Bayer_Malvar2004, \
    demosaicing_CFA_Bayer_bilinear

from img_processing.analyse import frame_analyser
from img_processing.analyse.frame_analyser import StarImageFrame

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

ser_header_dtype = np.dtype([
    ("fileID", "S14"),
    ("luID", "<u4"),
    ("colorID", "<u4"),
    ("littleEndian", "<u4"),
    ("imageWidth", "<u4"),
    ("imageHeight", "<u4"),
    ("pixelDepthPerPlane", "<u4"),
    ("frameCount", "<u4"),
    ("observer", "S40"),
    ("instrument", "S40"),
    ("telescope", "S40"),
    ("dateTime", "<u8"),
    ("dateTime_UTC", "<u8"),
])

_endianess_map: dict = {0: 'BIG_ENDIAN', 1: 'LITTLE_ENDIAN'}

_ser_header_length: int = 178


class SerReader:
    """ Class to read SER-files in a sequential way.
    SER-files are basically uncompressed image arrays. Details of the
    format are available under http://www.grischa-hahn.homepage.t-online.de/astro/ser/. Some other useful software to
    deal with this image series format, that is used to my knowledge mostly used in the astronomy imaging community,
    is PIPP (Planetary Image PreProcessing), available under https://sites.google.com/site/astropipp/.

    At the current design, the class itself is responsible for too many things: while reading the files (including
    meta-data) is fine, further processing of the image, e.g. to debayer the raw data *should* not be part of this
    code, but rather be moved to some filtering package. So, there's room for improvement, but let's follow the KISS
    principle first, unless things get really messy.
    """
    _file: BinaryIO
    _statinfo: stat_result

    menon = lambda self, img, pattern: demosaicing_CFA_Bayer_Menon2007(img, pattern=pattern, refining_step=True)
    malvar = lambda self, img, pattern: demosaicing_CFA_Bayer_Malvar2004(img, pattern=pattern)
    bilinear = lambda self, img, pattern: demosaicing_CFA_Bayer_bilinear(img, pattern=pattern)
    debayer_algorithms = (bilinear, malvar, menon)

    def __init__(self, file_name: str, start: int = 0, end: int = None, step: int = 1):
        self._meta_dict: dict = {}
        self._file_name = file_name
        self._next_frame = start
        self._end = end
        self._step = step
        self._current_frame = -1
        self._pos = 0
        self._file = None

    def __del__(self):
        self.close()

    def open(self):
        try:
            self._statinfo = os.stat(self._file_name)
            self._file = open(self._file_name, "rb")
        except IOError as exc:
            logger.error(f"Cannot open file {self._file_name}: {exc}")
            raise IOError(exc)

        file_size = self._statinfo.st_size  # file size (bytes)
        logger.info(f"File size: {file_size} bytes")

        ser_header = np.fromfile(self._file, dtype=ser_header_dtype, count=1)[0]
        self._pos = _ser_header_length

        for key in ser_header_dtype.names:
            logger.info(f"{key} of type {ser_header_dtype.fields[key]}: {ser_header[key]} ")
            obj = ser_header[key]
            encoding = 'ASCII'  # as stated in spec
            if isinstance(obj, np.bytes_):
                obj = obj.decode(encoding)
            elif isinstance(obj, np.generic):
                obj = obj.item()
            self._meta_dict[key] = obj

        self._set_header_props(ser_header)
        logger.info(f"colorId_label: {frame_analyser.colorId_map[self.colorId]['label']}")
        logger.info(f"littleEndian_label: {_endianess_map[self.littleEndian]}")
        self._meta_dict["littleEndian_label"] = _endianess_map[self.littleEndian]
        self._meta_dict["colorId_label"] = frame_analyser.colorId_map[self.colorId]['label']

        self._read_timestamps(file_size)

        # reset file position to first frame (after header)
        self._file.seek(_ser_header_length)
        self._pos = _ser_header_length

    def set_meta_dict(self, meta_dict: dict):
        """
        Presets the controlling properties of SerReader without having to read the binary file (in general, the dict
        is read from a json-file previously created from the real read.
        :param meta_dict: dictionary of values as read from a ser file (plus some additional)
        :return:
        """
        self._meta_dict = meta_dict
        self._set_header_props(meta_dict)

    def _set_header_props(self, meta_data):
        """
        Sets meta data and derives some required properties, e.g. _frameSize
        :param meta_data: Either type dict or ser_header_dtype
        :return:
        """
        self.colorId = meta_data["colorID"]
        self.littleEndian = meta_data["littleEndian"]
        self.imageWidth = meta_data["imageWidth"]
        self.imageHeight = meta_data["imageHeight"]
        self.pixelDepthPerPlane = meta_data["pixelDepthPerPlane"]
        self.frameCount = meta_data["frameCount"]
        self._numberOfPlanes = 1
        if self.colorId > 99:
            self._numberOfPlanes = 3
        self._frameSize = self.imageWidth * self.imageHeight * self._numberOfPlanes
        self._pixelDepthPerPlane_real = self.pixelDepthPerPlane
        self._dtype = np.uint8
        if self.pixelDepthPerPlane > 8:
            self._pixelDepthPerPlane_real = 14
            self._dtype = np.uint16

    def _read_timestamps(self, file_size: int):
        file_pos_timestamps = _ser_header_length + self._frameSize * self.frameCount * (self.pixelDepthPerPlane // 8)
        # check, if optional timestamps are there
        if file_size > file_pos_timestamps:
            self._timestamps_table = Table(
                names=[StarImageFrame.FRAME_PK_NAME, "timestamp", "next_timestamp_delta_ms", "datetime"],
                dtype=[int, np.uint64, np.uint64, datetime.datetime])
            # read timestamps of frames
            self._file.seek(file_pos_timestamps)
            timestamps = np.fromfile(self._file, dtype=np.uint64, count=self.frameCount)
            for fid in range(len(timestamps)):
                ts = timestamps[fid]
                ts_delta = 0
                if fid < len(timestamps) - 1:
                    # logger.info(f"Delta t={(ts - ts_begin) / 1E4}ms")
                    # delta of timestamps in ms units
                    # interpretation: upper bound of exposure time of frame, naturally unknown for last frame (set to 0)
                    ts_delta = (timestamps[fid + 1] - ts) / 1E4

                dt = datetime.datetime.min + datetime.timedelta(seconds=ts / 1E7)
                self._timestamps_table.add_row(vals=[fid, ts, ts_delta, dt])
                # logger.info(dt)

    def set_timestamps_table(self, timestamps_table: Table):
        self._timestamps_table = timestamps_table

    def close(self):
        if self._file is not None:
            logger.info('Closing file!')
            self._file.close()

    def meta_dict(self) -> dict:
        return self._meta_dict

    def has_next(self):
        return not (self._current_frame + self._step) >= self.frameCount

    def next_frame(self, x_0: int = 0, y_0: int = 0, width: int = None, height: int = None):
        if width is None:
            width = self.imageWidth - x_0
        if height is None:
            height = self.imageHeight - y_0

        if self._pos == 0:
            self._file.seek(_ser_header_length)
            self._pos = _ser_header_length

        # read forward to desired start frame
        for i in range(self._next_frame - self._current_frame):
            self._current_frame += 1
            self._data = np.fromfile(self._file, dtype=self._dtype, count=self._frameSize)
            self._pos += self._frameSize
        logger.info(f"Reading frame {self._current_frame}")
        self._next_frame += self._step

        # works only for monochrome (or Bayer) images, i.e. self.numberOfPlanes = 1!
        # self._img = np.right_shift(np.reshape(self._data, [self.imageHeight, self.imageWidth]),
        #                            self.pixelDepthPerPlane % self.pixelDepthPerPlane_real)[y_0:y_0 + height, x_0:x_0 + width]
        # self._img = np.right_shift(np.reshape(self._data, [self.imageHeight, self.imageWidth]),
        #                            self.pixelDepthPerPlane % self.pixelDepthPerPlane_real)
        self._img = np.reshape(self._data, [self.imageHeight, self.imageWidth])
        # self._img = np.right_shift(np.reshape(self._data, [self.imageHeight, self.imageWidth, self.numberOfPlanes]),
        #                            16 % self.pixelDepthPerPlane_real)
        return self._img

    @property
    def current_frame_id(self):
        return self._current_frame

    @property
    def timestamps_table(self) -> Table:
        return self._timestamps_table

    @property
    def file_name(self):
        return self._file_name

    @property
    def step(self):
        return self._step


class SerWriter:

    def __init__(self, file_name: str, header: ser_header_dtype):
        self._file_name = file_name
        self._frame_count = 0
        try:
            self._file = open(self._file_name, "ab")
            header.tofile(self._file_name)
            # self._file.write(header)
        except IOError as exc:
            logger.error('Cannot open file %s: %s', self._file_name, exc)
            raise IOError(exc)

    def __del__(self):
        self.close()

    def add_frame(self, img):
        self._frame_count += 1
        img.tofile(self._file)
        # self._file.write(img)

    def close(self):
        if self._file is not None:
            logger.info('Closing file!')
            self._file.close()
