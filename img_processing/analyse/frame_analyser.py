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
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import mad_std
from astropy.table import Table
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, demosaicing_CFA_Bayer_Malvar2004, \
    demosaicing_CFA_Bayer_bilinear
from photutils import DAOStarFinder, CircularAperture, aperture_photometry
from skimage import color

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

colorId_map: dict = {0: {'label': 'MONO'},
                     8: {'label': 'BAYER_RGGB', 'pattern': 'RGGB', 'matrix2rgb': (0, 1, 1, 2)},
                     9: {'label': 'BAYER_GRBG', 'pattern': 'GRBG', 'matrix2rgb': (1, 0, 2, 1)},
                     10: {'label': 'BAYER_GBRG'},
                     11: {'label': 'BAYER_BGGR'},
                     16: {'label': 'BAYER_CYYM'},
                     17: {'label': 'BAYER_YCMY'},
                     18: {'label': 'BAYER_YMCY'},
                     19: {'label': 'BAYER_MYYC'},
                     100: {'label': 'RGB'},
                     101: {'label': 'BGR'}}


class StarImageFrame:
    FRAME_PK_NAME = "frame_id"
    FRAME_PK_DTYPE = int
    SEQ_ID_NAME = "id"
    SEQ_ID_DTYPE = int
    OBJ_NAME = "object"
    OBJ_DTYPE = str

    OBJECTS_TABLE_NAMES = [FRAME_PK_NAME, SEQ_ID_NAME, OBJ_NAME]
    OBJECTS_TABLE_DTYPE = [FRAME_PK_DTYPE, SEQ_ID_DTYPE, OBJ_DTYPE]

    menon = lambda self, img, pattern: demosaicing_CFA_Bayer_Menon2007(img, pattern=pattern, refining_step=True)
    malvar = lambda self, img, pattern: demosaicing_CFA_Bayer_Malvar2004(img, pattern=pattern)
    bilinear = lambda self, img, pattern: demosaicing_CFA_Bayer_bilinear(img, pattern=pattern)
    debayer_algorithms = (bilinear, malvar, menon)
    add_pk = lambda self, table: table.add_column(self._frame_id, name=StarImageFrame.FRAME_PK_NAME, index=0)

    def __init__(self, frame_id: int, img: np.ndarray, color_id: int, fwhm: float, factor: float):
        self._frame_id = frame_id
        self._img = img
        self._color_id = color_id
        self._fwhm = fwhm
        self._factor = factor

        if self._img is not None:
            self._bkg_sigma = mad_std(self._img)
            self._median = np.median(self._img)
            self._sum = np.sum(self._img)

        self._sources = None
        self._positions: np.ndarray = None
        self._apertures_map: dict = {}
        self._photometry_map: dict = {}
        self._objects = None

    def run_analysis(self):
        self.sources()
        self.aperture_sum_map(aperture_radii={self._fwhm, 1.3 * self._fwhm})
        self.objects()

    def reset(self):
        self._sources = None
        self._apertures_map = None
        self._objects = None

    def sources(self) -> Table:
        if self._sources is None:
            # "Full Width Half Maximum" to identify stars
            logger.info(f"mad_std(img)={self._bkg_sigma}")
            img_gray = self.debayered_gray()

            # TODO: Analyse impact of various background subtraction and preprocessing (best practices)
            # img_proc = stretch_exposure(img)
            # img = img - np.median(img)
            ''' in case of a bayer matrix, mad_std (probably) computes some non-meaningful value due to 
            treating RGB-pixels in the same way'''

            daofind = DAOStarFinder(fwhm=self._fwhm, threshold=self._factor * self._bkg_sigma)
            self._sources: Table = daofind.find_stars(img_gray)
            # self.report_find_results([self._fwhm, ], sources)
            if self._sources is None:
                # daofind.find_stars() returns None, if no stars are found --> create empty table to indicate that.
                self._sources = Table()
            else:
                self.add_pk(self._sources)
                # cutouts: list[_StarCutout] = daofind._star_cutouts
        return self._sources

    def aperture_sum_all_map(self) -> dict[float, Table]:
        return self._photometry_map

    def aperture_sum_map(self, aperture_radii: set[float]) -> dict[float, Table]:
        '''
        Takes a table of star sources (e.g. by DAOStarfind) and does photometric measurements.
        Assumption: sources includes a sequential id of the stars (starting from 1), aperture_photometry also
        generates an id in sync with this one. Well, on second thought this assumption breaks when one position
        shall be evaluated more than once (given a list of radii for varying apertures).
        :param aperture_radii:
        :return:
        '''
        rv: dict = {}
        if self._sources is not None and len(self._sources) > 0:
            for radius in aperture_radii:
                pm = self._photometry_map.get(radius)
                if pm is None:
                    self._apertures_map[radius] = CircularAperture(self.positions, r=radius)
                    pm: Table = aperture_photometry(self._img, self._apertures_map[radius], method="center")
                    self.add_pk(pm)
                    pm.meta["radius"] = radius
                    self._photometry_map[radius] = pm
                rv[radius] = pm
        return rv

    def objects(self, classifier: Callable = None) -> Table:
        if self._objects is None:
            logger.info("Classifier implementation: classify_moons")
            logger.info(f"Working on {len(self._sources)} identified sources.")
            self._objects = Table(names=StarImageFrame.OBJECTS_TABLE_NAMES, dtype=StarImageFrame.OBJECTS_TABLE_DTYPE)

            if self._sources is not None:
                for src in self._sources:
                    obj = None
                    if 0 < src['xcentroid'] < 340:
                        obj = 'a'
                    elif 350 < src['xcentroid'] < 750:
                        obj = 'b'
                    if obj is not None:
                        self._objects.add_row([self._frame_id, src[StarImageFrame.SEQ_ID_NAME], obj])
                        logger.info(f"Identified object {obj}.")
        return self._objects

    def bayer_rgb(self) -> np.ndarray:
        # Check if buffered data exist!
        # Makes only sense, if sensor delivers Bayer-matrix!
        # RGB-Planes = 012
        # GRBG = 1021
        rgb_map = colorId_map[self._color_id]['matrix2rgb']
        bayer_planes = np.zeros((self._img.shape, 3), dtype=np.uint16)
        bayer_planes[::2, ::2, rgb_map[0]] = self._img[::2, ::2]  # ((+, -), (-, -))
        bayer_planes[1::2, ::2, rgb_map[1]] = self._img[1::2, ::2]  # ((-, -), (+, -))
        bayer_planes[::2, 1::2, rgb_map[2]] = self._img[::2, 1::2]  # ((-, +), (-, -))
        bayer_planes[1::2, 1::2, rgb_map[3]] = self._img[1::2, 1::2]  # ((-, -), (-, +))
        return bayer_planes

    def debayered_rgb(self, debayer_impl: Callable = menon):
        # return demosaicing_CFA_Bayer_Malvar2004(self._img, _colorId_map[self.colorId]['pattern'])
        return debayer_impl(self, self._img, colorId_map[self._color_id]['pattern'])

    def debayered_gray(self):
        return color.rgb2gray(self.debayered_rgb())

    def report(self):
        print(self._sources)
        print(self._objects)
        print(self._photometry_map)

    def report_find_results(self, radii: list[float], sources):
        if len(sources) > 2:
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(self._img, cmap='gray_r', origin='lower')
            for ap in apertures:
                ap.plot(color="blue", lw=1.5, alpha=0.5)
            # ax.set_xlim(0, img.shape[1])
            # ax.set_ylim(0, img.shape[0])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            plt.show()

    @property
    def frame_id(self):
        return self._frame_id

    @property
    def img(self):
        return self._img

    @property
    def positions(self):
        if self._positions is None:
            self._positions = np.transpose((self._sources['xcentroid'], self._sources['ycentroid']))
        return self._positions

    @property
    def bkg_sigma(self):
        return self._bkg_sigma

    @property
    def img_sum(self):
        return self._sum

    @property
    def img_median(self):
        return self._median
