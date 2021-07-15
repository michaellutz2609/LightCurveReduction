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

from astropy.table import Table

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def no_classifier(sources: Table):
    logger.info("Default (dummy) classifier implementation: No classification!")
    sources['satellite'] = '-'
    pass


def classify_moons(sources: Table):
    logger.info("Classifier implementation: classify_moons")
    logger.info(f"Working on {len(sources)} identified objects.")
    sources['satellite'] = '-'
    for src in sources:
        dist = math.sqrt(src['xcentroid'] ** 2 + src['ycentroid'] ** 2)
        if 90 < dist < 170:
            src['satellite'] = 'a'
        elif 330 < dist < 390:
            src['satellite'] = 'b'
        logger.info(f"Identified object {src['satellite']}.")

def classify_moons_01(sources: Table):
    logger.info("Classifier implementation: classify_moons")
    logger.info(f"Working on {len(sources)} identified objects.")
    sources['satellite'] = '-'
    for src in sources:
        # dist = math.sqrt(src['xcentroid'] ** 2 + src['ycentroid'] ** 2)
        if 300 < src['xcentroid'] < 350 and 290 < src['ycentroid'] < 330:
            src['satellite'] = 'a'
        elif 220 < src['xcentroid'] < 250 and 410 < src['ycentroid'] < 440:
            src['satellite'] = 'b'
        logger.info(f"Identified object {src['satellite']}.")

def classify_moons_02(sources: Table):
    logger.info("Classifier implementation: classify_moons")
    logger.info(f"Working on {len(sources)} identified objects.")
    sources['satellite'] = '-'
    for src in sources:
        # dist = math.sqrt(src['xcentroid'] ** 2 + src['ycentroid'] ** 2)
        if 0 < src['xcentroid'] < 340:
            src['satellite'] = 'a'
        elif 350 < src['xcentroid'] < 750:
            src['satellite'] = 'b'
        logger.info(f"Identified object {src['satellite']}.")
