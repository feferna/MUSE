# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2018-present,
# User Interfaces group, Aalto University, Finland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Third-party component used in this project:
#   WAVE (Weighted Affective Valence Estimates) metric
#   Origin: https://github.com/aalto-ui/aim
#   Local changes: packaging-only (single-file module); algorithm unchanged.

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

# Standard library modules
import base64
from io import BytesIO
from typing import Dict, List, Optional, Union

# Third-party modules
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------
# Metric
# ----------------------------------------------------------------------------


class Metric():
    """
    Metric: Average WAVE score (Weighted Affective Valence Estimates).
    """

    # Private constants

    # WAVE scores are from Fig. 1 in the [1]. There are a total of 37 colors:
    # For the 32 chromatic colors, the first letter is the
    # saturation/lightness level (S[aturated], L[ight], M[uted], or D[ark])
    # and the 2nd letter is the hue (R[ed], O[range], Y[ellow],
    # [c]H[artreuse], G[reen], C[yan], B[lue], and P[urple])
    # 5 achromatic colors, BK = Black, A1 = dark gray, A2 = medium gray,
    # A3 = light gray, and WH = white are excluded from the WAVE scores.
    # Related source: https://github.com/SchlossVRL/coloremoblues
    # Related source 2: https://github.com/SchlossVRL/ColorConceptAssociations

    _WAVE_COLORS: Dict = {
        (235, 45, 92): {
            "Level": "Saturated",
            "Hue": "Red",
            "Abbreviation": "SR",
            "Score": 0.5488069414316703,
        },
        (242, 149, 185): {
            "Level": "Light",
            "Hue": "Red",
            "Abbreviation": "LR",
            "Score": 0.4577006507592191,
        },
        (204, 119, 141): {
            "Level": "Muted",
            "Hue": "Red",
            "Abbreviation": "MR",
            "Score": 0.4859002169197397,
        },
        (162, 32, 66): {
            "Level": "Dark",
            "Hue": "Red",
            "Abbreviation": "DR",
            "Score": 0.8481561822125814,
        },
        (243, 145, 51): {
            "Level": "Saturated",
            "Hue": "Orange",
            "Abbreviation": "SO",
            "Score": 0.7114967462039046,
        },
        (251, 200, 166): {
            "Level": "Light",
            "Hue": "Orange",
            "Abbreviation": "LO",
            "Score": 0.3741865509761389,
        },
        (208, 154, 119): {
            "Level": "Muted",
            "Hue": "Orange",
            "Abbreviation": "MO",
            "Score": 0.39154013015184386,
        },
        (159, 90, 48): {
            "Level": "Dark",
            "Hue": "Orange",
            "Abbreviation": "DO",
            "Score": 0.18329718004338397,
        },
        (253, 228, 51): {
            "Level": "Saturated",
            "Hue": "Yellow",
            "Abbreviation": "SY",
            "Score": 0.7201735357917572,
        },
        (252, 232, 158): {
            "Level": "Light",
            "Hue": "Yellow",
            "Abbreviation": "LY",
            "Score": 0.5140997830802604,
        },
        (218, 198, 118): {
            "Level": "Muted",
            "Hue": "Yellow",
            "Abbreviation": "MY",
            "Score": 0.49132321041214755,
        },
        (162, 149, 59): {
            "Level": "Dark",
            "Hue": "Yellow",
            "Abbreviation": "DY",
            "Score": 0.0,
        },
        (179, 208, 68): {
            "Level": "Saturated",
            "Hue": "cHartreuse",
            "Abbreviation": "SH",
            "Score": 0.4652928416485901,
        },
        (224, 231, 153): {
            "Level": "Light",
            "Hue": "cHartreuse",
            "Abbreviation": "LH",
            "Score": 0.2928416485900217,
        },
        (177, 200, 101): {
            "Level": "Muted",
            "Hue": "cHartreuse",
            "Abbreviation": "MH",
            "Score": 0.33731019522776573,
        },
        (126, 152, 68): {
            "Level": "Dark",
            "Hue": "cHartreuse",
            "Abbreviation": "DH",
            "Score": 0.3579175704989154,
        },
        (101, 190, 131): {
            "Level": "Saturated",
            "Hue": "Green",
            "Abbreviation": "SG",
            "Score": 0.648590021691974,
        },
        (193, 224, 196): {
            "Level": "Light",
            "Hue": "Green",
            "Abbreviation": "LG",
            "Score": 0.46095444685466386,
        },
        (129, 199, 144): {
            "Level": "Muted",
            "Hue": "Green",
            "Abbreviation": "MG",
            "Score": 0.5726681127982647,
        },
        (37, 152, 114): {
            "Level": "Dark",
            "Hue": "Green",
            "Abbreviation": "DG",
            "Score": 0.7125813449023862,
        },
        (86, 197, 208): {
            "Level": "Saturated",
            "Hue": "Cyan",
            "Abbreviation": "SC",
            "Score": 0.8297180043383949,
        },
        (164, 219, 228): {
            "Level": "Light",
            "Hue": "Cyan",
            "Abbreviation": "LC",
            "Score": 0.7028199566160521,
        },
        (133, 204, 208): {
            "Level": "Muted",
            "Hue": "Cyan",
            "Abbreviation": "MC",
            "Score": 0.5932754880694144,
        },
        (24, 155, 154): {
            "Level": "Dark",
            "Hue": "Cyan",
            "Abbreviation": "DC",
            "Score": 0.6377440347071583,
        },
        (96, 163, 215): {
            "Level": "Saturated",
            "Hue": "Blue",
            "Abbreviation": "SB",
            "Score": 1.0,
        },
        (170, 194, 228): {
            "Level": "Light",
            "Hue": "Blue",
            "Abbreviation": "LB",
            "Score": 0.7537960954446855,
        },
        (124, 159, 201): {
            "Level": "Muted",
            "Hue": "Blue",
            "Abbreviation": "MB",
            "Score": 0.8318872017353579,
        },
        (59, 125, 181): {
            "Level": "Dark",
            "Hue": "Blue",
            "Abbreviation": "DB",
            "Score": 0.7396963123644252,
        },
        (156, 78, 155): {
            "Level": "Saturated",
            "Hue": "Purple",
            "Abbreviation": "SP",
            "Score": 0.6843817787418656,
        },
        (184, 158, 199): {
            "Level": "Light",
            "Hue": "Purple",
            "Abbreviation": "LP",
            "Score": 0.63882863340564,
        },
        (162, 115, 167): {
            "Level": "Muted",
            "Hue": "Purple",
            "Abbreviation": "MP",
            "Score": 0.7451193058568331,
        },
        (115, 56, 145): {
            "Level": "Dark",
            "Hue": "Purple",
            "Abbreviation": "DP",
            "Score": 0.8080260303687636,
        },
    }

    _MATCH_COLORS: List = list(_WAVE_COLORS.keys())
    _NUM_MATCH_COLORS: int = len(_MATCH_COLORS)

    # Public methods
    @classmethod
    def execute_metric(
        cls,
        gui_image: str,
    ) -> Optional[List[Union[int, float, str]]]:
        """
        Execute the metric.

        Args:
            gui_image: GUI image (PNG) encoded in Base64

        Kwargs:
            gui_type: GUI type, desktop = 0 (default), mobile = 1
            gui_segments: GUI segments (defaults to None)
            gui_url: GUI URL (defaults to None)

        Returns:
            Results (list of measures)
            - Average WAVE score across pixels (float, [0, 1))
        """
        try:
            # Create PIL image
            img: Image.Image = Image.open(BytesIO(base64.b64decode(gui_image)))

            # Convert image from ??? (e.g., RGBA) to RGB color space
            img_rgb: Image.Image = img.convert("RGB")

            # Get NumPy array
            img_rgb_nparray: np.ndarray = np.array(img_rgb)

            ax1, ax2, _ = img_rgb_nparray.shape

            # Repeated values for every possible match color (cls._MATCH_COLORS)
            repeated_img_nparray: np.ndarray = np.tile(
                img_rgb_nparray, cls._NUM_MATCH_COLORS
            ).reshape(ax1, ax2, cls._NUM_MATCH_COLORS, 3)

            # Find colors most closely matching cls._MATCH_COLORS across pixels.
            l2_norms: np.ndarray = (
                (repeated_img_nparray - np.array(cls._MATCH_COLORS)) ** 2
            ).sum(axis=3)
            match_indices = l2_norms.argmin(axis=2).flatten()

            # Mean over matched colors (size of the image) weight values.
            wave_values: List = [
                cls._WAVE_COLORS[cls._MATCH_COLORS[i]]["Score"]
                for i in match_indices
            ]
            wave_mean: float = float(np.mean(wave_values))

            return [wave_mean]
        except Exception:
            # Return explicit None to indicate failure to compute
            return [None]
