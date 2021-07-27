#!/usr/bin/env python3

import os
import sys
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from unet.dataset import read_line_gap_infos

if len(sys.argv) != 2:
    print("Usage: analyze_tessellations.py <path/to/tessellations>")
    sys.exit(1)

basepath = sys.argv[1]
with open(os.path.join(basepath, "params.json")) as f:
    img_size = json.load(f)["width"]

gaps, lines = read_line_gap_infos(basepath, img_size)

# number of gaps distribution
plt.figure()
plt.hist(lines.n_gaps, bins=np.arange(7))
plt.title("n_gaps")

# line strength distribution
plt.figure()
plt.hist(lines.strength)
plt.title("line_strength")

# line width distribution
plt.figure()
plt.hist(lines.width, bins=np.arange(10))
plt.title("line_width")

# relative gap length distribution
plt.figure()
plt.hist(gaps.length, bins=np.linspace(0, 1, 20))
plt.title("gap_length\n(relative to line length)")

# plot gap position
plt.figure()
plt.hist(gaps.position, bins=np.linspace(0, 1, 20))
plt.title("gap_position")

# plot gap length distribution
plt.figure()
plt.hist(gaps.length_manh, bins=np.logspace(0, 2.0, 50))
plt.title("gap length\nmanhattan distance")
plt.xscale("log")


plt.show()
