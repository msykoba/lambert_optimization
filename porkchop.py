# -*- coding: utf-8 -*-

import astropy.units as u
import matplotlib.pyplot as plt

from poliastro.plotting.porkchop import PorkchopPlotter
from poliastro.bodies import Earth, Mars
from poliastro.util import time_range

launch_span = time_range("2024-08-01", end="2024-12-01")
arrival_span = time_range("2025-05-01", end="2025-11-01")
fig, ax = plt.subplots(figsize=(15, 15))
ax.set_aspect('equal')

plotter = PorkchopPlotter(
    Earth, 
    Mars,
    launch_span, 
    arrival_span, 
    ax, 
    tfl=False, 
    vhp=True, 
    max_c3 =30 * u.km**2 / u.s **2, 
    max_vhp=3.5 * u.km / u.s
)
plotter.porkchop()
plt.savefig('./img/porkchop.png')
plt.close()
