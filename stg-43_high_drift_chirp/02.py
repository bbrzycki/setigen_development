import matplotlib.pyplot as plt

import numpy as np
from astropy import units as u
import blimpy as bl
import scipy.integrate

import setigen as stg


frame = stg.Frame.from_backend_params(1024, 16)
signal = frame.add_signal(stg.squared_path(f_start=frame.get_frequency(200),
                                           drift_rate=0.05*u.Hz/u.s),
                          stg.constant_t_profile(level=1),
                          stg.box_f_profile(width=20*u.Hz),
                          stg.constant_bp_profile(level=1),
                          num_smear=100)
frame.plot()
plt.savefig('02.pdf', bbox_inches='tight')