import numpy as np
import blimpy as bl
import pandas as pd
from astropy import units as u

import sys, os, glob, errno
import csv
import json
import h5py
import time
from astropy.stats import sigma_clip


from scipy.signal import butter, lfilter, filtfilt
import scipy.signal

sys.path.insert(0, "/home/bryanb/setigen/")
import setigen as stg

def db(x):
    """ Convert linear value to dB value """
    return 10*np.log10(x)


start = time.time()

sample_rate = 3e9
num_taps = 8
num_branches = 1024

antenna = stg.voltage.Antenna(sample_rate=sample_rate, num_pols=2)

antenna.x.add_noise(v_mean=0, 
                    v_std=1)
antenna.x.add_signal(f_start=(sample_rate/num_branches * (204.2))*u.Hz, 
                    drift_rate=50*u.Hz/u.s, 
                    level=0.1)
# antenna.x.add_signal(f_start=(sample_rate/num_branches * (204.1))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=0.1)
# antenna.x.add_signal(f_start=(sample_rate/num_branches * (203.2))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=0.1)
# antenna.x.add_signal(f_start=(sample_rate/num_branches * (5.2))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=1)
# antenna.x.add_signal(f_start=(sample_rate/num_branches * (4.2))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=1)
# antenna.x.add_signal(f_start=(sample_rate/num_branches * (3.2))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=1)
# antenna.x.add_signal(f_start=(sample_rate/num_branches * (2.2))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=1)

antenna.y.add_noise(v_mean=0, 
                    v_std=1)
antenna.y.add_signal(f_start=(sample_rate/num_branches * (204.2))*u.Hz, 
                    drift_rate=50*u.Hz/u.s, 
                    level=0.1)
# antenna.y.add_signal(f_start=(sample_rate/num_branches * (204.1))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=0.1)
# antenna.y.add_signal(f_start=(sample_rate/num_branches * (5.2))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=1)
# antenna.y.add_signal(f_start=(sample_rate/num_branches * (4.2))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=1)
# antenna.y.add_signal(f_start=(sample_rate/num_branches * (3.2))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=1)
# antenna.y.add_signal(f_start=(sample_rate/num_branches * (2.2))*u.Hz, 
#                     drift_rate=50*u.Hz/u.s, 
#                     level=1)

digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                           num_bits=4)

backend = stg.voltage.RawVoltageBackend(antenna,
                                     block_size=134217728,
                                     blocks_per_file=128,
                                     digitizer=digitizer,
                                     filterbank=filterbank,
                                     requantizer=requantizer)

backend.record(raw_file_stem='test_4bit',
           num_blocks=1, 
           start_chan=192,
           num_chans=64,
           num_subblocks=32,
           length_mode='num_blocks',
           header_dict={'HELLO': 'test_value'})

# backend.record(raw_file_stem='test_4bit',
#            num_blocks=4, 
#            start_chan=0,
#            num_chans=64,
#            num_subblocks=32,
#            length_mode='num_blocks',
#            header_dict={'HELLO': 'test_value'})

print(time.time() - start)