import numpy as np
import blimpy as bl
import pandas as pd
from astropy import units as u

try:
    import cupy as xp
except ImportError:
    import numpy as xp
    

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


chan_bw = sample_rate/num_branches

digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                           num_bits=8)


num_pols = 2

num_blocks = 4
fftlength = 1024
int_factor = 8




antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                              fch1=0,
                              ascending=True,
                              num_pols=num_pols)


rvb = stg.voltage.RawVoltageBackend(antenna,
                                    digitizer=digitizer,
                                    filterbank=filterbank,
                                    requantizer=requantizer,
                                    start_chan=0,
                                    num_chans=64,
                                    block_size=134217728,
                                    blocks_per_file=128,
                                    num_subblocks=32)

# Compute relevant quantities with helper functions above
unit_drift_rate = stg.get_unit_drift_rate(rvb, fftlength, int_factor)
signal_level = stg.get_intensity(10, 
                                 rvb,
                                 num_blocks=num_blocks,
                                 length_mode='num_blocks',
                                 fftlength=fftlength)


rand_array = np.random.random(5)
for stream in antenna.streams:
    stream.add_noise(v_mean=0, 
                     v_std=1)
    
    level = stream.get_total_noise_std() * signal_level

    for i in range(5):
        rand = rand_array[i]
        f = chan_bw / fftlength * (int(fftlength*(2+0.3-0.5))+rand+20*i)
        stream.add_constant_signal(f_start=f, 
                          drift_rate=0*u.Hz/u.s, 
                          level=level)
    
    for i in range(5):
        rand = rand_array[i]
        f = chan_bw / fftlength * (int(fftlength*(3+0.3-0.5))+rand+20*i)
        leakage_factor = stg.voltage.get_leakage_factor(f, rvb, fftlength)
        print(rand, leakage_factor)
        stream.add_constant_signal(f_start=f, 
                          drift_rate=0*u.Hz/u.s, 
                          level=level * leakage_factor)
        
    for i in range(5):
        f = chan_bw / fftlength * (int(fftlength*(4+0.3-0.5))+20*i)
        leakage_factor = stg.voltage.get_leakage_factor(f, rvb, fftlength)
        print(rand, leakage_factor)
        stream.add_constant_signal(f_start=f, 
                          drift_rate=0*u.Hz/u.s, 
                          level=level)
        
        



# Record to file
rvb.record(raw_file_stem='/datax/scratch/bbrzycki/data/raw_files/test_snr_actual',
           num_blocks=num_blocks, 
           length_mode='num_blocks',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'})

print(time.time() - start)

print(rvb.total_obs_num_samples)