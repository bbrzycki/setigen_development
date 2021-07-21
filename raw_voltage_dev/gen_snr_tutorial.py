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



decimation_factor = 128
sample_rate = int(3e9 / decimation_factor)
num_taps = 8
num_branches = int(1024 / decimation_factor)

# Params for high res data product
obs_length = 300
fftlength = 1048576
int_factor = 51

chan_bw = sample_rate/num_branches

digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                           num_bits=8)


antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                              fch1=0*u.GHz,
                              ascending=True,
                              num_pols=2)


rvb = stg.voltage.RawVoltageBackend(antenna,
                                    digitizer=digitizer,
                                    filterbank=filterbank,
                                    requantizer=requantizer,
                                    start_chan=0,
                                    num_chans=4,
                                    block_size=134217728//16,
                                    blocks_per_file=128,
                                    num_subblocks=16)


for stream in antenna.streams:
    stream.add_noise(v_mean=0, 
                     v_std=1)
    
    
signal_level = stg.voltage.get_intensity(snr=100, 
                                         raw_voltage_backend=rvb,
                                         obs_length=obs_length,
                                         length_mode='obs_length',
                                         fftlength=fftlength)


for f_start in np.linspace(3.1e6, 3.9e6, 9):
    leakage_factor = stg.voltage.get_leakage_factor(f_start, rvb, fftlength)
    print(f'{f_start/1e6} MHz leakage factor: {leakage_factor:.3f}')
    
    for stream in antenna.streams:
        level = stream.get_total_noise_std() * leakage_factor * signal_level
        antenna.x.add_constant_signal(f_start=f_start, 
                                      drift_rate=0*u.Hz/u.s, 
                                      level=level)

for stream in antenna.streams:
    level = stream.get_total_noise_std() * signal_level
    
    for i in range(10):
        f = chan_bw / fftlength * (int(fftlength*(2+0.3-0.5)) + 32*i)
        stream.add_constant_signal(f_start=f, 
                          drift_rate=0, 
                          level=level)
        

# Record to file
rvb.record(raw_file_stem='/datax/scratch/bbrzycki/data/raw_files/test_snr_tutorial',
           obs_length=300, 
           length_mode='obs_length',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'})

print(time.time() - start)

print(rvb.total_obs_num_samples)

print(rvb.sample_stage, rvb.digitizer_stage, rvb.filterbank_stage, rvb.requantizer_stage)




