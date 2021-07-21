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
fftlength = 1024

chan_bw = sample_rate/num_branches

digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                           num_bits=8)




num_pols = 2
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
                                    num_subblocks=16)

signal_level = stg.get_intensity(10, 
                                 rvb,
                                 num_blocks=1, 
                                 length_mode='num_blocks',
                                 fftlength=fftlength, 
                                 int_factor=1)

for stream in antenna.streams:
    stream.add_noise(v_mean=0, 
                     v_std=1)
    
    stream.add_constant_signal(f_start=chan_bw / fftlength * int((2.4-0.5)*fftlength), 
                      drift_rate=0*u.Hz/u.s, 
                      level=signal_level * stream.noise_std**0.5)
                      
print(f'frequency is {chan_bw / fftlength * int(2.2*fftlength)}')


# Record to file
rvb.record(raw_file_stem='/datax/scratch/bbrzycki/data/raw_files/test_lower_sampling_0',
           num_blocks=1, 
           length_mode='num_blocks',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'})

print(rvb.sample_stage, rvb.digitizer_stage, rvb.filterbank_stage, rvb.requantizer_stage)

############################

sample_rate = int(3e9 // 8)
num_taps = 8
num_branches = int(1024 // 8)
fftlength = 1024

chan_bw = sample_rate/num_branches

digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                           num_bits=8)


num_pols = 2
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
                                    num_subblocks=16)


signal_level = stg.get_intensity(10, 
                                 rvb,
                                 num_blocks=1,
                                 length_mode='num_blocks',
                                 fftlength=fftlength, 
                                 int_factor=1)

for stream in antenna.streams:
    stream.add_noise(v_mean=0, 
                     v_std=1)
    
    stream.add_constant_signal(f_start=chan_bw / fftlength * int((2.4-0.5)*fftlength), 
                      drift_rate=0*u.Hz/u.s, 
                      level=signal_level * stream.noise_std**0.5)
                      
print(f'frequency is {chan_bw / fftlength * int(2.2*fftlength)}')


# Record to file
rvb.record(raw_file_stem='/datax/scratch/bbrzycki/data/raw_files/test_lower_sampling_1',
           num_blocks=1, 
           length_mode='num_blocks',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'})

print(rvb.sample_stage, rvb.digitizer_stage, rvb.filterbank_stage, rvb.requantizer_stage)