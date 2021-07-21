import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import matplotlib.pyplot as plt

import numpy as np
from astropy import units as u
import blimpy as bl

import sys
sys.path.insert(0, "/mnt_home/bryanb/setigen/")
import setigen as stg

sample_rate = 3e9
num_taps = 8
num_branches = 1024

fftlength = 1048576
int_factor = 51

num_blocks = 128

start_chan = 0
input_file_stem = '/datax/scratch/bbrzycki/data/raw_files/blc00_guppi_59114_15868_TMC1_0010'

raw_params = stg.voltage.get_raw_params(input_file_stem=input_file_stem,
                                        start_chan=start_chan)
print(raw_params)

antenna = stg.voltage.Antenna(sample_rate=sample_rate,
                              **raw_params)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

rvb = stg.voltage.RawVoltageBackend.from_data(input_file_stem=input_file_stem,
                                              antenna_source=antenna,
                                              filterbank=filterbank,
                                              start_chan=start_chan,
                                              num_subblocks=32)

signal_level = stg.voltage.get_level(snr=1e6, 
                                     raw_voltage_backend=rvb,
                                     fftlength=fftlength,
                                     num_blocks=num_blocks,
                                     length_mode='num_blocks')

frequencies = raw_params['fch1'] + (raw_params['ascending'] * 2 - 1) * np.linspace(3.1e6, 3.9e6, 9)
for f_start in frequencies:
    leakage_factor = stg.voltage.get_leakage_factor(f_start, rvb, fftlength)
    print(f'{f_start/1e6:.4f} MHz leakage factor: {leakage_factor:.3f}')
    
    for stream in antenna.streams:
        # There's actually no noise present, but we set levels assuming a background noise_std of 1
        level = 1 * leakage_factor * signal_level
        stream.add_constant_signal(f_start=f_start, 
                                   drift_rate=0*u.Hz/u.s, 
                                   level=level)

rvb.record(output_file_stem=f"{input_file_stem}_injected",
           num_blocks=num_blocks, 
           length_mode='num_blocks',
           header_dict={'TELESCOP': 'GBT'},
           digitize=False,
           verbose=False)