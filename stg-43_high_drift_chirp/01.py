import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt

import numpy as np
import cupy as cp
from astropy import units as u
import blimpy as bl

import setigen as stg

subsample_factor = 128

sample_rate = 3e9 // subsample_factor
num_taps = 8
num_branches = 1024 // subsample_factor

fftlength = 1048576
int_factor = 51
obs_length = 300

chan_bw = sample_rate / num_branches

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

# block_size = stg.voltage.get_block_size(num_antennas=1,
#                                         tchans_per_block=128,
#                                         num_bits=8,
#                                         num_pols=2,
#                                         num_branches=num_branches,
#                                         num_chans=num_branches//2,
#                                         fftlength=fftlength,
#                                         int_factor=int_factor)
rvb = stg.voltage.RawVoltageBackend(antenna,
                                    digitizer=digitizer,
                                    filterbank=filterbank,
                                    requantizer=requantizer,
                                    start_chan=0,
                                    num_chans=4,
                                    block_size=134217728//16,
                                    blocks_per_file=128,
                                    num_subblocks=32)

for stream in antenna.streams:
    stream.add_noise(v_mean=0, 
                     v_std=1)

signal_level = stg.voltage.get_level(snr=100, 
                                     raw_voltage_backend=rvb,
                                     obs_length=obs_length,
                                     length_mode='obs_length',
                                     fftlength=fftlength)

drift_rate = stg.voltage.get_unit_drift_rate(rvb, 
                                             fftlength, 
                                             int_factor)


for stream in antenna.streams:
    level = stream.get_total_noise_std() * signal_level
    level = signal_level
    
    fine_chan_bw = chan_bw / fftlength
    
    f_start = fine_chan_bw * int(fftlength*(2+0.20-0.5))
    stream.add_constant_signal(f_start=f_start, 
                               drift_rate=drift_rate, 
                               level=level)
    f_start = fine_chan_bw * int(fftlength*(2+0.21-0.5))
    stream.add_constant_signal(f_start=f_start, 
                               drift_rate=2*drift_rate, 
                               level=level)
    f_start = fine_chan_bw * int(fftlength*(2+0.22-0.5))
    stream.add_constant_signal(f_start=f_start, 
                               drift_rate=3*drift_rate, 
                               level=level)
    f_start = fine_chan_bw * int(fftlength*(2+0.23-0.5))
    stream.add_constant_signal(f_start=f_start, 
                               drift_rate=4*drift_rate, 
                               level=level)
    f_start = fine_chan_bw * int(fftlength*(2+0.24-0.5))
    stream.add_constant_signal(f_start=f_start, 
                               drift_rate=8*drift_rate, 
                               level=level)
    
rvb.record(output_file_stem='drifting_100snr',
           obs_length=obs_length, 
           length_mode='obs_length',
           header_dict={'TELESCOP': 'GBT'},
           verbose=False)

