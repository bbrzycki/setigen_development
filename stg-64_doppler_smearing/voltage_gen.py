import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt

import numpy as np
from astropy import units as u
import blimpy as bl

import setigen as stg


subsample_factor = 4

sample_rate = 3e9 // subsample_factor
num_taps = 8
num_branches = 1024 // subsample_factor
print(f'Max "num_chans" is {num_branches // 2}.')

fftlength = 1048576
int_factor = 51
num_blocks = 1

digitizer = stg.voltage.RealQuantizer(target_fwhm=8,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=8,
                                           num_bits=8)

antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                              fch1=6*u.GHz,
                              ascending=True,
                              num_pols=2)

block_size = stg.voltage.get_block_size(num_antennas=1,
                                        tchans_per_block=1,
                                        num_bits=8,
                                        num_pols=2,
                                        num_branches=num_branches,
                                        num_chans=num_branches//2,
                                        fftlength=fftlength,
                                        int_factor=int_factor)
block_size = 134217728

# rvb = stg.voltage.RawVoltageBackend(antenna,
#                                     digitizer=digitizer,
#                                     filterbank=filterbank,
#                                     requantizer=requantizer,
#                                     start_chan=0,
#                                     num_chans=num_branches//2,
#                                     block_size=block_size,
#                                     blocks_per_file=128,
#                                     num_subblocks=32)
rvb = stg.voltage.RawVoltageBackend(antenna,
                                    digitizer=digitizer,
                                    filterbank=filterbank,
                                    requantizer=requantizer,
                                    start_chan=0,
                                    num_chans=64,
                                    block_size=block_size,
                                    blocks_per_file=128,
                                    num_subblocks=32)

# Add noise
for stream in antenna.streams:
    stream.add_noise(v_mean=0, 
                     v_std=1)
    
# Add signals
signal_level = stg.voltage.get_level(100, 
                                     rvb,
                                     obs_length=300, 
                                     length_mode='obs_length',
                                     fftlength=fftlength)

unit_drift_rate = stg.voltage.get_unit_drift_rate(rvb, fftlength, int_factor)

chan_bw = sample_rate / num_branches
df = np.abs(chan_bw / fftlength)
fch1=6e9
for i, f_start in enumerate(np.linspace(6003.1e6, 6003.9e6, 3)):
    f_start = fch1 + int((f_start - fch1) / df) * df
    # leakage_factor = stg.voltage.get_leakage_factor(f_start, rvb, fftlength)
    leakage_factor = 1
    for stream in antenna.streams:
        level = stream.get_total_noise_std() * leakage_factor * signal_level
        stream.add_constant_signal(f_start=f_start, 
                                   drift_rate=i * unit_drift_rate, 
                                   level=level)
        stream.add_constant_signal(f_start=f_start + 16 * df, 
                                   drift_rate=i * unit_drift_rate, 
                                   level=level)
        stream.add_constant_signal(f_start=f_start + 32 * df, 
                                   drift_rate=i * unit_drift_rate, 
                                   level=level)
        stream.add_constant_signal(f_start=f_start + 48 * df + df/2, 
                                   drift_rate=i * unit_drift_rate, 
                                   level=level)
        stream.add_constant_signal(f_start=f_start + 64 * df + df/2, 
                                   drift_rate=i * unit_drift_rate, 
                                   level=level)
        stream.add_constant_signal(f_start=f_start + 80 * df + df/2, 
                                   drift_rate=i * unit_drift_rate, 
                                   level=level)
    
DATA_DIR = '/datax/scratch/bbrzycki/data/raw_files/'
rvb.record(output_file_stem=f'{DATA_DIR}/doppler_smear_test_gpu_offset',
           obs_length=300, 
           length_mode='obs_length',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'},
           verbose=False)