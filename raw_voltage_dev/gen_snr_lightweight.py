import numpy as np
import blimpy as bl
import pandas as pd
from astropy import units as u

try:
    import cupy as xp
except ImportError:
    import numpy as xp
    
import time
from astropy.stats import sigma_clip

from scipy.signal import butter, lfilter, filtfilt
import scipy.signal

import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
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
                                      num_bits=8, 
                                      stats_calc_freq=-1,
                                      stats_calc_num_samples=10000)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                           num_bits=8, 
                                           stats_calc_freq=-1,
                                           stats_calc_num_samples=10000)


antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                              fch1=0,
                              ascending=True,
                              num_pols=2)


rvb = stg.voltage.RawVoltageBackend(antenna,
                                    digitizer=digitizer,
                                    filterbank=filterbank,
                                    requantizer=requantizer,
                                    start_rec_chan=0,
                                    num_rec_chans=4,
                                    block_size=134217728//16,
                                    blocks_per_file=128,
                                    num_subblocks=16)


# Compute relevant quantities with helper functions above
unit_drift_rate = stg.get_unit_drift_rate(rvb, fftlength, int_factor)
signal_level = stg.get_intensity(100, 
                                 rvb,
                                 obs_length=obs_length,
                                 length_mode='obs_length',
                                 fftlength=fftlength)

# for i, stream in enumerate(antenna.streams):
#     if i == 0:
#         noise_std = 2
#     else:
#         noise_std = 1
        
#     stream.add_noise(v_mean=0, 
#                      v_std=noise_std)
    
# #     level = signal_level * noise_std**1
#     level = stream.get_total_noise_std() * signal_level

#     fine_chan_bw = chan_bw / fftlength
#     stream.add_constant_signal(f_start=fine_chan_bw * int(fftlength*(2+0.3-0.5)), 
#                       drift_rate=0*u.Hz/u.s, 
#                       level=level)

#     for i in range(5):
#         stream.add_constant_signal(f_start=chan_bw / fftlength * (int(fftlength*(2+0.3-0.5))+20*i), 
#                           drift_rate=0*u.Hz/u.s, 
#                           level=level*(1 + i)**0.5)

rand_array = np.random.random(5)
for stream in antenna.streams:
    stream.add_noise(v_mean=0, 
                     v_std=1)
    
#     level = stream.get_total_noise_std() * signal_level
    
#     for i in range(64):
#         f = chan_bw / fftlength * (int(fftlength*(2+0.3-0.5)) + (i//4+16)*i)
#         leakage_factor = stg.get_leakage_factor(f, rvb, fftlength)
#         stream.add_constant_signal(f_start=f, 
#                           drift_rate=1/64 * i * unit_drift_rate, 
#                           level=level*1)
        
        

# Record to file
rvb.record(raw_file_stem='/datax/scratch/bbrzycki/data/raw_files/test_snr_lightweight',
           obs_length=100, 
           length_mode='obs_length',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'})

print(time.time() - start)

print(rvb.total_obs_num_samples)

print(rvb.sample_stage, rvb.digitizer_stage, rvb.filterbank_stage, rvb.requantizer_stage)






# signal_level = stg.voltage.get_intensity(snr=20, 
#                                          raw_voltage_backend=RawVoltageBackend(),
#                                          obs_length=300,
#                                          length_mode='obs_length',
#                                          fftlength=fftlength, 
#                                          int_factor=int_factor)


# my_frequency = x

# leakage_factor = stg.voltage.get_leakage_factor(my_frequency, raw_voltage_backend)

# level = signal_level * stream.get_total_noise_std() * leakage_factor

# stream.add_constant_signal(f_start=my_frequency, 
#                            drift_rate=0, 
#                            level=signal_level * leakage_factor)

