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


############################################################

def get_unit_drift_rate(raw_voltage_backend,
                        fftlength=1048576,
                        int_factor=1):
    df = raw_voltage_backend.chan_bw / fftlength
    dt = raw_voltage_backend.tbin * fftlength * int_factor
    return df / dt


def get_intensity(snr, 
                  raw_voltage_backend,
                  fftlength=1048576,
                  int_factor=1):
    dt = raw_voltage_backend.tbin * fftlength * int_factor
    tchans = raw_voltage_backend.time_per_block / dt
    
    chi_df = 2 * raw_voltage_backend.num_pols * int_factor
#     main_mean = (raw_voltage_backend.requantizer.target_sigma)**2 * chi_df * raw_voltage_backend.filterbank.max_mean_ratio
    
    I_per_SNR = np.sqrt(2 / chi_df) / tchans**0.5
 
    signal_level = 1 / (raw_voltage_backend.num_branches * fftlength / 4)**0.5 * (snr * I_per_SNR)**0.5 
    return signal_level

############################################################



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


# Params for high res data product
fftlength = 1048576
int_factor = 51


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


# Compute relevant quantities with helper functions above
unit_drift_rate = stg.get_unit_drift_rate(rvb, fftlength, int_factor)
signal_level = stg.get_intensity(10, rvb, fftlength, int_factor)



for stream in antenna.streams:
    stream.add_noise(v_mean=0, 
                     v_std=1)
    
    
    stream.add_constant_signal(f_start=chan_bw / fftlength * int(fftlength*(2+0.3-0.5)), 
                      drift_rate=0*u.Hz/u.s, 
                      level=signal_level)
                      
# #     stream.add_constant_signal(f_start=chan_bw / fftlength * (int(fftlength*(2+0.6-0.5))), 
# #                       drift_rate=0*u.Hz/u.s, 
# #                       level=signal_level)
    for i in range(5):
        stream.add_constant_signal(f_start=chan_bw / fftlength * (int(fftlength*(2+0.3-0.5))+20*i), 
                          drift_rate=0*u.Hz/u.s, 
                          level=signal_level)
                      
    stream.add_constant_signal(f_start=chan_bw / fftlength * (0.1+int(fftlength*(3+0.3-0.5))), 
                      drift_rate=0*u.Hz/u.s, 
                      level=signal_level * 1/np.sinc(0.1))
    stream.add_constant_signal(f_start=chan_bw / fftlength * (20+0.1+int(fftlength*(3+0.3-0.5))), 
                      drift_rate=0*u.Hz/u.s, 
                      level=signal_level)
                      
        
    stream.add_constant_signal(f_start=chan_bw / fftlength * (0.7+int(fftlength*(3+0.6-0.5))), 
                      drift_rate=0*u.Hz/u.s, 
                      level=signal_level * 1/np.sinc(0.3))
    stream.add_constant_signal(f_start=chan_bw / fftlength * (20+0.7+int(fftlength*(3+0.6-0.5))), 
                      drift_rate=0*u.Hz/u.s, 
                      level=signal_level)
                      
    stream.add_constant_signal(f_start=chan_bw / fftlength * int(fftlength*(4+0.2-0.5)), 
                      drift_rate=unit_drift_rate, 
                      level=signal_level)
                      
    stream.add_constant_signal(f_start=chan_bw / fftlength * (0.1+int(fftlength*(4+0.6-0.5))), 
                      drift_rate=unit_drift_rate, 
                      level=signal_level)
                      
    stream.add_constant_signal(f_start=chan_bw / fftlength * int(fftlength*(5+0.2-0.5)), 
                      drift_rate=2*unit_drift_rate, 
                      level=signal_level)
                      
    stream.add_constant_signal(f_start=chan_bw / fftlength * (0.5+int(fftlength*(5+0.6-0.5))), 
                      drift_rate=2*unit_drift_rate, 
                      level=signal_level)
                      
    stream.add_constant_signal(f_start=chan_bw / fftlength * int(fftlength*(7+0.2-0.5)), 
                      drift_rate=4*unit_drift_rate, 
                      level=signal_level)
                      
    stream.add_constant_signal(f_start=chan_bw / fftlength * (0.5+int(fftlength*(7+0.6-0.5))), 
                      drift_rate=4*unit_drift_rate, 
                      level=signal_level)
                      
    stream.add_constant_signal(f_start=chan_bw / fftlength * int(fftlength*(9+0.2-0.5)), 
                      drift_rate=8*unit_drift_rate, 
                      level=signal_level)
                      
    stream.add_constant_signal(f_start=chan_bw / fftlength * (0.5+int(fftlength*(9+0.6-0.5))), 
                      drift_rate=8*unit_drift_rate, 
                      level=signal_level)



# Record to file
rvb.record(raw_file_stem='/datax/scratch/bbrzycki/data/raw_files/test_snr_actual_5min',
           obs_length=300, 
           length_mode='obs_length',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'})

print(time.time() - start)

print(rvb.total_obs_num_samples)

print(rvb.sample_stage, rvb.digitizer_stage, rvb.filterbank_stage, rvb.requantizer_stage)