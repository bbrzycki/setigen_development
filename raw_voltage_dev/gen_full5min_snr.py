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

chan_bw = sample_rate/num_branches

antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                              fch1=6*u.GHz,
                              ascending=False,
                              num_pols=2)

for stream in antenna.x, antenna.y:
    stream.add_noise(v_mean=0, 
                     v_std=1)
    stream.add_signal(f_start=(6e9 - chan_bw * 10.1), 
                        drift_rate=0*u.Hz/u.s, 
                        snr=5,
                        mode='snr')
    stream.add_signal(f_start=(6e9 - chan_bw * 10.2), 
                        drift_rate=0*u.Hz/u.s, 
                        snr=10,
                        mode='snr')
    stream.add_signal(f_start=(6e9 - chan_bw * 10.3), 
                        drift_rate=0*u.Hz/u.s, 
                        snr=15,
                        mode='snr')
    stream.add_signal(f_start=(6e9 - chan_bw * 10.4), 
                        drift_rate=0*u.Hz/u.s, 
                        snr=20,
                        mode='snr')
    
    stream.add_signal(f_start=(6e9 - chan_bw * 20.1), 
                        drift_rate=0*u.Hz/u.s, 
                        snr=50,
                        mode='snr')
    stream.add_signal(f_start=(6e9 - chan_bw * 20.2), 
                        drift_rate=0*u.Hz/u.s, 
                        snr=100,
                        mode='snr')
    stream.add_signal(f_start=(6e9 - chan_bw * 20.3), 
                        drift_rate=0*u.Hz/u.s, 
                        snr=150,
                        mode='snr')
    stream.add_signal(f_start=(6e9 - chan_bw * 20.4), 
                        drift_rate=0*u.Hz/u.s, 
                        snr=200,
                        mode='snr')
    
    stream.add_signal(f_start=(6e9 - chan_bw * 40.4), 
                        drift_rate=0*u.Hz/u.s, 
                        snr=1000,
                        mode='snr')



digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                           num_bits=8)

rvp = stg.voltage.RawVoltageBackend(antenna,
                                     block_size=134217728,
                                     blocks_per_file=128,
                                     digitizer=digitizer,
                                     filterbank=filterbank,
                                     requantizer=requantizer)

rvp.record(raw_file_stem='/datax/scratch/bbrzycki/data/raw_files/test_5min_snr',
           obs_length=300, 
           start_chan=0,
           num_chans=64,
           num_subblocks=32,
           length_mode='obs_length',
           header_dict={'HELLO': 'test_value'})

print(time.time() - start)