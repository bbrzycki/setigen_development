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


def get_intensity(snr, sigma, num_samples):
    return snr * sigma / np.sqrt(num_samples)

def get_num_samples(num_blocks,
                    block_size,
                    num_antennas,
                    num_chans,
                    num_bits,
                    num_pols,
                    num_branches):
    pass #wtf


start = time.time()






sample_rate = 3e9
num_taps = 8
num_branches = 1024

chan_bw = sample_rate/num_branches



antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                              fch1=6*u.GHz,
                              ascending=True,
                              num_pols=1)

antenna.x.add_noise(v_mean=0, 
                    v_std=1)

# antenna.y.add_noise(v_mean=0, 
#                     v_std=1)



# for stream in antenna.x, antenna.y:
#     stream.add_signal(f_start=(6e9 - chan_bw * 10.25), 
#                         drift_rate=0*u.Hz/u.s, 
#                         snr=100,
#                         mode='snr')
#     stream.add_signal(f_start=(6e9 - chan_bw * 11.25), 
#                         drift_rate=0*u.Hz/u.s, 
#                         snr=150,
#                         mode='snr')
#     stream.add_signal(f_start=(6e9 - chan_bw * 12.25), 
#                         drift_rate=0*u.Hz/u.s, 
#                         snr=200,
#                         mode='snr')
#     stream.add_signal(f_start=(6e9 - chan_bw * 13.25), 
#                         drift_rate=0*u.Hz/u.s, 
#                         snr=250,
#                         mode='snr')


# antenna.x.add_signal(f_start=(6e9 - chan_bw * 12.7), 
#                     drift_rate=0*u.Hz/u.s, 
#                     snr=100,
#                     mode='snr')
# antenna.y.add_signal(f_start=(6e9 - chan_bw * 12.7), 
#                     drift_rate=0*u.Hz/u.s, 
#                     snr=100,
#                     phase=np.pi/2,
#                     mode='snr')

# antenna.x.add_signal(f_start=(6e9 - chan_bw * 16.7), 
#                     drift_rate=3*u.Hz/u.s, 
#                     snr=100,
#                     mode='snr')
# antenna.y.add_signal(f_start=(6e9 - chan_bw * 16.7), 
#                     drift_rate=3*u.Hz/u.s, 
#                     snr=100,
#                     phase=np.pi/2,
#                     mode='snr')


# antenna.x.add_signal(f_start=(6e9 - chan_bw * 14.7), 
#                     drift_rate=0*u.Hz/u.s, 
#                     snr=1000,
#                     mode='snr')
# antenna.y.add_signal(f_start=(6e9 - chan_bw * 14.7), 
#                     drift_rate=0*u.Hz/u.s, 
#                     snr=1000,
#                     phase=np.pi/2,
#                     mode='snr')


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

rvp.record(raw_file_stem='/datax/scratch/bbrzycki/data/raw_files/test_snr_0',
           num_blocks=1, 
           start_chan=0,
           num_chans=64,
           num_subblocks=32,
           length_mode='num_blocks',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'})

print(time.time() - start)


for stream in [antenna.x]:#, antenna.y]:
#     stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*0, 
#                         drift_rate=0*u.Hz/u.s, 
#                         snr=100,
#                         mode='snr')
#     stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*1, 
#                         drift_rate=0*u.Hz/u.s, 
#                         snr=200,
#                         mode='snr')
#     stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*2, 
#                         drift_rate=0*u.Hz/u.s,
#                         snr=100,
#                         mode='snr')
#     stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*3, 
#                         drift_rate=0*u.Hz/u.s, 
#                         snr=200,
#                         mode='snr')
#     stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*4, 
#                         drift_rate=0*u.Hz/u.s,
#                         snr=100,
#                         mode='snr')
#     stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*5, 
#                         drift_rate=0*u.Hz/u.s, 
#                         snr=200,
#                         mode='snr')
    
    stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*0, 
                        drift_rate=0*u.Hz/u.s, 
                        level=0.005,
                        mode='level')
    stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*1, 
                        drift_rate=0*u.Hz/u.s, 
                        level=0.01,
                        mode='level')
    stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*2, 
                        drift_rate=0*u.Hz/u.s, 
                        level=0.005,
                        mode='level')
    stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*3, 
                        drift_rate=0*u.Hz/u.s, 
                        level=0.01,
                        mode='level')
    stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*4, 
                        drift_rate=0*u.Hz/u.s, 
                        level=0.005,
                        mode='level')
    stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*5, 
                        drift_rate=0*u.Hz/u.s, 
                        level=0.01,
                        mode='level')
    stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*6, 
                        drift_rate=0*u.Hz/u.s, 
                        level=0.005,
                        mode='level')
    stream.add_signal(f_start=6e9+(chan_bw*3/4)+(chan_bw*1/32)*7, 
                        drift_rate=0*u.Hz/u.s, 
                        level=0.01,
                        mode='level')

#     stream.add_signal(f_start=6002.1e6, 
#                         drift_rate=0*u.Hz/u.s, 
#                         level=0.005,
#                         mode='level')
#     stream.add_signal(f_start=6002.2e6, 
#                         drift_rate=0*u.Hz/u.s, 
#                         level=0.01,
#                         mode='level')
#     stream.add_signal(f_start=6002.3e6, 
#                         drift_rate=0*u.Hz/u.s, 
#                         level=0.005,
#                         mode='level')
#     stream.add_signal(f_start=6002.4e6, 
#                         drift_rate=0*u.Hz/u.s, 
#                         level=0.01,
#                         mode='level')
#     stream.add_signal(f_start=6002.5e6, 
#                         drift_rate=0*u.Hz/u.s, 
#                         level=0.005,
#                         mode='level')
#     stream.add_signal(f_start=6002.6e6, 
#                         drift_rate=0*u.Hz/u.s, 
#                         level=0.01,
#                         mode='level')
#     stream.add_signal(f_start=6002.7e6, 
#                         drift_rate=0*u.Hz/u.s, 
#                         level=0.005,
#                         mode='level')
#     stream.add_signal(f_start=6002.8e6, 
#                         drift_rate=0*u.Hz/u.s, 
#                         level=0.01,
#                         mode='level')

rvp.record(raw_file_stem='/datax/scratch/bbrzycki/data/raw_files/test_snr_1',
           num_blocks=1, 
           start_chan=0,
           num_chans=64,
           num_subblocks=32,
           length_mode='num_blocks',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'})