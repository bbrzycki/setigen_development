
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

M     = 8        # Number of taps
P     = 1024       # Number of 'branches', also fft length
W     = 2**14 + 1       # Number of windows of length M*P in input time stream
n_int = 2          # Number of time integrations on output data
sample_rate = 3e9

stream = stg.voltage.DataStream(sample_rate=3e9)
stream.add_noise(v_mean=0, 
                 v_std=1)
stream.add_signal(f_start=(sample_rate/P * (204.2))*u.Hz, 
                  drift_rate=50000000*u.Hz/u.s, 
                  level=0.1)
stream.get_samples(num_samples=M*P*W)

digitize_stream = stg.voltage.quantize_real(stream.v,
                                                    target_fwhm=30,
                                                    n_bits=8)

pfb_voltages = stg.voltage.get_pfb_voltages(digitize_stream, M, P)
print(pfb_voltages.shape)

# Perform quantization
n_bits = 8
target_fwhm = 30 * 2**n_bits / 2**8 
q_pfb_voltages = stg.voltage.quantize_complex(pfb_voltages, 
                                                      target_fwhm=target_fwhm,
                                                      n_bits=n_bits)

# # Here is where we would write to raw files!

XX_psd = stg.voltage.get_pfb_waterfall(pfb_voltages=q_pfb_voltages,
                                               n_int=n_int,
                                               fftlength=256,
                                               start_channel=203,
                                               num_channels=4)


print(time.time() - start)


start = time.time()

def get_pfb_voltages(x, n_taps, n_chan, start_channel, num_channels, window_fn='hamming'):
    """
    Produce complex raw voltage data as a function of time and coarse channel.
    """
    # Generate window coefficients
    win_coeffs = stg.voltage.get_pfb_window(n_taps, n_chan, window_fn)
#     win_coeffs /= np.max(win_coeffs)
    # Apply frontend, take FFT, then take power (i.e. square)
    x_fir = stg.voltage.pfb_frontend(x, win_coeffs, n_taps, n_chan)
    
    pfb_comp = []
    for i in range(num_channels):
        w = np.exp(-2*np.pi*1j*(1/P * (start_channel + i))*np.repeat(np.arange(P)[np.newaxis, :], x_fir.shape[0], axis=0))
        x_pfb = np.sum(x_fir * w, axis=1, keepdims=True)
        pfb_comp.append(x_pfb)
    
    return np.concatenate(pfb_comp, axis=1)
    


stream = stg.voltage.DataStream(sample_rate=3e9)
stream.add_noise(v_mean=0, 
                 v_std=1)
stream.add_signal(f_start=(sample_rate/P * (204.2))*u.Hz, 
                  drift_rate=50000000*u.Hz/u.s, 
                  level=0.1)
stream.get_samples(num_samples=M*P*W)

digitize_stream = stg.voltage.quantize_real(stream.v,
                                                    target_fwhm=30,
                                                    n_bits=8)

pfb_voltages = get_pfb_voltages(digitize_stream, M, P, 203, 4)
print(pfb_voltages.shape)

# Perform quantization
n_bits = 8
target_fwhm = 30 * 2**n_bits / 2**8 
q_pfb_voltages = stg.voltage.quantize_complex(pfb_voltages, 
                                                      target_fwhm=target_fwhm,
                                                      n_bits=n_bits)

# # Here is where we would write to raw files!

XX_psd = stg.voltage.get_pfb_waterfall(pfb_voltages=q_pfb_voltages,
                                               n_int=n_int,
                                               fftlength=256,
                                               start_channel=0,
                                               num_channels=4)

print(time.time() - start)



start = time.time()

def get_pfb_voltages1(x, n_taps, n_chan, start_channel, num_channels, window_fn='hamming'):
    """
    Produce complex raw voltage data as a function of time and coarse channel.
    """
    # Generate window coefficients
    win_coeffs = stg.voltage.get_pfb_window(n_taps, n_chan, window_fn)
#     win_coeffs /= np.max(win_coeffs)
    # Apply frontend, take FFT, then take power (i.e. square)
    x_fir = stg.voltage.pfb_frontend(x, win_coeffs, n_taps, n_chan)
    
    pfb_comp = []
    for i in range(num_channels):
        w = np.exp(-2*np.pi*1j*(1/P * (start_channel + i))*np.repeat(np.arange(P)[np.newaxis, :], x_fir.shape[0], axis=0))
        x_pfb = np.sum(x_fir * w, axis=1, keepdims=True)
        pfb_comp.append(x_pfb)
    
    return np.concatenate(pfb_comp, axis=1)
    
stream = stg.voltage.DataStream(sample_rate=3e9)
stream.add_noise(v_mean=0, 
                 v_std=1)
stream.add_signal(f_start=(sample_rate/P * (204.2))*u.Hz, 
                  drift_rate=50000000*u.Hz/u.s, 
                  level=0.1)
stream.get_samples(num_samples=M*P*W)

digitize_stream = stg.voltage.quantize_real(stream.v,
                                                    target_fwhm=30,
                                                    n_bits=8)

pfb_voltages = get_pfb_voltages1(digitize_stream, M, P, 203, 4)
print(pfb_voltages.shape)

# Perform quantization
n_bits = 8
target_fwhm = 30 * 2**n_bits / 2**8 
q_pfb_voltages = stg.voltage.quantize_complex(pfb_voltages, 
                                                      target_fwhm=target_fwhm,
                                                      n_bits=n_bits)

# # Here is where we would write to raw files!

XX_psd = stg.voltage.get_pfb_waterfall(pfb_voltages=q_pfb_voltages,
                                               n_int=n_int,
                                               fftlength=256,
                                               start_channel=0,
                                               num_channels=4)

print(time.time() - start)