import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import numpy as np
from astropy import units as u
import blimpy as bl
import setigen as stg


DATA_DIR = '/datax/scratch/bbrzycki/data/raw_files/inject_voltage'
f_start = 25637.0e6 + 2.9296875e6
f_stop = 25637.5e6 + 2.9296875e6

num_taps = 8
num_branches = 1024
raw_params = stg.voltage.get_raw_params(input_file_stem=f'{DATA_DIR}/gbt', 
                                        start_chan=0)
sample_rate = np.abs(raw_params['chan_bw'] * num_branches) 
print('sample rate', sample_rate)
print('\n input file parameters: \n ', raw_params)


def inject(source='gbt', digitize='n', snr=100, fftlength=2048, int_factor=1):
    antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                                  **raw_params)

    digitizer = stg.voltage.RealQuantizer(target_fwhm=8,
                                          num_bits=8)

    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                                 num_branches=num_branches)

    rvb = stg.voltage.RawVoltageBackend.from_data(input_file_stem=f'{DATA_DIR}/{source}',
                                                  antenna_source=antenna,
                                                  digitizer=digitizer,
                                                  filterbank=filterbank,
                                                  start_chan=0,
                                                  num_subblocks=128)
    
    signal_level = stg.voltage.get_level(snr=snr,
                                         raw_voltage_backend=rvb,
                                         fftlength=fftlength,
                                         num_blocks=1,
                                         length_mode='num_blocks')
    print('\nsignal level', signal_level)
    
    if digitize == 'y':
        for stream in antenna.streams:
            stream.add_noise(v_mean=0, 
                             v_std=1)
            
    df = np.abs(raw_params['chan_bw'] / fftlength)
            
    for n_sig, sig_f in enumerate(np.linspace(f_start, f_stop, 3)):
        sig_f = raw_params['fch1'] + int((sig_f - raw_params['fch1']) / df) * df
        leakage_factor = stg.voltage.get_leakage_factor(sig_f, rvb, fftlength)
        print(f'{sig_f/1e6:.4f} MHz leakage factor: {leakage_factor:.3f}')

        for stream in antenna.streams:
            print(stream.get_total_noise_std())
            # There's actually no noise present, but we set levels assuming a background noise_std of 1
            level = 1 * leakage_factor * signal_level 
            print('snr: ',snr*(level/(leakage_factor * signal_level))**0.5)
            print('level: ',level)
            print('sig_f: ',sig_f)
            stream.add_constant_signal(f_start=sig_f, 
                                       drift_rate=0, 
                                       level=level)

    rvb.record(output_file_stem=f'{DATA_DIR}/{source}_D{digitize}_SNR{snr}_FFT{fftlength}',
               num_blocks=1, 
               length_mode='num_blocks',
               header_dict={'TELESCOP': 'GBT'},
               digitize=(digitize == 'y'),
               verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str)
    parser.add_argument('digitize', type=str)
    parser.add_argument('snr', type=int)
    parser.add_argument('fftlength', type=int, default=2048)
    parser.add_argument('int_factor', type=int, default=1)
    args = parser.parse_args()
    
    # Convert args to dictionary
    params = vars(args)
    inject(**params)


if __name__ == '__main__':
    main()