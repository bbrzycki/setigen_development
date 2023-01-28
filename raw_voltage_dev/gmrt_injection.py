import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import matplotlib.pyplot as plt

import numpy as np
from astropy import units as u
import blimpy as bl
import setigen as stg

output_dir = "/datax/scratch/bbrzycki/data/raw_files/"

def NB_inject(ip_file_stem, SNR,drift_rate, num_branches,f_start,f_stop,num_signals,op_file_stem ):
    
    start_chan = 0
    raw_params = stg.voltage.get_raw_params(input_file_stem=ip_file_stem, start_chan=start_chan)
    sample_rate=raw_params['chan_bw']*num_branches # chan_bw = sample_rate / num_branches to keep this constant 
    print('sample rate', sample_rate)
    num_taps=8
    

    print('\n input file parameters: \n ',raw_params)
    
    antenna = stg.voltage.Antenna(sample_rate=sample_rate, **raw_params)
    
    

    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                                 num_branches=num_branches)

    rvb = stg.voltage.RawVoltageBackend.from_data(input_file_stem=ip_file_stem,
                                                  antenna_source=antenna,
                                                  filterbank=filterbank,
                                                  start_chan=start_chan,
                                                  num_subblocks=32)

    
    assert f_start> rvb.fch1 and f_stop< rvb.fch1+rvb.chan_bw*rvb.num_chans, 'injection frequencies not in range '

#     print('\n Antenna noise std',antenna.x.get_total_noise_std())

    fftlength = 8192
    # num blocks
#     nb = 3
    nb=stg.voltage.get_total_blocks(ip_file_stem)
    
    print('\n total blocks in the file', nb)
    
    signal_level = stg.voltage.get_level(snr=SNR,
                                raw_voltage_backend=rvb,
                                fftlength=fftlength,
    #                             obs_length=4.652881237333333,
    #                             length_mode='obs_length'
                                num_blocks=nb,
                                length_mode='num_blocks'
                                        )

    print('\nsignal level ', signal_level)
    
    df = raw_params['chan_bw'] / fftlength

    for n_sig, fstart in enumerate(np.linspace(f_start, f_stop, num_signals)):
        fstart = raw_params['fch1'] + int((fstart - raw_params['fch1']) / df) * df
        leakage_factor = stg.voltage.get_leakage_factor(fstart, rvb, fftlength)
        print(f'{fstart/1e6:.4f} MHz leakage factor: {leakage_factor:.3f}')

        for stream in antenna.streams:
            # There's actually no noise present, but we set levels assuming a background noise_std of 1
            level = 1 * leakage_factor * signal_level * 10**((n_sig + 0)/2)
            print('snr: ',SNR*(level/(leakage_factor * signal_level))**0.5)
            print('level: ',level)
            stream.add_constant_signal(f_start=fstart, 
                                       drift_rate=drift_rate*u.Hz/u.s, 
                                       level=1*level)
    
    print('unit drift rate corres. 1x1 pixel in final data',stg.voltage.get_unit_drift_rate(rvb, fftlength, 1))
            
    rvb.record(output_file_stem=op_file_stem,
               num_blocks=nb, 
               length_mode='num_blocks',
              header_dict={'TELESCOP': 'GMRT', 'OBSERVER': 'AE' , 'SRC_NAME' : '7202'},
               digitize=False,
               verbose=True)

    
ip_file_stem='/home/eakshay/amity/raghav/GSB_iFFT_pipeline/PASV_to_GUPPI/TEST_padding_v2_guppi'
SNR=100
drift_rate=0   
num_branches=4096
f_start=555.5e6
f_stop=557.5e6
num_signals=9
op_file_stem='/datax/scratch/bbrzycki/data/raw_files/inj_573_50'

NB_inject(ip_file_stem, SNR, drift_rate, num_branches,f_start,f_stop,num_signals,op_file_stem )

