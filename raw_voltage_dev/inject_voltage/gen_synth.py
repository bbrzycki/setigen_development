import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
from astropy import units as u
import blimpy as bl
import setigen as stg


DATA_DIR = '/datax/scratch/bbrzycki/data/raw_files/inject_voltage'

num_taps = 8
num_branches = 1024
raw_params = stg.voltage.get_raw_params(input_file_stem=f'{DATA_DIR}/gbt', 
                                        start_chan=0)
sample_rate = np.abs(raw_params['chan_bw'] * num_branches) 
print('sample rate', sample_rate)
print('\n input file parameters: \n ', raw_params)

antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                              **raw_params)
    
digitizer = stg.voltage.RealQuantizer(target_fwhm=8,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=8,
                                           num_bits=8)

rvb = stg.voltage.RawVoltageBackend.from_data(input_file_stem=f'{DATA_DIR}/gbt',
                                              antenna_source=antenna,
                                              digitizer=digitizer,
                                              filterbank=filterbank,
                                              start_chan=0,
                                              num_subblocks=128)

rvb = stg.voltage.RawVoltageBackend(antenna,
                                    digitizer=digitizer,
                                    filterbank=filterbank,
                                    requantizer=requantizer,
                                    start_chan=0,
                                    num_chans=64,
                                    block_size=rvb.block_size,
                                    blocks_per_file=128,
                                    num_subblocks=128)

for stream in antenna.streams:
    stream.add_noise(v_mean=0, 
                     v_std=3)

rvb.record(output_file_stem=f'{DATA_DIR}/synth3',
           num_blocks=1, 
           length_mode='num_blocks',
           header_dict={'TELESCOP': 'GBT'},
           verbose=True)