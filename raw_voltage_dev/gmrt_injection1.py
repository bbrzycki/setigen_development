import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import matplotlib.pyplot as plt

import numpy as np
from astropy import units as u
import blimpy as bl
import setigen as stg

output_dir = "/datax/scratch/bbrzycki/data/raw_files/"



wf = bl.Waterfall('/datax/scratch/bbrzycki/data/raw_files/inj_573_50.rawspec.0000.fil',
                  f_start=555,
                  f_stop=558)

plt.figure(figsize=(10, 6))
wf.plot_waterfall()
plt.savefig('gmrt_wf.png')

frame = stg.Frame(wf)

spectrum = stg.integrate(frame, normalize=True)

plt.figure(figsize=(8, 6))
plt.plot(spectrum)
plt.xlabel('Frequency bins')
plt.ylabel('SNR')
plt.savefig('gmrt_spec.png')

plt.figure(figsize=(8, 6))
plt.plot(spectrum)
plt.xlabel('Frequency bins')
plt.ylabel('SNR')
plt.yscale('symlog')
plt.savefig('gmrt_spec1.png')