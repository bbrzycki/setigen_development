import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import matplotlib.pyplot as plt

import numpy as np
from astropy import units as u
import blimpy as bl
import setigen as stg

output_dir = "/datax/scratch/bbrzycki/data/raw_files/"

fig, axs = plt.subplots(nrows=4, figsize=(6, 8))

wf = bl.Waterfall('/datax/scratch/bbrzycki/data/raw_files/blc00_guppi_59114_15868_TMC1_0010_injected.rawspec.0000.fil',
                  f_start=25635.464844,
                  f_stop=25638.564844)

plt.sca(axs[0])
wf.plot_waterfall()
# plt.savefig('gbt_wf.png')

frame = stg.Frame(wf).get_slice(250, 1500)

print(frame.df, frame.dt)
print(frame.shape)

spectrum = stg.integrate(frame, normalize=True)

plt.sca(axs[1])
plt.plot(spectrum)
plt.xlabel('Frequency bins')
plt.ylabel('SNR')
plt.grid()
# plt.savefig('gbt_spec.png')

plt.sca(axs[2])
plt.plot(spectrum)
plt.xlabel('Frequency bins')
plt.ylabel('SNR')
plt.yscale('symlog')
plt.grid()
# plt.savefig('gbt_spec1.png')

frame = frame.get_slice(700, 1200)
spectrum = stg.integrate(frame, normalize=True)

plt.sca(axs[3])
plt.plot(spectrum)
plt.xlabel('Frequency bins')
plt.ylabel('SNR')
plt.grid()

plt.tight_layout()
plt.savefig('gbt_spec_multi.png')