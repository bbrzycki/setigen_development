import matplotlib.pyplot as plt

import os
import argparse
import numpy as np
from astropy import units as u
import blimpy as bl
import setigen as stg


def plot(filename):
    fig, axs = plt.subplots(nrows=4, figsize=(6, 8))
    wf = bl.Waterfall(filename,
                      f_start=25635.464844 + 2.9296875,
                      f_stop=25638.564844 + 2.9296875)

    plt.sca(axs[0])
    wf.plot_waterfall()

    frame = stg.Frame(wf).get_slice(250, 1500)

    print(frame.df, frame.dt)
    print(frame.shape)

    spectrum = stg.integrate(frame, normalize=True)

    plt.sca(axs[1])
    plt.plot(spectrum)
    plt.xlabel('Frequency bins')
    plt.ylabel('SNR')
    plt.grid()

    plt.sca(axs[2])
    plt.plot(spectrum)
    plt.xlabel('Frequency bins')
    plt.ylabel('SNR')
    plt.yscale('symlog')
    plt.grid()

    frame = frame.get_slice(700, 1200)
    spectrum = stg.integrate(frame, normalize=True)

    plt.sca(axs[3])
    plt.plot(spectrum)
    plt.xlabel('Frequency bins')
    plt.ylabel('SNR')
    plt.grid()

    plt.tight_layout()
    DIR, FN = os.path.split(filename) 
    plt.savefig(f"/home/bryanb/setigen_development/raw_voltage_dev/inject_voltage/plots/spectra_{FN.split('.')[0]}.png")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    
    # Convert args to dictionary
    params = vars(args)
    plot(**params)
    
    
if __name__ == '__main__':
    main()