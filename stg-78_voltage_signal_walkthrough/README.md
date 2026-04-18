# Voltage Signal Walkthrough

This folder contains a notebook that walks through a minimal voltage workflow:

- construct a dual-polarization antenna source
- inspect raw time-domain voltage samples
- record a small synthetic GUPPI RAW file
- reduce the first RAW block to a fine-channelized spectrogram
- visualize the injected signals in the final product

Open [voltage_signal_walkthrough.ipynb](./voltage_signal_walkthrough.ipynb) in
Jupyter and run the cells top to bottom.

The notebook assumes `setigen` is already available in the active environment,
for example via `pip install -e .` from the main `setigen` repository.
