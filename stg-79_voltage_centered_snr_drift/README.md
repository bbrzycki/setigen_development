# Centered Voltage SNR Drift Walkthrough

This folder contains a narrower voltage example built directly on the existing
raw-voltage SNR notebooks:

- two synthetic voltage signals placed on exact fine-bin centers in the flat middle of one coarse channel
- target SNRs of 25 and 50 via `setigen.voltage.get_level()`
- a longer single-coarse-channel RAW recording and 11 Hz-class fine channelization via the native `setigen.voltage.reduce_raw(...)` path
- reduction of that single recorded coarse channel to a `.fil` product, reopening with `blimpy.Waterfall`, then `setigen.dedrift()` and normalized spectra
- an explicit backend working-set budget so the large fine-channelization example stays bounded in memory

Open [voltage_centered_snr_drift.ipynb](./voltage_centered_snr_drift.ipynb) in
Jupyter and run the cells top to bottom.

The notebook assumes `setigen` is already available in the active environment,
for example via `pip install -e .` from the main `setigen` repository.
