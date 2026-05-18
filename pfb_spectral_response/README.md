# PFB Spectral Response Investigation

This folder is a self-contained voltage-domain investigation of the PFB-driven
spectral response visible in fine-channelized products. It is intentionally
separate from the `setigen` package code so the scientific assumptions can be
inspected before we turn any of this into public API.

## Working Questions

- What ideal PFB response is already available from `setigen.voltage`?
- Does measured white-noise power follow that ideal response after the voltage
  pipeline?
- For a fixed input voltage tone, does measured signal power and local SNR
  follow the same response?
- What is PFB/instrumental structure, as opposed to natural sky continuum or
  observed RFI occupancy?

## Current Findings

`setigen.voltage.PolyphaseFilterbank.get_response()` already computes a
PFB-only power response from the FIR/window coefficients. `tile_response()`
mirrors and repeats that response across coarse channels, which is the right
starting point for an ideal scalloping overlay.

In a direct voltage spectrogram with white Gaussian input noise, the measured
mean bandpass follows the tiled ideal response closely. This supports using the
ideal response as a PFB-only overlay, not as a complete telescope/instrument
bandpass model.

For the default small experiment in these notebooks (`num_branches=64`,
`fftlength=256`, one selected coarse channel, no digitization or requantization),
the noise-only overlay produced:

- measured/ideal correlation: 0.985
- measured min/mean: 0.552
- measured max/mean: 1.176
- ideal min/mean: 0.568
- ideal max/mean: 1.108

For a fixed input voltage tone, the tone's measured power also changes with
fine-channel position. In the central part of a coarse channel, the tone power
and local noise scale similarly enough that a local-noise SNR estimate is much
less response-dependent than a global-background estimate. Near coarse-channel
edges, the coherent tone response can fall more sharply, so peak-bin SNR can
drop instead of simply improving because the local noise floor is lower.

The practical implication is that voltage-domain injection SNR is not just
"lower noise means proportionally higher SNR." If the signal is injected before
the PFB, the PFB acts on both signal and noise. If a signal is injected directly
into a spectrogram with fixed additive intensity, then local SNR can vary
inversely with the local noise floor unless the injected intensity is adjusted
by an explicit response model.

The default tone sweep used fine-channel bins `[8, 32, 64, 96, 128, 160, 192,
224, 248]`. Relative to the sweep mean, the central bins have signal power,
local noise, and path-summed SNR close to each other, while the near-edge bins
show much lower coherent tone power and SNR. That is the main calibration point
to chase next with adjacent coarse-channel accounting.

The edge-focused sweep in `04_edge_detected_intensity.ipynb` uses a fixed
original voltage-tone amplitude and measures detected excess power in noisy
spectrograms above a modeled PFB bandpass. For each injected fine-bin position,
the notebook fits a scaled ideal PFB response to the final flattened spectrum
while masking the signal neighborhood. The residual at the injected final
channel is the detected excess.

Relative to a center-bin tone in the default noisy modeled-bandpass run
(`tone_level=0.02`, four trials):

- the ideal PFB response at the exact coarse-channel edge is about 0.52
- the detected final-channel peak excess at the exact edge is about 0.24-0.26
  after subtracting the modeled bandpass at that same final channel
- a `+/-3` final-channel aperture is also measured to test whether the detector
  recovers power spread into adjacent final channels near the edge; in the
  current run, the aperture edge response is about 0.21-0.23
- this is the calibration-relevant quantity for ordinary spectrogram analysis:
  one final frequency axis, one modeled baseline, one local signal aperture

This means a center-channel signal and an edge-channel signal with the same
original voltage amplitude do not have the same detected spectrogram intensity.
The correction must be tied to the detector aperture we actually use: peak
channel, local final-frequency aperture, or dedrifted path aperture.

## Notebooks

1. `01_ideal_pfb_response.ipynb`
   Plots the ideal PFB response from the filterbank coefficients and shows the
   repeat pattern across coarse channels.

2. `02_noise_response_overlay.ipynb`
   Generates a small direct voltage spectrogram from white noise and overlays
   measured noise power with the ideal PFB response.

3. `03_tone_response_and_snr.ipynb`
   Sweeps fixed-amplitude voltage tones across fine-channel positions and
   compares tone power, local noise, and path-summed SNR.

4. `04_edge_detected_intensity.ipynb`
   Uses noisy simulations to measure detected excess power above a scaled ideal
   PFB bandpass for a fixed original voltage tone, with dense sampling near the
   coarse-channel edges.

## Report Artifacts

- `pfb_response_report.tex` contains the LaTeX report source.
- `pfb_response_report.pdf` is the rendered report, built in the `seti` conda
  environment with Tectonic.
- `response_experiments.py` reproduces the derivation checks, figures, and
  `response_experiment_results.json`.
- `figures/` contains the generated transfer-function and modeled-excess plots.
- `render_report_pdf.py` is a Matplotlib fallback renderer for machines that do
  not have a working TeX engine; it writes `pfb_response_report_fallback.pdf`.

## Rebuilding

From this folder, regenerate the notebooks with:

```bash
conda run -n seti python make_notebooks.py
```

Validate the notebook code cells without requiring notebook execution packages:

```bash
MPLCONFIGDIR=/tmp/matplotlib-seti conda run -n seti python validate_notebooks.py
```

Reproduce the report experiments and render the PDF with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-seti conda run -n seti python response_experiments.py
conda run -n seti tectonic --only-cached pfb_response_report.tex
```

If Tectonic is not available, render the fallback PDF with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-seti conda run -n seti python render_report_pdf.py
```

The notebooks assume `setigen` is already installed in the `seti` conda
environment, for example with `pip install -e .` from the main `setigen`
repository.

## Next Scientific Checks

- Compare this ideal response against RAW files generated through the full
  write/read/reduce path, not only direct voltage spectrogram generation.
- Repeat with quantization enabled to separate pure PFB effects from digitizer
  and requantizer behavior.
- Extend tone tests across adjacent coarse channels so edge splitting/leakage is
  measured explicitly, not inferred from one recorded coarse channel.
- Compare `hamming` and other PFB windows/tap counts.
- Decide whether `Frame`-level injection should expose an optional
  response-aware target-SNR mode for synthetic spectrogram injection.
