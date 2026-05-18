from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_DIR = Path(__file__).resolve().parent


def _lines(source: str) -> list[str]:
    return [f"{line}\n" for line in dedent(source).strip().splitlines()]


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(NOTEBOOK_DIR / name, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")


COMMON_IMPORTS = """
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from pfb_response_tools import (
    PFBExperimentConfig,
    detected_intensity_sweep,
    ideal_response,
    local_noise_stats,
    modeled_bandpass_excess_sweep,
    modeled_bandpass_summary,
    noise_overlay_summary,
    normalize_column,
    run_spectrogram,
    tone_response_sweep,
)

plt.rcParams.update({
    "figure.figsize": (9, 4),
    "axes.grid": True,
    "grid.alpha": 0.25,
})
"""


def make_01() -> None:
    write_notebook(
        "01_ideal_pfb_response.ipynb",
        [
            md(
                """
                # Ideal PFB Spectral Response

                This notebook inspects the ideal PFB-only response already
                exposed by `setigen.voltage.PolyphaseFilterbank`. This is the
                response from the PFB FIR/window coefficients, not a complete
                telescope bandpass and not a sky continuum model.
                """
            ),
            code(COMMON_IMPORTS),
            code(
                """
                config = PFBExperimentConfig()
                response = ideal_response(config)

                summary = {
                    "fine channels": len(response),
                    "min / mean": response.min(),
                    "max / mean": response.max(),
                    "max / min": response.max() / response.min(),
                    "df Hz": config.df,
                    "coarse channel bandwidth Hz": config.chan_bw,
                }
                summary
                """
            ),
            code(
                """
                fine = np.arange(config.fftlength)
                fine_offset = (fine - config.fftlength / 2) * config.df / 1e3

                fig, ax = plt.subplots()
                ax.plot(fine_offset, response, lw=2, color="tab:blue")
                ax.axvline(0, color="0.25", lw=1, alpha=0.7)
                ax.set_xlabel("Fine-channel offset from coarse-channel center (kHz)")
                ax.set_ylabel("PFB power response / mean")
                ax.set_title("Ideal PFB response within one coarse channel")
                display(fig)
                plt.close(fig)
                """
            ),
            code(
                """
                tiled = ideal_response(config, num_chans=4)
                flat_index = np.arange(len(tiled))

                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(flat_index, tiled, lw=1.5, color="tab:purple")
                for boundary in range(config.fftlength, len(tiled), config.fftlength):
                    ax.axvline(boundary, color="0.3", lw=0.8, alpha=0.5)
                ax.set_xlabel("Flattened fine-channel index")
                ax.set_ylabel("PFB power response / mean")
                ax.set_title("Response tiled across coarse channels")
                display(fig)
                plt.close(fig)
                """
            ),
            md(
                """
                The important feature is the repeated coarse-channel structure:
                the ideal response is high through much of the coarse-channel
                interior and lower near the coarse-channel edges. Any SNR
                estimate over wide spectral regions should not assume a single
                stationary background variance.
                """
            ),
        ],
    )


def make_02() -> None:
    write_notebook(
        "02_noise_response_overlay.ipynb",
        [
            md(
                """
                # Noise Response Overlay

                This notebook generates a small direct voltage spectrogram from
                white Gaussian voltage noise. It then overlays the measured
                power bandpass with the ideal PFB response.

                The run uses `digitize=False` and `requantize=False` so the
                result isolates the PFB and fine-channelization behavior from
                quantization effects.
                """
            ),
            code(COMMON_IMPORTS),
            code(
                """
                config = PFBExperimentConfig()
                data = run_spectrogram(config, seed=20260502, noise=True)
                measured = data.mean(axis=0)
                measured_norm = measured / measured.mean()
                response = ideal_response(config)

                summary = noise_overlay_summary(config, seed=20260502)
                summary
                """
            ),
            code(
                """
                fine = np.arange(config.fftlength)
                fine_offset = (fine - config.fftlength / 2) * config.df / 1e3

                fig, ax = plt.subplots()
                ax.plot(fine_offset, measured_norm, lw=1.5, label="Measured noise mean")
                ax.plot(fine_offset, response, lw=2, alpha=0.8, label="Ideal PFB response")
                ax.set_xlabel("Fine-channel offset from coarse-channel center (kHz)")
                ax.set_ylabel("Power / band mean")
                ax.set_title("Noise bandpass vs ideal PFB response")
                ax.legend()
                display(fig)
                plt.close(fig)
                """
            ),
            code(
                """
                measured_std = data.std(axis=0)
                measured_std_norm = measured_std / measured_std.mean()

                fig, ax = plt.subplots()
                ax.plot(fine_offset, measured_norm, label="Mean power")
                ax.plot(fine_offset, measured_std_norm, label="Std per pixel")
                ax.plot(fine_offset, response, lw=2, alpha=0.75, label="Ideal response")
                ax.set_xlabel("Fine-channel offset from coarse-channel center (kHz)")
                ax.set_ylabel("Relative level")
                ax.set_title("Noise mean and variance both inherit PFB structure")
                ax.legend()
                display(fig)
                plt.close(fig)
                """
            ),
            code(
                """
                channel_index = config.fftlength // 2
                noise_mean, noise_std, sample_count = local_noise_stats(
                    data,
                    channel_index,
                    context_bins=32,
                    guard_bins=3,
                )
                {
                    "center channel index": channel_index,
                    "local sigma-clipped mean": noise_mean,
                    "local sigma-clipped std": noise_std,
                    "samples after clipping": sample_count,
                }
                """
            ),
            md(
                """
                The overlay is the main sanity check: for white voltage noise,
                measured power follows the PFB response. This supports our
                recent decision to estimate SNR from local spectral windows:
                local windows follow the nearby PFB structure, while a global
                mean/std mixes coarse-channel positions with different response
                levels.
                """
            ),
        ],
    )


def make_03() -> None:
    write_notebook(
        "03_tone_response_and_snr.ipynb",
        [
            md(
                """
                # Tone Response And SNR

                This notebook asks what happens to a fixed-amplitude voltage
                tone at different positions within one coarse channel.

                For each fine-channel position we run two direct voltage
                spectrograms:

                - signal only, to measure coherent tone power at the target bin
                - noise only, to estimate the local power-domain noise std

                This separates deterministic transfer through the PFB from
                random noise scatter.
                """
            ),
            code(COMMON_IMPORTS),
            code(
                """
                config = PFBExperimentConfig()
                bins = [8, 32, 64, 96, 128, 160, 192, 224, 248]
                rows = tone_response_sweep(config, bins, tone_level=0.005)
                rows
                """
            ),
            code(
                """
                fine_bins = np.asarray([row["fine_bin"] for row in rows])
                fine_offset = (fine_bins - config.fftlength / 2) * config.df / 1e3

                signal_rel = normalize_column(rows, "signal_power")
                noise_rel = normalize_column(rows, "local_noise_std")
                snr_rel = normalize_column(rows, "path_snr")
                ideal_rel = normalize_column(rows, "ideal_response")

                fig, ax = plt.subplots()
                ax.plot(fine_offset, ideal_rel, "o-", label="Ideal PFB response")
                ax.plot(fine_offset, signal_rel, "o-", label="Signal-only tone power")
                ax.plot(fine_offset, noise_rel, "o-", label="Local noise std")
                ax.plot(fine_offset, snr_rel, "o-", label="Path-summed SNR")
                ax.axhline(1, color="0.25", lw=1, alpha=0.5)
                ax.set_xlabel("Fine-channel offset from coarse-channel center (kHz)")
                ax.set_ylabel("Relative to sweep mean")
                ax.set_title("Fixed voltage tone across PFB response")
                ax.legend()
                display(fig)
                plt.close(fig)
                """
            ),
            code(
                """
                compact = [
                    {
                        "fine_bin": row["fine_bin"],
                        "ideal_rel": ideal_rel[i],
                        "signal_power_rel": signal_rel[i],
                        "noise_std_rel": noise_rel[i],
                        "path_snr_rel": snr_rel[i],
                    }
                    for i, row in enumerate(rows)
                ]
                compact
                """
            ),
            md(
                """
                The central bins show the expected behavior: signal power and
                local noise both inherit the PFB response, so local-noise SNR is
                much less response-dependent than a global-background estimate.

                The near-edge bins are different. The coherent tone peak drops
                sharply near coarse-channel edges, and peak-bin SNR drops with
                it. This is a warning that a single "noise is lower, so SNR is
                proportionally higher" rule is not right for voltage-domain
                signals. The PFB transfer function acts on the signal too, and
                edge behavior likely needs adjacent-coarse-channel accounting
                before we use this as a calibrated correction.
                """
            ),
            md(
                """
                For frame-level synthetic injection, this means there are two
                separate contracts:

                - voltage-domain injection should be calibrated against the PFB
                  transfer of both signal and noise
                - spectrogram-domain injection with fixed additive intensity
                  should optionally adjust intensity by a response/noise model
                  if the target is constant local SNR across the band
                """
            ),
        ],
    )


def make_04() -> None:
    write_notebook(
        "04_edge_detected_intensity.ipynb",
        [
            md(
                """
                # Detected Intensity Above A Modeled PFB Bandpass

                This notebook targets the central science question: if the
                original voltage-domain signal has fixed amplitude, how much
                detected spectrogram power rises above the local background at
                different positions in a coarse channel?

                We inject a zero-drift tone centered on exact fine-channel bins
                across one coarse channel. Every tone has the same original
                voltage amplitude. For each noisy realization, we fit a scaled
                ideal PFB bandpass to the final flattened spectrum while
                masking the injected channel neighborhood. The signal
                measurement is the residual above that modeled bandpass.

                This keeps the measurement on the actual final-frequency axis:
                there is no same-fine-bin cross-coarse summing. The optional
                aperture measurement is only a local final-frequency window
                around the injected channel.
                """
            ),
            code(COMMON_IMPORTS),
            code(
                """
                config = PFBExperimentConfig(spectra_factor=32)
                target_coarse_offset = 1
                num_chans = 3

                edge_bins = [
                    0, 1, 2, 4, 8, 16, 32, 64,
                    96, 128, 160, 192, 224, 240, 248, 252, 254, 255,
                ]

                bandpass_check = modeled_bandpass_summary(
                    config,
                    seed=20260502,
                    num_chans=num_chans,
                )

                rows = modeled_bandpass_excess_sweep(
                    config,
                    edge_bins,
                    tone_level=0.02,
                    target_coarse_offset=target_coarse_offset,
                    num_chans=num_chans,
                    aperture_half_width=3,
                    fit_guard_bins=8,
                    num_trials=4,
                    reference_bin=config.fftlength // 2,
                )

                bandpass_check, rows[:3], rows[-3:]
                """
            ),
            code(
                """
                noise_data = run_spectrogram(
                    config,
                    seed=20260502,
                    noise=True,
                    num_chans=num_chans,
                )
                noise_spectrum = noise_data.mean(axis=0)
                response = ideal_response(config, num_chans=num_chans)
                response_model = noise_spectrum.mean() * response
                response_model *= noise_spectrum.mean() / response_model.mean()

                final_frequency_index = np.arange(len(noise_spectrum))
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(final_frequency_index, noise_spectrum / noise_spectrum.mean(), lw=1.2, label="Noise mean")
                ax.plot(final_frequency_index, response_model / response_model.mean(), lw=2, alpha=0.8, label="Scaled ideal PFB model")
                for boundary in range(config.fftlength, len(noise_spectrum), config.fftlength):
                    ax.axvline(boundary, color="0.25", lw=0.8, alpha=0.4)
                ax.set_xlabel("Final flattened frequency-channel index")
                ax.set_ylabel("Relative power")
                ax.set_title("Noise baseline follows the repeated PFB bandpass")
                ax.legend()
                display(fig)
                plt.close(fig)
                """
            ),
            code(
                """
                fine_offset_khz = np.asarray([row["fine_offset_hz"] for row in rows]) / 1e3
                center_row = rows[edge_bins.index(config.fftlength // 2)]
                center_peak = center_row["peak_excess_power"]
                center_aperture = center_row["aperture_excess_power"]

                fig, ax = plt.subplots()
                ax.plot(
                    fine_offset_khz,
                    [row["ideal_response_relative_to_center"] for row in rows],
                    "o-",
                    label="Ideal PFB response at target bin",
                )
                ax.errorbar(
                    fine_offset_khz,
                    [row["peak_excess_relative_to_center"] for row in rows],
                    yerr=[row["peak_excess_sem"] / center_peak for row in rows],
                    fmt="o-",
                    capsize=3,
                    label="Peak-channel excess above modeled bandpass",
                )
                ax.errorbar(
                    fine_offset_khz,
                    [row["aperture_excess_relative_to_center"] for row in rows],
                    yerr=[row["aperture_excess_sem"] / center_aperture for row in rows],
                    fmt="o-",
                    capsize=3,
                    label="+/-3 final-channel aperture excess above model",
                )
                ax.axhline(1, color="0.25", lw=1, alpha=0.5)
                ax.set_xlabel("Fine-channel offset from coarse-channel center (kHz)")
                ax.set_ylabel("Relative to center-bin detected excess")
                ax.set_title("Same original tone amplitude: excess above modeled bandpass")
                ax.legend()
                display(fig)
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots()
                ax.errorbar(
                    fine_offset_khz,
                    [row["path_snr_relative_to_center"] for row in rows],
                    yerr=[row["peak_excess_sem"] / center_peak for row in rows],
                    fmt="o-",
                    capsize=3,
                    label="Peak-bin path SNR",
                )
                ax.plot(
                    fine_offset_khz,
                    [row["local_noise_std"] / center_row["local_noise_std"] for row in rows],
                    "o-",
                    label="Local noise std",
                )
                ax.plot(
                    fine_offset_khz,
                    [row["relative_residual_rms"] / center_row["relative_residual_rms"] for row in rows],
                    "o-",
                    label="Bandpass residual RMS",
                )
                ax.axhline(1, color="0.25", lw=1, alpha=0.5)
                ax.set_xlabel("Fine-channel offset from coarse-channel center (kHz)")
                ax.set_ylabel("Relative to center")
                ax.set_title("Modeled-bandpass excess, local noise, and fit quality")
                ax.legend()
                display(fig)
                plt.close(fig)
                """
            ),
            code(
                """
                compact = [
                    {
                        "fine_bin": row["fine_bin"],
                        "offset_kHz": row["fine_offset_hz"] / 1e3,
                        "ideal_rel": row["ideal_response_relative_to_center"],
                        "peak_excess_rel": row["peak_excess_relative_to_center"],
                        "peak_excess_sem_rel": row["peak_excess_sem"] / center_peak,
                        "aperture_excess_rel": row["aperture_excess_relative_to_center"],
                        "path_snr_rel": row["path_snr_relative_to_center"],
                    }
                    for row in rows
                ]
                compact
                """
            ),
            md(
                """
                This is the distinction we need for inference. The fixed
                original tone does not retain the same detected intensity across
                the coarse channel. The measurement here is exactly the final
                spectrum quantity we care about: data minus a modeled PFB
                bandpass baseline. In the middle of the coarse channel, the
                excess approximately follows the ideal PFB response. Near the
                edge, the peak-channel excess can fall below that smooth model,
                and the local final-frequency aperture tells us whether the
                detector recovers power spread into adjacent final channels.

                For real observations this means a narrowband candidate near a
                coarse-channel edge cannot be interpreted with the same
                intrinsic-intensity calibration as one near the coarse-channel
                center. The first useful correction is likely a PFB-position
                transfer curve for synthetic voltage tones, with separate
                treatment for peak-bin detection and integrated-window
                detection.
                """
            ),
        ],
    )


def main() -> None:
    make_01()
    make_02()
    make_03()
    make_04()


if __name__ == "__main__":
    main()
