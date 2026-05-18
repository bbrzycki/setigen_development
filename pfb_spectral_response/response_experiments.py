from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import setigen as stg
from pfb_response_tools import (
    PFBExperimentConfig,
    flattened_bin_index,
    ideal_response,
    modeled_bandpass_excess_sweep,
    modeled_bandpass_summary,
    run_spectrogram,
)


OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
FIGURES.mkdir(exist_ok=True)


EDGE_BINS = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 240, 248, 252, 254, 255]


def prototype_responses(config: PFBExperimentConfig) -> dict[str, np.ndarray]:
    """Compute PFB prototype responses for tone and noise predictions."""
    filterbank = stg.voltage.PolyphaseFilterbank(config.num_taps, config.num_branches)
    window = np.asarray(filterbank.window)
    fftlength = config.fftlength
    padded_len = config.num_branches * fftlength
    padded = np.zeros(padded_len)
    padded[: len(window)] = window
    H = np.fft.fft(padded)
    P = np.abs(H) ** 2
    center_power = P[0]
    offsets = (np.arange(fftlength) - fftlength // 2) / fftlength

    single_branch = []
    aliased_noise = []
    for offset in offsets:
        index = int(round(offset * fftlength)) % padded_len
        left_alias = int(round((offset - 1) * fftlength)) % padded_len
        right_alias = int(round((offset + 1) * fftlength)) % padded_len
        single_branch.append(P[index] / center_power)
        aliased_noise.append((P[left_alias] + P[index] + P[right_alias]) / center_power)

    single_branch = np.asarray(single_branch)
    aliased_noise = np.asarray(aliased_noise)
    setigen_noise = ideal_response(config)

    return {
        "offsets": offsets,
        "single_branch": single_branch,
        "aliased_noise": aliased_noise / aliased_noise[fftlength // 2],
        "setigen_noise": setigen_noise / setigen_noise[fftlength // 2],
    }


def simulate_signal_transfer(config: PFBExperimentConfig, bins: list[int]) -> np.ndarray:
    """Measure clean target-channel tone response from the actual voltage pipeline."""
    powers = []
    for fine_bin in bins:
        data = run_spectrogram(
            config,
            seed=40_000 + fine_bin,
            noise=False,
            tone_bin=fine_bin,
            tone_coarse_offset=1,
            tone_level=0.02,
            num_chans=3,
        )
        target_index = flattened_bin_index(config, fine_bin, coarse_offset=1)
        powers.append(float(data[:, target_index].mean()))
    powers = np.asarray(powers)
    return powers / powers[bins.index(config.fftlength // 2)]


def plot_transfer_functions(config: PFBExperimentConfig, responses: dict[str, np.ndarray]) -> None:
    fine_offset_khz = responses["offsets"] * config.chan_bw / 1e3
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(fine_offset_khz, responses["single_branch"], lw=2, label="Tone: single PFB branch")
    ax.plot(fine_offset_khz, responses["setigen_noise"], lw=2, label="Noise: setigen get_response()")
    ax.plot(fine_offset_khz, responses["aliased_noise"], "--", lw=1.5, label="Noise: adjacent-alias sum")
    ax.axvline(0, color="0.2", lw=0.8, alpha=0.5)
    ax.axhline(1, color="0.2", lw=0.8, alpha=0.5)
    ax.set_xlabel("Fine-channel offset from coarse-channel center (kHz)")
    ax.set_ylabel("Relative power transfer")
    ax.set_title("PFB tone transfer is not the same as the noise bandpass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "transfer_functions.pdf")
    fig.savefig(FIGURES / "transfer_functions.png", dpi=200)
    plt.close(fig)


def plot_signal_prediction(
    config: PFBExperimentConfig,
    responses: dict[str, np.ndarray],
    bins: list[int],
    simulated: np.ndarray,
) -> float:
    predicted = responses["single_branch"][bins]
    max_abs_error = float(np.max(np.abs(simulated - predicted)))
    fine_offset_khz = (np.asarray(bins) - config.fftlength // 2) * config.df / 1e3
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(fine_offset_khz, predicted, "o-", label="Predicted from PFB window")
    ax.plot(fine_offset_khz, simulated, "s", label="Voltage simulation")
    ax.set_xlabel("Fine-channel offset from coarse-channel center (kHz)")
    ax.set_ylabel("Peak-channel signal power / center signal power")
    ax.set_title("PFB window exactly predicts clean tone transfer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "signal_prediction.pdf")
    fig.savefig(FIGURES / "signal_prediction.png", dpi=200)
    plt.close(fig)
    return max_abs_error


def plot_modeled_bandpass_sweep(config: PFBExperimentConfig, rows: list[dict[str, float]]) -> None:
    offsets = np.asarray([row["fine_offset_hz"] for row in rows]) / 1e3
    center_peak = rows[EDGE_BINS.index(config.fftlength // 2)]["peak_excess_power"]
    center_aperture = rows[EDGE_BINS.index(config.fftlength // 2)]["aperture_excess_power"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(offsets, [row["ideal_response_relative_to_center"] for row in rows], "o-", label="Noise bandpass model")
    ax.errorbar(
        offsets,
        [row["peak_excess_relative_to_center"] for row in rows],
        yerr=[row["peak_excess_sem"] / center_peak for row in rows],
        fmt="o-",
        capsize=3,
        label="Peak excess above model",
    )
    ax.errorbar(
        offsets,
        [row["aperture_excess_relative_to_center"] for row in rows],
        yerr=[row["aperture_excess_sem"] / center_aperture for row in rows],
        fmt="s-",
        capsize=3,
        label="+/-3 final-channel excess above model",
    )
    ax.axhline(1, color="0.2", lw=0.8, alpha=0.5)
    ax.set_xlabel("Fine-channel offset from coarse-channel center (kHz)")
    ax.set_ylabel("Relative to center")
    ax.set_title("Noisy detected excess above modeled PFB bandpass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "modeled_bandpass_excess.pdf")
    fig.savefig(FIGURES / "modeled_bandpass_excess.png", dpi=200)
    plt.close(fig)


def main() -> None:
    config = PFBExperimentConfig(spectra_factor=32)
    responses = prototype_responses(config)
    plot_transfer_functions(config, responses)

    simulated = simulate_signal_transfer(config, EDGE_BINS)
    max_abs_error = plot_signal_prediction(config, responses, EDGE_BINS, simulated)

    bandpass_summary = modeled_bandpass_summary(config, seed=20260502, num_chans=3)
    rows = modeled_bandpass_excess_sweep(
        config,
        EDGE_BINS,
        tone_level=0.02,
        target_coarse_offset=1,
        num_chans=3,
        aperture_half_width=3,
        fit_guard_bins=8,
        num_trials=4,
        reference_bin=config.fftlength // 2,
    )
    plot_modeled_bandpass_sweep(config, rows)

    center_row = rows[EDGE_BINS.index(config.fftlength // 2)]
    edge_row = rows[0]
    summary = {
        "config": {
            "num_taps": config.num_taps,
            "num_branches": config.num_branches,
            "fftlength": config.fftlength,
            "coarse_channel_bw_hz": config.chan_bw,
            "fine_channel_bw_hz": config.df,
            "spectra_factor": config.spectra_factor,
        },
        "bandpass_summary": bandpass_summary,
        "signal_prediction_max_abs_error": max_abs_error,
        "edge_relative": {
            "single_branch_prediction": float(responses["single_branch"][0]),
            "noise_bandpass_model": float(responses["setigen_noise"][0]),
            "modeled_peak_excess": edge_row["peak_excess_relative_to_center"],
            "modeled_peak_sem": edge_row["peak_excess_sem"] / center_row["peak_excess_power"],
            "modeled_aperture_excess": edge_row["aperture_excess_relative_to_center"],
            "modeled_aperture_sem": edge_row["aperture_excess_sem"] / center_row["aperture_excess_power"],
        },
        "rows": rows,
    }
    with open(OUT / "response_experiment_results.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(json.dumps(summary["edge_relative"], indent=2))
    print(f"signal_prediction_max_abs_error={max_abs_error:.6g}")


if __name__ == "__main__":
    main()
