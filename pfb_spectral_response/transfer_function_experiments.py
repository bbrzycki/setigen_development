from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pfb_response_tools import PFBExperimentConfig, run_spectrogram
from pfb_transfer_model import (
    coarse_coordinate_to_frequency,
    fine_bin_coordinate,
    flattened_channel_coordinates,
    nearest_flat_channel,
    noise_alias_response,
    predict_dedrift_path,
    predict_flat_noise_power,
    predict_flat_signal_power,
    tone_branch_response,
)


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
RESULTS = ROOT / "transfer_function_results.json"


def _window_sum(values: np.ndarray, center: int, half_width: int) -> float:
    start = max(0, center - half_width)
    stop = min(values.size, center + half_width + 1)
    return float(values[start:stop].sum())


def _local_max(values: np.ndarray, center: int, half_width: int) -> float:
    start = max(0, center - half_width)
    stop = min(values.size, center + half_width + 1)
    return float(values[start:stop].max())


def _run_clean_spectrum(
    config: PFBExperimentConfig,
    *,
    coordinate: float,
    drift_rate_hz_s: float = 0.0,
    seed: int,
    tone_level: float,
    num_chans: int,
) -> np.ndarray:
    data = run_spectrogram(
        config,
        seed=seed,
        noise=False,
        tone_frequency_hz=coarse_coordinate_to_frequency(config, coordinate),
        drift_rate_hz_s=drift_rate_hz_s,
        tone_level=tone_level,
        num_chans=num_chans,
    )
    return data


def exact_bin_signal_validation(config: PFBExperimentConfig, *, num_chans: int, target_coarse_offset: int) -> dict:
    bins = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 240, 248, 252, 254, 255]
    tone_level = 0.02
    reference_coord = fine_bin_coordinate(config, config.fftlength // 2, coarse_offset=target_coarse_offset)
    reference_data = _run_clean_spectrum(
        config,
        coordinate=reference_coord,
        seed=70_000,
        tone_level=tone_level,
        num_chans=num_chans,
    )
    reference_spectrum = reference_data.mean(axis=0)
    reference_index = target_coarse_offset * config.fftlength + config.fftlength // 2
    reference_peak = float(reference_spectrum[reference_index])
    reference_aperture = _window_sum(reference_spectrum, reference_index, 3)

    rows = []
    for fine_bin in bins:
        coordinate = fine_bin_coordinate(config, fine_bin, coarse_offset=target_coarse_offset)
        target_index = target_coarse_offset * config.fftlength + fine_bin
        data = _run_clean_spectrum(
            config,
            coordinate=coordinate,
            seed=70_000 + fine_bin,
            tone_level=tone_level,
            num_chans=num_chans,
        )
        spectrum = data.mean(axis=0)
        predicted = predict_flat_signal_power(config, coordinate, num_chans=num_chans)
        support = predicted > 1e-5
        rows.append(
            {
                "fine_bin": int(fine_bin),
                "coordinate": float(coordinate),
                "offset_khz": float((coordinate - target_coarse_offset) * config.chan_bw / 1e3),
                "target_index": int(target_index),
                "actual_peak_rel": float(spectrum[target_index] / reference_peak),
                "predicted_peak_rel": float(predicted[target_index]),
                "actual_aperture_rel": float(_window_sum(spectrum, target_index, 3) / reference_aperture),
                "predicted_aperture_rel": float(_window_sum(predicted, target_index, 3)),
                "actual_coarse_sum_rel": float(sum(spectrum[c * config.fftlength + fine_bin] for c in range(num_chans)) / reference_peak),
                "predicted_coarse_sum_rel": float(sum(predicted[c * config.fftlength + fine_bin] for c in range(num_chans))),
                "support_max_abs_error": float(np.max(np.abs(spectrum[support] / reference_peak - predicted[support]))),
                "time_samples": int(data.shape[0]),
            }
        )

    return {
        "bins": bins,
        "reference_peak_power": reference_peak,
        "reference_aperture_power": reference_aperture,
        "rows": rows,
        "metrics": {
            "peak_max_abs_error": float(max(abs(r["actual_peak_rel"] - r["predicted_peak_rel"]) for r in rows)),
            "aperture_max_abs_error": float(max(abs(r["actual_aperture_rel"] - r["predicted_aperture_rel"]) for r in rows)),
            "support_max_abs_error": float(max(r["support_max_abs_error"] for r in rows)),
        },
    }


def off_bin_signal_validation(config: PFBExperimentConfig, *, num_chans: int, target_coarse_offset: int) -> dict:
    cases = [
        {"fine_bin": 8, "fraction": 0.25},
        {"fine_bin": 8, "fraction": 0.50},
        {"fine_bin": 32, "fraction": 0.25},
        {"fine_bin": 32, "fraction": 0.50},
        {"fine_bin": 128, "fraction": 0.25},
        {"fine_bin": 128, "fraction": 0.50},
        {"fine_bin": 248, "fraction": 0.25},
        {"fine_bin": 248, "fraction": 0.50},
    ]
    tone_level = 0.02
    reference_coord = fine_bin_coordinate(config, config.fftlength // 2, coarse_offset=target_coarse_offset)
    reference_data = _run_clean_spectrum(
        config,
        coordinate=reference_coord,
        seed=80_000,
        tone_level=tone_level,
        num_chans=num_chans,
    )
    reference_peak = float(reference_data.mean(axis=0)[target_coarse_offset * config.fftlength + config.fftlength // 2])

    rows = []
    for case_index, case in enumerate(cases):
        fine_bin = case["fine_bin"]
        fraction = case["fraction"]
        coordinate = fine_bin_coordinate(config, fine_bin + fraction, coarse_offset=target_coarse_offset)
        nearest = nearest_flat_channel(config, coordinate, num_chans=num_chans)
        data = _run_clean_spectrum(
            config,
            coordinate=coordinate,
            seed=80_000 + 100 * fine_bin + int(100 * fraction),
            tone_level=tone_level,
            num_chans=num_chans,
        )
        spectrum = data.mean(axis=0) / reference_peak
        predicted = predict_flat_signal_power(config, coordinate, num_chans=num_chans)
        start = max(0, nearest - 12)
        stop = min(num_chans * config.fftlength, nearest + 13)
        rows.append(
            {
                "case_index": case_index,
                "fine_bin": int(fine_bin),
                "fraction": float(fraction),
                "coordinate": float(coordinate),
                "offset_khz": float((coordinate - target_coarse_offset) * config.chan_bw / 1e3),
                "nearest_index": int(nearest),
                "actual_nearest_rel": float(spectrum[nearest]),
                "predicted_nearest_rel": float(predicted[nearest]),
                "actual_local_max_rel": _local_max(spectrum, nearest, 12),
                "predicted_local_max_rel": _local_max(predicted, nearest, 12),
                "actual_aperture_rel": _window_sum(spectrum, nearest, 3),
                "predicted_aperture_rel": _window_sum(predicted, nearest, 3),
                "local_max_abs_error": float(np.max(np.abs(spectrum[start:stop] - predicted[start:stop]))),
            }
        )
    return {
        "cases": cases,
        "rows": rows,
        "metrics": {
            "nearest_max_abs_error": float(max(abs(r["actual_nearest_rel"] - r["predicted_nearest_rel"]) for r in rows)),
            "local_max_abs_error": float(max(r["local_max_abs_error"] for r in rows)),
            "aperture_max_abs_error": float(max(abs(r["actual_aperture_rel"] - r["predicted_aperture_rel"]) for r in rows)),
        },
    }


def noise_validation(base_config: PFBExperimentConfig, *, num_chans: int) -> dict:
    cases = [
        replace(base_config, fftlength=128, integration_factor=1, spectra_factor=128),
        replace(base_config, fftlength=256, integration_factor=1, spectra_factor=128),
        replace(base_config, fftlength=256, integration_factor=4, spectra_factor=128),
        replace(base_config, fftlength=512, integration_factor=1, spectra_factor=64),
    ]
    rows = []
    for i, config in enumerate(cases):
        data = run_spectrogram(
            config,
            seed=90_000 + i,
            noise=True,
            num_chans=num_chans,
        )
        measured = data.mean(axis=0)
        predicted = predict_flat_noise_power(config, num_chans=num_chans)
        scale = float(np.dot(measured, predicted) / np.dot(predicted, predicted))
        measured_rel = measured / scale
        residual = measured_rel - predicted
        target_center = config.fftlength + config.fftlength // 2
        rows.append(
            {
                "fftlength": int(config.fftlength),
                "integration_factor": int(config.integration_factor),
                "df_hz": float(config.df),
                "dt_s": float(config.num_branches / config.sample_rate * config.fftlength * config.integration_factor),
                "time_samples": int(data.shape[0]),
                "correlation": float(np.corrcoef(measured_rel, predicted)[0, 1]),
                "relative_residual_rms": float(np.sqrt(np.mean(residual**2))),
                "fit_scale": scale,
                "measured_edge_rel": float(measured_rel[config.fftlength]),
                "predicted_edge_rel": float(predicted[config.fftlength]),
                "measured_center_rel": float(measured_rel[target_center]),
                "predicted_center_rel": float(predicted[target_center]),
            }
        )
    return {"rows": rows}


def drift_validation(config: PFBExperimentConfig, *, num_chans: int, target_coarse_offset: int) -> dict:
    drift_config = replace(config, spectra_factor=16, integration_factor=1)
    start_coordinate = fine_bin_coordinate(drift_config, drift_config.fftlength // 2, coarse_offset=target_coarse_offset)
    fine_dt = drift_config.num_branches / drift_config.sample_rate * drift_config.fftlength
    unit_drift_hz_s = drift_config.df / fine_dt
    slopes = [0.0, 0.25, 0.5, 1.0, 2.0]
    tone_level = 0.02

    reference_data = _run_clean_spectrum(
        drift_config,
        coordinate=start_coordinate,
        drift_rate_hz_s=0.0,
        seed=100_000,
        tone_level=tone_level,
        num_chans=num_chans,
    )
    reference_rows = reference_data.shape[0]
    reference_prediction = predict_dedrift_path(
        drift_config,
        start_coordinate=start_coordinate,
        drift_rate_hz_s=0.0,
        num_integrations=reference_rows,
        num_chans=num_chans,
    )

    rows = []
    reference_sim_sum = None
    reference_sim_snr = None
    for slope in slopes:
        drift_rate = slope * unit_drift_hz_s
        data = _run_clean_spectrum(
            drift_config,
            coordinate=start_coordinate,
            drift_rate_hz_s=drift_rate,
            seed=100_000 + int(100 * slope),
            tone_level=tone_level,
            num_chans=num_chans,
        )
        prediction = predict_dedrift_path(
            drift_config,
            start_coordinate=start_coordinate,
            drift_rate_hz_s=drift_rate,
            num_integrations=data.shape[0],
            num_chans=num_chans,
        )
        sim_sum = 0.0
        for row_index, channel_index in enumerate(prediction.channel_indices):
            sim_sum += float(data[row_index, channel_index])
        sim_snr_proxy = sim_sum / prediction.noise_variance_sum**0.5
        if reference_sim_sum is None:
            reference_sim_sum = sim_sum
            reference_sim_snr = sim_snr_proxy
        rows.append(
            {
                "slope_bins_per_row": float(slope),
                "drift_rate_hz_s": float(drift_rate),
                "time_samples": int(data.shape[0]),
                "path_start_index": int(prediction.channel_indices[0]),
                "path_stop_index": int(prediction.channel_indices[-1]),
                "path_unique_channels": int(len(set(prediction.channel_indices))),
                "actual_path_power_rel": float(sim_sum / reference_sim_sum),
                "predicted_path_power_rel": float(prediction.signal_sum / reference_prediction.signal_sum),
                "actual_snr_proxy_rel": float(sim_snr_proxy / reference_sim_snr),
                "predicted_snr_proxy_rel": float(prediction.snr_proxy / reference_prediction.snr_proxy),
            }
        )
    return {
        "config": {
            "fftlength": drift_config.fftlength,
            "integration_factor": drift_config.integration_factor,
            "df_hz": drift_config.df,
            "dt_s": fine_dt * drift_config.integration_factor,
            "unit_drift_hz_s": unit_drift_hz_s,
        },
        "rows": rows,
        "metrics": {
            "snr_proxy_max_abs_error": float(max(abs(r["actual_snr_proxy_rel"] - r["predicted_snr_proxy_rel"]) for r in rows)),
            "path_power_max_abs_error": float(max(abs(r["actual_path_power_rel"] - r["predicted_path_power_rel"]) for r in rows)),
        },
    }


def plot_transfer_curves(config: PFBExperimentConfig) -> None:
    offsets = np.linspace(-0.5, 0.5, 501)
    tone = tone_branch_response(config, offsets)
    noise = noise_alias_response(config, offsets)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(offsets * config.chan_bw / 1e3, tone, label="Tone branch response")
    ax.plot(offsets * config.chan_bw / 1e3, noise, label="Noise alias response")
    ax.plot(offsets * config.chan_bw / 1e3, tone / noise, label="Local SNR proxy")
    ax.axhline(1, color="0.25", lw=0.8, alpha=0.5)
    ax.set_xlabel("Offset from coarse-channel center (kHz)")
    ax.set_ylabel("Relative response")
    ax.set_title("Analytic transfer functions from the PFB prototype")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "transfer_function_model_curves.png", dpi=200)
    fig.savefig(FIGURES / "transfer_function_model_curves.pdf")
    plt.close(fig)


def main() -> None:
    config = PFBExperimentConfig(spectra_factor=32)
    num_chans = 3
    target_coarse_offset = 1
    plot_transfer_curves(config)
    results = {
        "config": {
            "sample_rate_hz": config.sample_rate,
            "num_taps": config.num_taps,
            "num_branches": config.num_branches,
            "fftlength": config.fftlength,
            "integration_factor": config.integration_factor,
            "coarse_channel_bw_hz": config.chan_bw,
            "fine_channel_bw_hz": config.df,
            "dt_s": config.num_branches / config.sample_rate * config.fftlength * config.integration_factor,
            "spectra_factor": config.spectra_factor,
            "num_chans": num_chans,
            "target_coarse_offset": target_coarse_offset,
        },
        "exact_bin_signal_validation": exact_bin_signal_validation(
            config,
            num_chans=num_chans,
            target_coarse_offset=target_coarse_offset,
        ),
        "off_bin_signal_validation": off_bin_signal_validation(
            config,
            num_chans=num_chans,
            target_coarse_offset=target_coarse_offset,
        ),
        "noise_validation": noise_validation(config, num_chans=num_chans),
        "drift_validation": drift_validation(
            config,
            num_chans=num_chans,
            target_coarse_offset=target_coarse_offset,
        ),
    }
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    summary = {
        "exact_peak_max_abs_error": results["exact_bin_signal_validation"]["metrics"]["peak_max_abs_error"],
        "offbin_nearest_max_abs_error": results["off_bin_signal_validation"]["metrics"]["nearest_max_abs_error"],
        "drift_snr_proxy_max_abs_error": results["drift_validation"]["metrics"]["snr_proxy_max_abs_error"],
        "noise_cases": len(results["noise_validation"]["rows"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
