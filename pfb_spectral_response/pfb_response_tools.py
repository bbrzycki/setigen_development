from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from astropy.stats import sigma_clip

import setigen as stg
from setigen.voltage.spectrogram import generate_voltage_spectrogram


@dataclass(frozen=True)
class PFBExperimentConfig:
    sample_rate: float = 64e6
    antenna_fch1: float = 1e9
    num_taps: int = 8
    num_branches: int = 64
    start_chan: int = 4
    num_chans: int = 1
    fftlength: int = 256
    integration_factor: int = 1
    num_pols: int = 1
    num_bits: int = 8
    spectra_factor: int = 128
    num_subblocks: int = 8

    @property
    def chan_bw(self) -> float:
        return self.sample_rate / self.num_branches

    @property
    def df(self) -> float:
        return self.chan_bw / self.fftlength

    @property
    def bytes_per_sample(self) -> int:
        return 2 * self.num_pols * self.num_bits // 8

    @property
    def block_size(self) -> int:
        return (
            self.num_chans
            * self.num_taps
            * self.bytes_per_sample
            * self.fftlength
            * self.spectra_factor
        )


def ideal_response(config: PFBExperimentConfig, *, num_chans: int | None = None) -> np.ndarray:
    """Return the tiled PFB power response normalized to mean one."""
    filterbank = stg.voltage.PolyphaseFilterbank(
        num_taps=config.num_taps,
        num_branches=config.num_branches,
    )
    if num_chans is None:
        num_chans = config.num_chans
    response = np.asarray(
        filterbank.tile_response(num_chans=num_chans, fftlength=config.fftlength),
        dtype=float,
    )
    return response / response.mean()


def fine_bin_frequency(
    config: PFBExperimentConfig,
    fine_bin: int,
    *,
    coarse_offset: int = 0,
) -> float:
    """Return the sky frequency for a fine bin in a selected coarse channel."""
    return (
        config.antenna_fch1
        + (config.start_chan + coarse_offset) * config.chan_bw
        - config.chan_bw / 2
        + fine_bin * config.df
    )


def flattened_bin_index(config: PFBExperimentConfig, fine_bin: int, *, coarse_offset: int = 0) -> int:
    """Return the flattened spectrogram-channel index for a coarse/fine bin."""
    return coarse_offset * config.fftlength + fine_bin


def make_backend(
    config: PFBExperimentConfig,
    *,
    seed: int,
    noise: bool,
    tone_bin: int | None = None,
    tone_frequency_hz: float | None = None,
    tone_coarse_offset: int = 0,
    tone_level: float = 0.005,
    drift_rate_hz_s: float = 0.0,
    num_chans: int | None = None,
) -> stg.voltage.RawVoltageBackend:
    """Build a small voltage backend for one reproducible response experiment."""
    if num_chans is None:
        num_chans = config.num_chans
    if tone_bin is not None and tone_frequency_hz is not None:
        raise ValueError("Specify either tone_bin or tone_frequency_hz, not both.")

    antenna = stg.voltage.Antenna(
        sample_rate=config.sample_rate,
        fch1=config.antenna_fch1,
        ascending=True,
        num_pols=config.num_pols,
        seed=seed,
    )
    for stream in antenna.streams:
        if noise:
            stream.add_noise(v_mean=0, v_std=1)
        if tone_bin is not None or tone_frequency_hz is not None:
            if tone_frequency_hz is None:
                tone_frequency_hz = fine_bin_frequency(
                    config,
                    tone_bin,
                    coarse_offset=tone_coarse_offset,
                )
            stream.add_constant_signal(
                f_start=tone_frequency_hz,
                drift_rate=drift_rate_hz_s,
                level=tone_level,
            )

    filterbank = stg.voltage.PolyphaseFilterbank(
        num_taps=config.num_taps,
        num_branches=config.num_branches,
    )
    return stg.voltage.RawVoltageBackend(
        antenna,
        digitizer=stg.voltage.RealQuantizer(num_bits=config.num_bits),
        filterbank=filterbank,
        requantizer=stg.voltage.ComplexQuantizer(num_bits=config.num_bits),
        start_chan=config.start_chan,
        num_chans=num_chans,
        block_size=config.block_size * num_chans // config.num_chans,
        num_subblocks=config.num_subblocks,
    )


def run_spectrogram(
    config: PFBExperimentConfig,
    *,
    seed: int,
    noise: bool,
    tone_bin: int | None = None,
    tone_frequency_hz: float | None = None,
    tone_coarse_offset: int = 0,
    tone_level: float = 0.005,
    drift_rate_hz_s: float = 0.0,
    num_chans: int | None = None,
    num_blocks: int = 1,
    digitize: bool = False,
    requantize: bool = False,
) -> np.ndarray:
    """Generate a direct voltage spectrogram and return total-power data."""
    if num_chans is None:
        num_chans = config.num_chans
    backend = make_backend(
        config,
        seed=seed,
        noise=noise,
        tone_bin=tone_bin,
        tone_frequency_hz=tone_frequency_hz,
        tone_coarse_offset=tone_coarse_offset,
        tone_level=tone_level,
        drift_rate_hz_s=drift_rate_hz_s,
        num_chans=num_chans,
    )
    spec = stg.voltage.VoltageSpectrogramSpec(
        fftlength=config.fftlength,
        integration_factor=config.integration_factor,
        start_chan=0,
        num_chans=num_chans,
    )
    result = generate_voltage_spectrogram(
        backend,
        spec,
        num_blocks=num_blocks,
        length_mode="num_blocks",
        digitize=digitize,
        requantize=requantize,
        verbose=False,
        xp=np,
    )
    return result.data[:, 0, :]


def local_noise_stats(
    data: np.ndarray,
    channel_index: int,
    *,
    context_bins: int = 32,
    guard_bins: int = 3,
    sigma: float = 3,
    maxiters: int = 5,
) -> tuple[float, float, int]:
    """Estimate local power-domain noise stats around one fine channel."""
    start = max(0, channel_index - context_bins)
    stop = min(data.shape[1], channel_index + context_bins + 1)
    keep = np.ones(stop - start, dtype=bool)
    guard_start = max(0, channel_index - guard_bins - start)
    guard_stop = min(stop - start, channel_index + guard_bins + 1 - start)
    keep[guard_start:guard_stop] = False
    if not np.any(keep):
        raise ValueError("Local noise window contains no unguarded channels.")

    samples = data[:, start:stop][:, keep].ravel()
    clipped = sigma_clip(
        samples,
        sigma=sigma,
        maxiters=maxiters,
        masked=False,
    )
    return float(np.mean(clipped)), float(np.std(clipped)), int(len(clipped))


def noise_overlay_summary(config: PFBExperimentConfig, *, seed: int = 1234) -> dict[str, float]:
    """Run a noise-only spectrogram and compare the bandpass with the ideal response."""
    data = run_spectrogram(config, seed=seed, noise=True)
    measured = data.mean(axis=0)
    measured_norm = measured / measured.mean()
    response = ideal_response(config)
    return {
        "correlation": float(np.corrcoef(measured_norm, response)[0, 1]),
        "measured_min_over_mean": float(measured_norm.min()),
        "measured_max_over_mean": float(measured_norm.max()),
        "ideal_min_over_mean": float(response.min()),
        "ideal_max_over_mean": float(response.max()),
        "num_time_samples": int(data.shape[0]),
        "num_frequency_channels": int(data.shape[1]),
    }


def tone_response_sweep(
    config: PFBExperimentConfig,
    bins: Iterable[int],
    *,
    tone_level: float = 0.005,
) -> list[dict[str, float]]:
    """Compare signal-only tone power with local noise and path SNR."""
    response = ideal_response(config)
    rows = []
    for fine_bin in bins:
        signal = run_spectrogram(
            config,
            seed=10_000 + fine_bin,
            noise=False,
            tone_bin=fine_bin,
            tone_level=tone_level,
        )
        noise = run_spectrogram(
            config,
            seed=20_000 + fine_bin,
            noise=True,
        )
        _, noise_std, noise_samples = local_noise_stats(noise, fine_bin)
        signal_power = float(signal[:, fine_bin].mean())
        path_snr = float(signal_power * np.sqrt(signal.shape[0]) / noise_std)
        rows.append(
            {
                "fine_bin": int(fine_bin),
                "ideal_response": float(response[fine_bin]),
                "signal_power": signal_power,
                "local_noise_std": float(noise_std),
                "path_snr": path_snr,
                "time_samples": int(signal.shape[0]),
                "noise_samples": int(noise_samples),
            }
        )
    return rows


def detected_intensity_sweep(
    config: PFBExperimentConfig,
    bins: Iterable[int],
    *,
    tone_level: float = 0.005,
    target_coarse_offset: int = 1,
    num_chans: int = 3,
    integration_half_width: int = 3,
    num_trials: int = 4,
    reference_bin: int | None = None,
) -> list[dict[str, float]]:
    """Measure detected excess power for fixed-amplitude tones across a coarse channel.

    The primary estimator is a matched noisy pair:
    ``power(noise + signal) - power(noise)``. Each pair uses the same random
    seed, so the noise-only power is removed exactly. The remaining finite
    cross term averages down over time samples and independent trials.
    """
    if reference_bin is None:
        reference_bin = config.fftlength // 2
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")

    response = ideal_response(config, num_chans=num_chans)
    rows = []
    for fine_bin in bins:
        target_index = flattened_bin_index(
            config,
            fine_bin,
            coarse_offset=target_coarse_offset,
        )
        window_start = max(0, target_index - integration_half_width)
        window_stop = min(num_chans * config.fftlength, target_index + integration_half_width + 1)
        same_fine_indices = [
            flattened_bin_index(config, fine_bin, coarse_offset=coarse)
            for coarse in range(num_chans)
        ]

        peak_trials = []
        window_trials = []
        coarse_sum_trials = []
        coarse_max_trials = []
        coarse_channel_trials = []
        noise_std_trials = []
        noise_sample_trials = []
        time_sample_trials = []

        for trial in range(num_trials):
            seed = 30_000 + 1_000 * int(fine_bin) + trial
            combined = run_spectrogram(
                config,
                seed=seed,
                noise=True,
                tone_bin=fine_bin,
                tone_coarse_offset=target_coarse_offset,
                tone_level=tone_level,
                num_chans=num_chans,
            )
            noise = run_spectrogram(
                config,
                seed=seed,
                noise=True,
                num_chans=num_chans,
            )
            excess = combined - noise
            coarse_channel_excess = excess[:, same_fine_indices].mean(axis=0)
            peak_trials.append(float(coarse_channel_excess[target_coarse_offset]))
            window_trials.append(float(excess[:, window_start:window_stop].sum(axis=1).mean()))
            coarse_sum_trials.append(float(coarse_channel_excess.sum()))
            coarse_max_trials.append(float(coarse_channel_excess.max()))
            coarse_channel_trials.append(np.asarray(coarse_channel_excess, dtype=float))
            _, noise_std, noise_samples = local_noise_stats(noise, target_index)
            noise_std_trials.append(float(noise_std))
            noise_sample_trials.append(int(noise_samples))
            time_sample_trials.append(int(excess.shape[0]))

        def mean_sem(values: list[float]) -> tuple[float, float]:
            array = np.asarray(values, dtype=float)
            mean = float(array.mean())
            if len(array) == 1:
                return mean, 0.0
            return mean, float(array.std(ddof=1) / len(array)**0.5)

        peak_excess, peak_sem = mean_sem(peak_trials)
        window_excess, window_sem = mean_sem(window_trials)
        coarse_sum_excess, coarse_sum_sem = mean_sem(coarse_sum_trials)
        coarse_max_excess, coarse_max_sem = mean_sem(coarse_max_trials)
        noise_std = float(np.mean(noise_std_trials))
        time_samples = int(np.mean(time_sample_trials))
        coarse_channel_excess = np.mean(coarse_channel_trials, axis=0)
        coarse_channel_sem = (
            np.zeros(num_chans)
            if num_trials == 1
            else np.std(coarse_channel_trials, axis=0, ddof=1) / num_trials**0.5
        )
        path_snr = float(peak_excess * np.sqrt(time_samples) / noise_std)
        window_response = float(response[window_start:window_stop].sum())
        neighbor_excess = coarse_sum_excess - peak_excess
        neighbor_fraction = (
            float(neighbor_excess / coarse_sum_excess)
            if coarse_sum_excess != 0
            else float("nan")
        )

        rows.append(
            {
                "fine_bin": int(fine_bin),
                "target_index": int(target_index),
                "fine_offset_hz": float((fine_bin - config.fftlength / 2) * config.df),
                "ideal_response": float(response[target_index]),
                "ideal_window_response": window_response,
                "peak_excess_power": peak_excess,
                "window_excess_power": window_excess,
                "coarse_sum_excess_power": coarse_sum_excess,
                "coarse_max_excess_power": coarse_max_excess,
                "peak_excess_sem": peak_sem,
                "window_excess_sem": window_sem,
                "coarse_sum_excess_sem": coarse_sum_sem,
                "coarse_max_excess_sem": coarse_max_sem,
                "coarse_channel_excess_power": coarse_channel_excess.tolist(),
                "coarse_channel_excess_sem": coarse_channel_sem.tolist(),
                "coarse_neighbor_fraction": neighbor_fraction,
                "coarse_support_ratio": (
                    float(coarse_sum_excess / peak_excess)
                    if peak_excess != 0
                    else float("nan")
                ),
                "local_noise_std": float(noise_std),
                "path_snr": path_snr,
                "time_samples": time_samples,
                "noise_samples": int(np.mean(noise_sample_trials)),
                "num_trials": int(num_trials),
            }
        )

    ref_index = next(
        i for i, row in enumerate(rows) if row["fine_bin"] == reference_bin
    )
    for row in rows:
        row["peak_excess_relative_to_center"] = (
            row["peak_excess_power"] / rows[ref_index]["peak_excess_power"]
        )
        row["window_excess_relative_to_center"] = (
            row["window_excess_power"] / rows[ref_index]["window_excess_power"]
        )
        row["coarse_sum_excess_relative_to_center"] = (
            row["coarse_sum_excess_power"] / rows[ref_index]["coarse_sum_excess_power"]
        )
        row["coarse_max_excess_relative_to_center"] = (
            row["coarse_max_excess_power"] / rows[ref_index]["coarse_max_excess_power"]
        )
        row["coarse_channel_excess_relative_to_center"] = [
            value / rows[ref_index]["peak_excess_power"]
            for value in row["coarse_channel_excess_power"]
        ]
        row["ideal_response_relative_to_center"] = (
            row["ideal_response"] / rows[ref_index]["ideal_response"]
        )
        row["ideal_window_response_relative_to_center"] = (
            row["ideal_window_response"] / rows[ref_index]["ideal_window_response"]
        )
        row["path_snr_relative_to_center"] = (
            row["path_snr"] / rows[ref_index]["path_snr"]
        )
    return rows


def _fit_pfb_bandpass_scale(
    spectrum: np.ndarray,
    response: np.ndarray,
    *,
    target_index: int | None = None,
    guard_bins: int = 8,
) -> float:
    """Fit a scalar PFB bandpass model to one mean spectrum."""
    mask = np.ones(len(spectrum), dtype=bool)
    if target_index is not None:
        start = max(0, target_index - guard_bins)
        stop = min(len(spectrum), target_index + guard_bins + 1)
        mask[start:stop] = False
    if not np.any(mask):
        raise ValueError("Bandpass fit mask contains no channels.")
    return float(np.dot(spectrum[mask], response[mask]) / np.dot(response[mask], response[mask]))


def modeled_bandpass_summary(
    config: PFBExperimentConfig,
    *,
    seed: int = 1234,
    num_chans: int = 3,
) -> dict[str, float]:
    """Compare a noise-only spectrum against a scaled ideal PFB model."""
    data = run_spectrogram(config, seed=seed, noise=True, num_chans=num_chans)
    spectrum = data.mean(axis=0)
    response = ideal_response(config, num_chans=num_chans)
    scale = _fit_pfb_bandpass_scale(spectrum, response)
    model = scale * response
    residual = spectrum - model
    return {
        "correlation": float(np.corrcoef(spectrum, model)[0, 1]),
        "model_scale": scale,
        "relative_residual_rms": float(np.sqrt(np.mean(residual**2)) / np.mean(spectrum)),
        "measured_min_over_mean": float(spectrum.min() / spectrum.mean()),
        "measured_max_over_mean": float(spectrum.max() / spectrum.mean()),
        "model_min_over_mean": float(model.min() / model.mean()),
        "model_max_over_mean": float(model.max() / model.mean()),
        "num_time_samples": int(data.shape[0]),
        "num_frequency_channels": int(data.shape[1]),
    }


def modeled_bandpass_excess_sweep(
    config: PFBExperimentConfig,
    bins: Iterable[int],
    *,
    tone_level: float = 0.008,
    target_coarse_offset: int = 1,
    num_chans: int = 3,
    aperture_half_width: int = 3,
    fit_guard_bins: int = 8,
    num_trials: int = 4,
    reference_bin: int | None = None,
) -> list[dict[str, float]]:
    """Measure noisy signal excess above a scaled modeled PFB bandpass.

    The signal is placed at exact fine-channel centers in one target coarse
    channel. For each noisy trial, the time-mean spectrum is fit with a scaled
    ideal PFB response while masking a guard region around the injected signal.
    Detected excess is then measured in the final flattened spectrum, either at
    the peak channel or in a local final-frequency aperture.
    """
    if reference_bin is None:
        reference_bin = config.fftlength // 2
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")

    response = ideal_response(config, num_chans=num_chans)
    rows = []
    for fine_bin in bins:
        target_index = flattened_bin_index(
            config,
            fine_bin,
            coarse_offset=target_coarse_offset,
        )
        aperture_start = max(0, target_index - aperture_half_width)
        aperture_stop = min(num_chans * config.fftlength, target_index + aperture_half_width + 1)
        peak_trials = []
        aperture_trials = []
        scale_trials = []
        residual_rms_trials = []
        local_noise_trials = []
        time_sample_trials = []

        for trial in range(num_trials):
            seed = 50_000 + 1_000 * int(fine_bin) + trial
            data = run_spectrogram(
                config,
                seed=seed,
                noise=True,
                tone_bin=fine_bin,
                tone_coarse_offset=target_coarse_offset,
                tone_level=tone_level,
                num_chans=num_chans,
            )
            spectrum = data.mean(axis=0)
            scale = _fit_pfb_bandpass_scale(
                spectrum,
                response,
                target_index=target_index,
                guard_bins=fit_guard_bins,
            )
            model = scale * response
            residual = spectrum - model
            peak_trials.append(float(residual[target_index]))
            aperture_trials.append(float(residual[aperture_start:aperture_stop].sum()))
            scale_trials.append(float(scale))
            residual_rms_trials.append(float(np.sqrt(np.mean(residual**2)) / np.mean(spectrum)))
            _, noise_std, _ = local_noise_stats(
                data,
                target_index,
                context_bins=32,
                guard_bins=max(3, fit_guard_bins),
            )
            local_noise_trials.append(float(noise_std))
            time_sample_trials.append(int(data.shape[0]))

        def mean_sem(values: list[float]) -> tuple[float, float]:
            array = np.asarray(values, dtype=float)
            mean = float(array.mean())
            if len(array) == 1:
                return mean, 0.0
            return mean, float(array.std(ddof=1) / len(array)**0.5)

        peak_excess, peak_sem = mean_sem(peak_trials)
        aperture_excess, aperture_sem = mean_sem(aperture_trials)
        local_noise = float(np.mean(local_noise_trials))
        time_samples = int(np.mean(time_sample_trials))
        path_snr = float(peak_excess * np.sqrt(time_samples) / local_noise)

        rows.append(
            {
                "fine_bin": int(fine_bin),
                "target_index": int(target_index),
                "fine_offset_hz": float((fine_bin - config.fftlength / 2) * config.df),
                "ideal_response": float(response[target_index]),
                "peak_excess_power": peak_excess,
                "aperture_excess_power": aperture_excess,
                "peak_excess_sem": peak_sem,
                "aperture_excess_sem": aperture_sem,
                "bandpass_scale": float(np.mean(scale_trials)),
                "relative_residual_rms": float(np.mean(residual_rms_trials)),
                "local_noise_std": local_noise,
                "path_snr": path_snr,
                "time_samples": time_samples,
                "num_trials": int(num_trials),
            }
        )

    ref_index = next(
        i for i, row in enumerate(rows) if row["fine_bin"] == reference_bin
    )
    for row in rows:
        row["ideal_response_relative_to_center"] = (
            row["ideal_response"] / rows[ref_index]["ideal_response"]
        )
        row["peak_excess_relative_to_center"] = (
            row["peak_excess_power"] / rows[ref_index]["peak_excess_power"]
        )
        row["aperture_excess_relative_to_center"] = (
            row["aperture_excess_power"] / rows[ref_index]["aperture_excess_power"]
        )
        row["path_snr_relative_to_center"] = (
            row["path_snr"] / rows[ref_index]["path_snr"]
        )
    return rows


def normalize_column(rows: list[dict[str, float]], key: str) -> np.ndarray:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return values / values.mean()
