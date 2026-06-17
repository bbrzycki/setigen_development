from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import setigen as stg
from pfb_response_tools import PFBExperimentConfig


def coarse_coordinate_to_frequency(config: PFBExperimentConfig, coordinate: float) -> float:
    """Convert a recorded-relative coarse-channel coordinate to sky frequency."""
    return config.antenna_fch1 + (config.start_chan + coordinate) * config.chan_bw


def fine_bin_coordinate(config: PFBExperimentConfig, fine_bin: float, *, coarse_offset: int = 0) -> float:
    """Return recorded-relative coarse coordinate for a fine-bin location."""
    return coarse_offset + (fine_bin - config.fftlength / 2) / config.fftlength


def pfb_prototype_transform(config: PFBExperimentConfig, offsets: Sequence[float] | np.ndarray) -> np.ndarray:
    """Evaluate the PFB prototype transform at offsets in coarse-channel units."""
    offsets_array = np.atleast_1d(np.asarray(offsets, dtype=float))
    filterbank = stg.voltage.PolyphaseFilterbank(
        num_taps=config.num_taps,
        num_branches=config.num_branches,
    )
    window = np.asarray(filterbank.window, dtype=float)
    sample_index = np.arange(window.size, dtype=float)
    phase = np.exp(-2j * np.pi * offsets_array[:, np.newaxis] * sample_index[np.newaxis, :] / config.num_branches)
    return phase @ window


def tone_branch_response(config: PFBExperimentConfig, offsets: Sequence[float] | np.ndarray) -> np.ndarray:
    """Power response for a coherent tone in one PFB analysis branch."""
    transform = pfb_prototype_transform(config, offsets)
    center = pfb_prototype_transform(config, [0.0])[0]
    return np.abs(transform) ** 2 / np.abs(center) ** 2


def noise_alias_response(
    config: PFBExperimentConfig,
    offsets: Sequence[float] | np.ndarray,
    *,
    alias_span: int | None = None,
) -> np.ndarray:
    """White-noise baseline response from aliased PFB prototype power."""
    offsets_array = np.atleast_1d(np.asarray(offsets, dtype=float))
    if alias_span is None:
        alias_span = config.num_taps
    aliases = np.arange(-alias_span, alias_span + 1, dtype=float)
    all_offsets = (offsets_array[:, np.newaxis] + aliases[np.newaxis, :]).reshape(-1)
    powers = np.abs(pfb_prototype_transform(config, all_offsets)) ** 2
    powers = powers.reshape(offsets_array.size, aliases.size).sum(axis=1)
    center_power = float((np.abs(pfb_prototype_transform(config, aliases)) ** 2).sum())
    return powers / center_power


def fine_fft_response(fftlength: int, bin_offsets: Sequence[float] | np.ndarray) -> np.ndarray:
    """Rectangular fine-FFT peak-bin power response normalized to one on-bin."""
    offsets = np.asarray(bin_offsets, dtype=float)
    wrapped = ((offsets + fftlength / 2) % fftlength) - fftlength / 2
    numerator = np.sin(np.pi * wrapped)
    denominator = fftlength * np.sin(np.pi * wrapped / fftlength)
    response = np.empty_like(wrapped, dtype=float)
    on_bin = np.isclose(wrapped, 0.0, atol=1e-12)
    response[on_bin] = 1.0
    response[~on_bin] = (numerator[~on_bin] / denominator[~on_bin]) ** 2
    return response


def flattened_channel_coordinates(config: PFBExperimentConfig, num_chans: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coarse index, fine index, and coarse-coordinate center per flat channel."""
    coarse = np.repeat(np.arange(num_chans), config.fftlength)
    fine = np.tile(np.arange(config.fftlength), num_chans)
    coordinate = coarse + (fine - config.fftlength / 2) / config.fftlength
    return coarse, fine, coordinate


def predict_flat_signal_power(
    config: PFBExperimentConfig,
    signal_coordinate: float,
    *,
    num_chans: int,
) -> np.ndarray:
    """Predict coherent signal power across the final flattened frequency axis.

    The result is normalized so a center-bin tone measured in its center channel
    has predicted peak power of one.
    """
    coarse, fine, _ = flattened_channel_coordinates(config, num_chans)
    branch_offsets = signal_coordinate - coarse
    fine_offsets = (fine - config.fftlength / 2) / config.fftlength
    fine_bin_offsets = config.fftlength * (branch_offsets - fine_offsets)
    return tone_branch_response(config, branch_offsets) * fine_fft_response(config.fftlength, fine_bin_offsets)


def predict_flat_noise_power(config: PFBExperimentConfig, *, num_chans: int) -> np.ndarray:
    """Predict white-noise power baseline across the final flattened frequency axis."""
    _, fine, _ = flattened_channel_coordinates(config, num_chans)
    fine_offsets = (fine - config.fftlength / 2) / config.fftlength
    return noise_alias_response(config, fine_offsets)


def nearest_flat_channel(config: PFBExperimentConfig, coordinate: float, *, num_chans: int) -> int:
    """Map a coarse-coordinate frequency to the nearest flattened channel."""
    coarse = int(np.floor(coordinate + 0.5))
    coarse = min(max(coarse, 0), num_chans - 1)
    fine_float = (coordinate - coarse + 0.5) * config.fftlength
    fine = int(np.round(fine_float))
    fine = min(max(fine, 0), config.fftlength - 1)
    return coarse * config.fftlength + fine


@dataclass(frozen=True)
class PathPrediction:
    signal_sum: float
    noise_variance_sum: float
    snr_proxy: float
    channel_indices: list[int]


def predict_dedrift_path(
    config: PFBExperimentConfig,
    *,
    start_coordinate: float,
    drift_rate_hz_s: float,
    num_integrations: int,
    num_chans: int,
    aperture_half_width: int = 0,
    samples_per_integration: int | None = None,
) -> PathPrediction:
    """Predict a dedrifted path SNR proxy by sampling the transfer model in time.

    This is a piecewise-stationary model: each fine spectrum is evaluated at
    its midpoint frequency. It intentionally excludes the within-FFT chirp
    kernel, so high drift rates should be treated as a validation target rather
    than an already solved case.
    """
    if samples_per_integration is None:
        samples_per_integration = config.integration_factor
    fine_dt = config.num_branches / config.sample_rate * config.fftlength
    integration_signal = []
    noise_variance = []
    channel_indices = []
    noise_model = predict_flat_noise_power(config, num_chans=num_chans)

    for integration_index in range(num_integrations):
        signal_total = 0.0
        midpoint_coordinate = start_coordinate
        for sub_index in range(samples_per_integration):
            fine_index = integration_index * samples_per_integration + sub_index
            t_mid = (fine_index + 0.5) * fine_dt
            coordinate = start_coordinate + drift_rate_hz_s * t_mid / config.chan_bw
            midpoint_coordinate = coordinate
            flat_power = predict_flat_signal_power(config, coordinate, num_chans=num_chans)
            center = nearest_flat_channel(config, coordinate, num_chans=num_chans)
            start = max(0, center - aperture_half_width)
            stop = min(num_chans * config.fftlength, center + aperture_half_width + 1)
            signal_total += float(flat_power[start:stop].sum())

        path_channel = nearest_flat_channel(config, midpoint_coordinate, num_chans=num_chans)
        start = max(0, path_channel - aperture_half_width)
        stop = min(num_chans * config.fftlength, path_channel + aperture_half_width + 1)
        integration_signal.append(signal_total)
        noise_variance.append(float(np.sum(noise_model[start:stop] ** 2) * samples_per_integration))
        channel_indices.append(path_channel)

    signal_sum = float(np.sum(integration_signal))
    noise_variance_sum = float(np.sum(noise_variance))
    return PathPrediction(
        signal_sum=signal_sum,
        noise_variance_sum=noise_variance_sum,
        snr_proxy=float(signal_sum / noise_variance_sum**0.5),
        channel_indices=channel_indices,
    )
