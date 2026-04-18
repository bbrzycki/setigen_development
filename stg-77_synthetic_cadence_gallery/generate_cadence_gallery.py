from __future__ import annotations

import os
from pathlib import Path
import sys

from astropy import units as u
from astropy.time import Time


REPO_ROOT = Path(__file__).resolve().parents[2] / "setigen"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MPLCONFIGDIR = Path(__file__).resolve().parent / ".mpl-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import setigen as stg

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

FCHANS = 768
TCHANS = 32
DF = 2.7939677238464355 * u.Hz
DT = 18.253611008 * u.s
FCH1 = 6095.214842353016 * u.MHz
FRAME_OBS_LENGTH = int((TCHANS * DT).to_value(u.s))
SLEW_TIME = 18
MJD_START = 60310


def _make_time_starts(order: str, *, obs_length: int, slew_time: int, mjd_start: int) -> list[float]:
    t_start_arr = [Time(mjd_start, format="mjd").unix]
    for i in range(1, len(order)):
        t_start_arr.append(t_start_arr[i - 1] + obs_length + slew_time)
    return t_start_arr


def _make_ordered_cadence(order: str, *, seed_offset: int) -> stg.OrderedCadence:
    t_start_arr = _make_time_starts(order,
                                    obs_length=FRAME_OBS_LENGTH,
                                    slew_time=SLEW_TIME,
                                    mjd_start=MJD_START)
    frames = [
        stg.Frame(
            fchans=FCHANS,
            tchans=TCHANS,
            df=DF,
            dt=DT,
            fch1=FCH1,
            t_start=t_start_arr[i],
            source_name=f"Pointing {label}{i}",
            seed=seed_offset + i,
        )
        for i, label in enumerate(order)
    ]
    cadence = stg.OrderedCadence(frames, order=order)
    cadence.apply(lambda fr: fr.add_noise(x_mean=8e5, noise_type="chi2"))
    _inject_signals(cadence)
    return cadence


def _inject_signals(cadence: stg.OrderedCadence) -> None:
    reference = cadence[0]

    if "A" in cadence.order:
        cadence.by_label("A").add_signal(
            stg.constant_path(
                f_start=reference.get_frequency(index=210),
                drift_rate=0.18 * u.Hz / u.s,
            ),
            stg.constant_t_profile(level=reference.get_intensity(snr=32)),
            stg.sinc2_f_profile(
                width=2 * reference.df * u.Hz,
                width_mode=stg.WidthMode.CROSSING,
                trunc=False,
            ),
            stg.constant_bp_profile(level=1),
            doppler_smearing=True,
        )

    if "B" in cadence.order:
        cadence.by_label("B").add_signal(
            stg.sine_path(
                f_start=reference.get_frequency(index=430),
                drift_rate=-0.03 * u.Hz / u.s,
                period=10 * reference.dt * u.s,
                amplitude=28 * reference.df * u.Hz,
            ),
            stg.periodic_gaussian_t_profile(
                pulse_width=2 * reference.dt * u.s,
                period=8 * reference.dt * u.s,
                pulse_direction=stg.PulseDirection.UP,
                amplitude=reference.get_intensity(snr=14),
                level=0,
                min_level=0,
                seed=2026,
            ),
            stg.gaussian_f_profile(width=7 * reference.df * u.Hz),
            stg.constant_bp_profile(level=1),
        )

    if "C" in cadence.order:
        cadence.by_label("C").add_signal(
            stg.simple_rfi_path(
                f_start=reference.get_frequency(index=600),
                drift_rate=0 * u.Hz / u.s,
                spread=24 * reference.df * u.Hz,
                spread_type=stg.SpreadType.UNIFORM,
                rfi_type=stg.RfiType.RANDOM_WALK,
                seed=77,
            ),
            stg.constant_t_profile(level=reference.get_intensity(snr=12)),
            stg.box_f_profile(width=3 * reference.df * u.Hz),
            stg.constant_bp_profile(level=1),
        )

    if "D" in cadence.order:
        cadence.by_label("D").add_signal(
            stg.constant_path(
                f_start=reference.get_frequency(index=120),
                drift_rate=-0.08 * u.Hz / u.s,
            ),
            stg.sine_t_profile(
                period=12 * reference.dt * u.s,
                amplitude=0.5 * reference.get_intensity(snr=10),
                level=reference.get_intensity(snr=10),
            ),
            stg.lorentzian_f_profile(width=10 * reference.df * u.Hz),
            stg.constant_bp_profile(level=1),
            doppler_smearing=True,
        )


def _save_cadence_plot(cadence: stg.Cadence,
                       output_path: Path,
                       *,
                       title: str,
                       slew_times: bool = False,
                       db: bool = True) -> None:
    fig = plt.figure(figsize=(12, 10))
    cadence.plot(ftype="fmid",
                 ttype="same",
                 db=db,
                 slew_times=slew_times,
                 labels=True,
                 title=True,
                 grid=False)
    plt.suptitle(title, y=0.995)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

    abacad = _make_ordered_cadence("ABACAD", seed_offset=100)
    _save_cadence_plot(abacad,
                       OUTPUT_DIR / "abacad_multi_signal.png",
                       title="ABACAD Synthetic Cadence With Multiple Signals")
    _save_cadence_plot(abacad,
                       OUTPUT_DIR / "abacad_multi_signal_slew.png",
                       title="ABACAD Synthetic Cadence With Slew Spacing",
                       slew_times=True)

    ababab = _make_ordered_cadence("ABABAB", seed_offset=300)
    _save_cadence_plot(ababab,
                       OUTPUT_DIR / "ababab_crosscheck.png",
                       title="ABABAB Crosscheck Cadence",
                       db=False)


if __name__ == "__main__":
    main()
