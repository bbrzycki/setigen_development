from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
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


SETUP = r"""
    %matplotlib inline

    from pathlib import Path

    from IPython import get_ipython
    from IPython.display import display
    import matplotlib
    _ipython = get_ipython()
    if _ipython is not None:
        _ipython.run_line_magic("matplotlib", "inline")
        matplotlib.use("module://matplotlib_inline.backend_inline", force=True)

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy import units as u

    import setigen as stg

    OUT = Path("generated")
    OUT.mkdir(exist_ok=True)

    np.set_printoptions(precision=4, suppress=True)
    print("matplotlib backend:", matplotlib.get_backend())
    print("setigen from:", stg.__file__)
"""


NOTEBOOKS = {
    "01_small_frame_signal_injection.ipynb": notebook([
        markdown(
            """
            # Small Frame Signal Injection

            This notebook starts with the classic in-memory `Frame` workflow:
            create a modest dynamic spectrum, add synthetic radiometer noise,
            estimate the local noise level, inject a narrowband drifting signal,
            and inspect the resulting spectrum and time series.

            The scientific contract we want to preserve is that the injected
            signal is represented consistently by the current frame axes,
            that SNR helpers use explicit noise statistics when supplied, and
            that derived products preserve useful provenance.
            """
        ),
        code(SETUP),
        code(
            """
            frame = stg.Frame(
                fchans=1024,
                tchans=32,
                df=2.7939677238464355 * u.Hz,
                dt=18.253611008 * u.s,
                fch1=6095.214842353016 * u.MHz,
                ascending=False,
                seed=11,
                source_name="Synthetic small frame",
            )
            frame.add_noise(x_mean=10, noise_type="chi2")

            path = stg.constant_path(
                f_start=frame.get_frequency(frame.fchans // 2),
                drift_rate=1.5 * u.Hz / u.s,
            )
            f_profile = stg.gaussian_f_profile(width=35 * u.Hz)
            noise_config = stg.NoiseEstimationConfig(
                method="sigma_clip",
                context_width=256,
                guard_width=24,
            )
            stats = frame.estimate_noise_stats(
                path=path,
                f_profile=f_profile,
                auto_bounding=True,
                truncate_below=1e-3,
                config=noise_config,
            )
            level = frame.get_intensity(snr=30, noise_stats=stats)
            stats, level
            """
        ),
        code(
            """
            signal = frame.add_signal(
                path=path,
                t_profile=stg.constant_t_profile(level=level),
                f_profile=f_profile,
                bp_profile=stg.constant_bp_profile(level=1),
                integrate_path=True,
                integrate_t_profile=True,
                integrate_f_profile=True,
                t_subsamples=8,
                f_subsamples=8,
                auto_bounding=True,
                truncate_below=1e-3,
            )

            print("signal shape:", signal.shape)
            print("nonzero signal pixels:", np.count_nonzero(signal))
            print("frame noise stats:", frame.get_noise_stats())
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(10, 4))
            frame.plot(ftype="fmid", ttype="trel", db=True, colorbar=True)
            ax.set_title("Small in-memory frame after signal injection")
            display(fig)
            plt.close(fig)
            """
        ),
        code(
            """
            spectrum = frame.spectrum(mode="sum", normalize=True)
            time_series = frame.timeseries(mode="mean")

            fig, axs = plt.subplots(1, 2, figsize=(12, 3))
            plt.sca(axs[0])
            spectrum.plot(ftype="fmid")
            axs[0].set_title("Sigma-normalized spectrum")
            plt.sca(axs[1])
            time_series.plot(ttype="trel")
            axs[1].set_title("Band-averaged time series")
            plt.tight_layout()
            display(fig)
            plt.close(fig)

            print("spectrum derived metadata:")
            spectrum.metadata["derived"]
            """
        ),
    ]),
    "02_file_write_load_loops.ipynb": notebook([
        markdown(
            """
            # File Write And Load Loops

            This notebook exercises the normal file loop:

            1. create an eager frame,
            2. write `.h5` and `.fil`,
            3. reload eagerly through `Frame(waterfall=...)`,
            4. open file-backed for bounded reads,
            5. inject into a safe copy with `Frame.open_copy(...)`.

            The contract is that saved/reloaded data match, copy-backed
            mutation leaves the source unchanged, and plots/readbacks see the
            data that were actually written to disk.
            """
        ),
        code(SETUP),
        code(
            """
            frame = stg.Frame(
                tchans=16,
                fchans=512,
                df=3 * u.Hz,
                dt=2 * u.s,
                fch1=6_000_001_533 * u.Hz,
                ascending=False,
                seed=21,
                source_name="Write-load loop",
            )
            frame.add_noise(8, noise_type="chi2")
            frame.add_constant_signal(
                f_start=frame.get_frequency(frame.fchans // 2),
                drift_rate=0.5 * frame.unit_drift_rate,
                level=5,
                width=4 * frame.df,
                f_profile_type="sinc2",
                doppler_smearing=True,
            )

            h5_path = OUT / "loop_source.h5"
            fil_path = OUT / "loop_source.fil"
            frame.save_hdf5(h5_path)
            frame.save_fil(fil_path)
            h5_path, fil_path
            """
        ),
        code(
            """
            eager_h5 = stg.Frame(waterfall=h5_path)
            eager_fil = stg.Frame(waterfall=fil_path)

            print("h5 matches:", np.allclose(eager_h5.data, frame.data))
            print("fil matches:", np.allclose(eager_fil.data, frame.data))
            print("h5 source:", eager_h5.source_name)
            print("fil source:", eager_fil.source_name)
            """
        ),
        code(
            """
            with stg.Frame.open(h5_path, mode="r") as backed:
                region = backed.read_frame(
                    f_index_range=(220, 300),
                    t_index_range=(0, backed.tchans),
                )
                print("file-backed?", backed.is_file_backed)
                print("region shape:", region.shape)
                print("region derived metadata:", region.metadata["derived"])

                fig, ax = plt.subplots(figsize=(8, 3))
                backed.plot(
                    f_index_range=(220, 300),
                    t_index_range=(0, backed.tchans),
                    db=False,
                    colorbar=True,
                )
                ax.set_title("Region read from file-backed HDF5")
                display(fig)
                plt.close(fig)
            """
        ),
        code(
            """
            injected_h5 = OUT / "loop_injected.h5"
            with stg.Frame.open_copy(h5_path, injected_h5, overwrite=True, max_chunk_bytes=4096) as backed:
                result = backed.add_signal(
                    path=stg.constant_path(
                        backed.get_frequency(backed.fchans // 2 + 60),
                        drift_rate=-0.4 * backed.unit_drift_rate,
                    ),
                    t_profile=stg.constant_t_profile(level=3),
                    f_profile=stg.box_f_profile(width=5 * backed.df),
                    auto_bounding=True,
                )
                print(result)

            original = stg.Frame(waterfall=h5_path)
            injected = stg.Frame(waterfall=injected_h5)
            print("source unchanged:", np.allclose(original.data, frame.data))
            print("output changed:", np.max(np.abs(injected.data - original.data)) > 0)
            """
        ),
    ]),
    "03_file_backed_large_frequency_ranges.ipynb": notebook([
        markdown(
            """
            # File-Backed Larger Frequency Ranges

            This notebook shows the current large-band pattern. Creating a
            completely synthetic frame from scratch is still an eager operation:
            the in-memory `Frame` owns a NumPy array. Once a large observation
            exists on disk, `Frame.open(...)` and `Frame.open_copy(...)` let us
            read, plot, reduce, and inject bounded regions without loading the
            full spectrogram.

            The point is to keep narrowband science operations local even when
            the underlying observation covers a broad frequency range.
            """
        ),
        code(SETUP),
        code(
            """
            def estimate_bytes(tchans, fchans, dtype=np.float32):
                return tchans * fchans * np.dtype(dtype).itemsize

            hypothetical_fchans = 2**22
            demo_fchans = 2**15
            tchans = 12

            print("hypothetical float32 size, GiB:", estimate_bytes(tchans, hypothetical_fchans) / 1024**3)
            print("demo float64 scratch size, MiB:", estimate_bytes(tchans, demo_fchans, np.float64) / 1024**2)
            """
        ),
        code(
            """
            scratch = stg.Frame(
                tchans=tchans,
                fchans=demo_fchans,
                df=1 * u.Hz,
                dt=1 * u.s,
                fch1=(6e9 + demo_fchans - 1) * u.Hz,
                ascending=False,
                seed=31,
                source_name="Large-band scratch demo",
            )
            scratch.data[:] = 100.0
            large_source = OUT / "large_band_source.h5"
            scratch.save_hdf5(large_source)
            print("wrote", large_source)
            """
        ),
        code(
            """
            with stg.Frame.open(large_source, mode="r") as backed:
                print("shape:", backed.shape)
                print("is file-backed:", backed.is_file_backed)
                print("full materialization would be MiB:", backed.tchans * backed.fchans * 8 / 1024**2)

                small = backed.read_frame(
                    f_index_range=(demo_fchans // 2 - 64, demo_fchans // 2 + 64),
                    t_index_range=(0, backed.tchans),
                )
                print("bounded region shape:", small.shape)
            """
        ),
        code(
            """
            large_injected = OUT / "large_band_injected.h5"
            with stg.Frame.open_copy(
                large_source,
                large_injected,
                overwrite=True,
                max_chunk_bytes=4096,
            ) as backed:
                stats = backed.estimate_noise_stats(
                    bounding_f_range=(
                        backed.get_frequency(backed.fchans // 2 - 2),
                        backed.get_frequency(backed.fchans // 2 + 2),
                    ),
                    t_index_range=(0, backed.tchans),
                    config=stg.NoiseEstimationConfig(context_width=128, guard_width=8),
                )
                result = backed.add_signal(
                    path=stg.constant_path(
                        backed.get_frequency(backed.fchans // 2),
                        drift_rate=0.25 * backed.unit_drift_rate,
                    ),
                    t_profile=stg.constant_t_profile(level=25),
                    f_profile=stg.gaussian_f_profile(width=8 * backed.df),
                    auto_bounding=True,
                    truncate_below=1e-3,
                )
                print("noise stats:", stats)
                print("injection result:", result)

                fig, ax = plt.subplots(figsize=(9, 3))
                backed.plot(
                    f_index_range=(backed.fchans // 2 - 80, backed.fchans // 2 + 80),
                    t_index_range=(0, backed.tchans),
                    db=False,
                    colorbar=True,
                )
                ax.set_title("Patched region read back from file-backed output")
                display(fig)
                plt.close(fig)
            """
        ),
        code(
            """
            with stg.Frame.open(large_injected, mode="r", max_chunk_bytes=4096) as backed:
                spectrum = backed.spectrum(
                    mode="sum",
                    f_index_range=(backed.fchans // 2 - 128, backed.fchans // 2 + 128),
                    max_chunk_bytes=4096,
                )
                ts = backed.timeseries(
                    mode="mean",
                    f_index_range=(backed.fchans // 2 - 128, backed.fchans // 2 + 128),
                    max_chunk_bytes=4096,
                )

            print("spectrum shape:", spectrum.shape)
            print("time series shape:", ts.shape)
            print("spectrum derived metadata:", spectrum.metadata["derived"])
            """
        ),
    ]),
    "04_cadence_signal_workflow.ipynb": notebook([
        markdown(
            """
            # Cadence Signal Workflow

            Cadences encode a sequence of compatible observations. This
            notebook builds an ordered ABACAD-style cadence, injects only into
            the on-target frames, checks the off-target frames, and consolidates
            the result for a single dynamic-spectrum view.
            """
        ),
        code(SETUP),
        code("from astropy.time import Time"),
        code(
            """
            mjd_start = 60000
            obs_length = 120
            slew_time = 15
            starts = [Time(mjd_start, format="mjd").unix]
            for _ in range(1, 6):
                starts.append(starts[-1] + obs_length + slew_time)

            frames = []
            for i, start in enumerate(starts):
                fr = stg.Frame(
                    fchans=512,
                    tchans=16,
                    df=2 * u.Hz,
                    dt=(obs_length / 16) * u.s,
                    fch1=6_000_001_022 * u.Hz,
                    ascending=False,
                    t_start=start,
                    source_name=f"Obs{i}",
                    seed=i,
                )
                fr.add_noise(10, noise_type="chi2")
                frames.append(fr)

            cadence = stg.OrderedCadence(frames, order="ABACAD")
            [(frame.source_name, frame.metadata.get("order_label")) for frame in cadence]
            """
        ),
        code(
            """
            on_target = cadence.by_label("A")
            reference = cadence[0]
            on_target.add_signal(
                stg.constant_path(
                    f_start=reference.get_frequency(reference.fchans // 2),
                    drift_rate=0.2 * reference.unit_drift_rate,
                ),
                stg.constant_t_profile(level=reference.get_intensity(snr=80)),
                stg.sinc2_f_profile(width=2 * reference.df, trunc=False),
                stg.constant_bp_profile(level=1),
                doppler_smearing=True,
            )

            peak_by_frame = [float(np.max(stg.spectrum(frame, mode="sum").data)) for frame in cadence]
            list(zip([frame.metadata.get("order_label") for frame in cadence], peak_by_frame))
            """
        ),
        code(
            """
            fig = plt.figure(figsize=(12, 6))
            cadence.plot(ftype="fmid", slew_times=True, db=True)
            plt.suptitle("Ordered cadence with synthetic signal in A frames")
            display(fig)
            plt.close(fig)
            """
        ),
        code(
            """
            consolidated = cadence.consolidate()
            print("consolidated shape:", consolidated.shape)
            print("cadence observation span, s:", cadence.obs_range)

            fig, ax = plt.subplots(figsize=(10, 4))
            consolidated.plot(ftype="fmid", ttype="trel", db=True, colorbar=True)
            ax.set_title("Consolidated cadence frame")
            display(fig)
            plt.close(fig)
            """
        ),
    ]),
    "05_spectrum_timeseries_products.ipynb": notebook([
        markdown(
            """
            # Spectrum And Time-Series Products

            This notebook focuses on the singleton-axis products that come from
            common reductions:

            - `Spectrum`: shape `(1, fchans)`, collapse time.
            - `TimeSeries`: shape `(tchans, 1)`, collapse frequency.

            The new derived metadata records where the product came from and
            what operation produced it.
            """
        ),
        code(SETUP),
        code(
            """
            frame = stg.Frame(
                fchans=1024,
                tchans=32,
                df=1 * u.Hz,
                dt=1 * u.s,
                fch1=6e9 + 1023,
                ascending=False,
                seed=41,
                source_name="Derived product demo",
            )
            frame.add_noise(5, noise_type="chi2")
            drift_rate = 0.6 * frame.unit_drift_rate
            frame.add_constant_signal(
                f_start=frame.get_frequency(420),
                drift_rate=drift_rate,
                level=8,
                width=4 * frame.df,
                f_profile_type="gaussian",
            )
            frame.add_metadata({"expected_drift_rate": float(drift_rate)})

            fig, ax = plt.subplots(figsize=(9, 4))
            frame.plot(ftype="px", db=True, colorbar=True)
            ax.set_title("Input frame")
            display(fig)
            plt.close(fig)
            """
        ),
        code(
            """
            dd = stg.dedrift(frame, drift_rate=drift_rate)
            spectrum = dd.spectrum(mode="sum", normalize=True)
            peak_index = int(np.argmax(spectrum.data))

            fig, ax = plt.subplots(figsize=(10, 3))
            spectrum.plot(ftype="px")
            ax.axvline(peak_index, color="k", linestyle="--", alpha=0.7)
            ax.set_title("Dedrifted, summed, sigma-normalized spectrum")
            display(fig)
            plt.close(fig)

            print("peak index:", peak_index)
            print("dedrift metadata:", dd.metadata["derived"])
            print("spectrum metadata:", spectrum.metadata["derived"])
            """
        ),
        code(
            """
            f_center = dd.get_frequency(peak_index)
            ts = dd.timeseries(
                mode="mean",
                f_range=(f_center - 8 * dd.df, f_center + 8 * dd.df),
            )

            fig, ax = plt.subplots(figsize=(8, 3))
            ts.plot(ttype="trel")
            ax.set_title("Mean power in a narrow band around the dedrifted peak")
            display(fig)
            plt.close(fig)

            print("time series shape:", ts.shape)
            print("time series metadata:", ts.metadata["derived"])
            """
        ),
        code(
            """
            one_d_spectrum = stg.Spectrum(df=dd.df, dt=dd.obs_length, fch1=dd.fch1, data=spectrum.array())
            one_d_timeseries = stg.TimeSeries(df=ts.df, dt=ts.dt, fch1=ts.fch1, data=ts.array())
            print("1D spectrum coerced shape:", one_d_spectrum.shape)
            print("1D time series coerced shape:", one_d_timeseries.shape)
            """
        ),
        code(
            """
            h5_path = OUT / "derived_product_source.h5"
            frame.save_hdf5(h5_path)
            with stg.Frame.open(h5_path, mode="r", max_chunk_bytes=4096) as backed:
                backed_spectrum = backed.spectrum(
                    mode="sum",
                    f_index_range=(350, 500),
                    max_chunk_bytes=4096,
                )
                backed_ts = backed.timeseries(
                    mode="mean",
                    f_index_range=(350, 500),
                    max_chunk_bytes=4096,
                )

            print("file-backed spectrum:", backed_spectrum.shape, backed_spectrum.metadata["derived"])
            print("file-backed time series:", backed_ts.shape, backed_ts.metadata["derived"])
            """
        ),
    ]),
    "06_file_backed_performance_profile.ipynb": notebook([
        markdown(
            """
            # File-Backed Performance Profile

            This notebook profiles a fixed synthetic HDF5 spectrogram so file-backed
            operations can be compared directly against the default eager
            `Frame(waterfall=...)` path.

            The timings are intentionally simple wall-clock measurements. They are
            useful for internal reporting and regression checks, but they are not a
            substitute for a dedicated benchmark suite. Run from a fresh kernel and
            keep the demo dimensions fixed when comparing branches or machines.
            """
        ),
        code(SETUP),
        code(
            """
            import gc
            import shutil
            import time
            import tracemalloc

            import pandas as pd
            import psutil

            PROCESS = psutil.Process()

            PROFILE_TCHANS = 24
            PROFILE_FCHANS = 2**16
            PROFILE_DF = 1 * u.Hz
            PROFILE_DT = 1 * u.s
            PROFILE_FCH1 = (6e9 + PROFILE_FCHANS - 1) * u.Hz
            PROFILE_CHUNK_BYTES = 512 * 1024
            PROFILE_CONTEXT_WIDTH = 2048
            PROFILE_GUARD_WIDTH = 64
            PROFILE_BOUND_HALF_WIDTH = 256
            REBUILD_SOURCE = True

            source_path = OUT / "profile_source.h5"

            def rss_mib():
                return PROCESS.memory_info().rss / 1024**2

            def profile_case(label, func, records, **metadata):
                gc.collect()
                rss_before = rss_mib()
                tracemalloc.start()
                t0 = time.perf_counter()
                result = func()
                elapsed = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rss_after = rss_mib()

                record = {
                    "case": label,
                    "elapsed_s": elapsed,
                    "rss_delta_mib": rss_after - rss_before,
                    "python_peak_mib": peak / 1024**2,
                }
                record.update(metadata)
                records.append(record)
                return result

            def show_table(records, sort_by=None):
                table = pd.DataFrame(records)
                if sort_by is not None:
                    table = table.sort_values(sort_by)
                display(table.round({
                    "elapsed_s": 4,
                    "rss_delta_mib": 2,
                    "python_peak_mib": 2,
                }))
                return table

            def describe_shape(tchans, fchans, dtype=np.float64):
                bytes_ = tchans * fchans * np.dtype(dtype).itemsize
                return bytes_ / 1024**2

            print("profile shape:", (PROFILE_TCHANS, PROFILE_FCHANS))
            print("float64 materialized array, MiB:", describe_shape(PROFILE_TCHANS, PROFILE_FCHANS))
            print("chunk budget, MiB:", PROFILE_CHUNK_BYTES / 1024**2)
            """
        ),
        code(
            """
            if REBUILD_SOURCE or not source_path.exists():
                source = stg.Frame(
                    tchans=PROFILE_TCHANS,
                    fchans=PROFILE_FCHANS,
                    df=PROFILE_DF,
                    dt=PROFILE_DT,
                    fch1=PROFILE_FCH1,
                    ascending=False,
                    seed=61,
                    source_name="Performance profile source",
                )
                source.add_noise(10, noise_type="chi2")
                source.save_hdf5(source_path)

            print("source:", source_path)
            print("source size, MiB:", source_path.stat().st_size / 1024**2)
            """
        ),
        code(
            """
            read_records = []
            f0 = PROFILE_FCHANS // 2 - PROFILE_BOUND_HALF_WIDTH
            f1 = PROFILE_FCHANS // 2 + PROFILE_BOUND_HALF_WIDTH

            def eager_full_load():
                return stg.Frame(waterfall=source_path)

            def file_backed_bounded_read():
                with stg.Frame.open(source_path, mode="r", max_chunk_bytes=PROFILE_CHUNK_BYTES) as backed:
                    return backed.read_frame(
                        f_index_range=(f0, f1),
                        t_index_range=(0, backed.tchans),
                    )

            def file_backed_full_read():
                with stg.Frame.open(source_path, mode="r", max_chunk_bytes=PROFILE_CHUNK_BYTES) as backed:
                    return backed.read_frame()

            eager = profile_case(
                "eager full load: Frame(waterfall=...)",
                eager_full_load,
                read_records,
                operation="read",
                fchans_read=PROFILE_FCHANS,
                tchans_read=PROFILE_TCHANS,
            )
            bounded = profile_case(
                "file-backed bounded read_frame",
                file_backed_bounded_read,
                read_records,
                operation="read",
                fchans_read=f1 - f0,
                tchans_read=PROFILE_TCHANS,
            )
            full_backed = profile_case(
                "file-backed full read_frame",
                file_backed_full_read,
                read_records,
                operation="read",
                fchans_read=PROFILE_FCHANS,
                tchans_read=PROFILE_TCHANS,
            )

            print("bounded shape:", bounded.shape)
            print("full file-backed read matches eager:", np.allclose(full_backed.data, eager.data))
            read_table = show_table(read_records)
            """
        ),
        code(
            """
            write_records = []

            def eager_rewrite():
                frame = stg.Frame(waterfall=source_path)
                output = OUT / "profile_eager_rewrite.h5"
                frame.save_hdf5(output)
                return output

            def file_backed_copy():
                output = OUT / "profile_file_backed_copy.h5"
                with stg.Frame.open_copy(source_path, output, overwrite=True, max_chunk_bytes=PROFILE_CHUNK_BYTES):
                    pass
                return output

            eager_out = profile_case(
                "eager load + save_hdf5",
                eager_rewrite,
                write_records,
                operation="write/copy",
                materializes_full_frame=True,
            )
            copy_out = profile_case(
                "file-backed open_copy",
                file_backed_copy,
                write_records,
                operation="write/copy",
                materializes_full_frame=False,
            )

            print("eager rewrite size, MiB:", eager_out.stat().st_size / 1024**2)
            print("file-backed copy size, MiB:", copy_out.stat().st_size / 1024**2)
            write_table = show_table(write_records)
            """
        ),
        code(
            """
            noise_records = []
            noise_config = stg.NoiseEstimationConfig(
                method="sigma_clip",
                context_width=PROFILE_CONTEXT_WIDTH,
                guard_width=PROFILE_GUARD_WIDTH,
            )

            def local_noise_kwargs(frame):
                return {
                    "path": stg.constant_path(
                        f_start=frame.get_frequency(frame.fchans // 2),
                        drift_rate=0.25 * frame.unit_drift_rate,
                    ),
                    "f_profile": stg.gaussian_f_profile(width=8 * frame.df),
                    "auto_bounding": True,
                    "truncate_below": 1e-3,
                    "config": noise_config,
                }

            def eager_noise_estimation():
                frame = stg.Frame(waterfall=source_path)
                return frame.estimate_noise_stats(**local_noise_kwargs(frame))

            def file_backed_noise_estimation():
                with stg.Frame.open(source_path, mode="r", max_chunk_bytes=PROFILE_CHUNK_BYTES) as backed:
                    return backed.estimate_noise_stats(**local_noise_kwargs(backed))

            eager_noise = profile_case(
                "eager full load + local noise stats",
                eager_noise_estimation,
                noise_records,
                operation="noise estimation",
                materializes_full_frame=True,
            )
            backed_noise = profile_case(
                "file-backed local noise stats",
                file_backed_noise_estimation,
                noise_records,
                operation="noise estimation",
                materializes_full_frame=False,
            )

            print("eager noise stats:", eager_noise)
            print("file-backed noise stats:", backed_noise)
            noise_table = show_table(noise_records)
            """
        ),
        code(
            """
            try:
                with stg.Frame.open_copy(
                    source_path,
                    OUT / "profile_noise_mutation_attempt.h5",
                    overwrite=True,
                    max_chunk_bytes=PROFILE_CHUNK_BYTES,
                ) as backed:
                    backed.add_noise(10, noise_type="chi2")
            except NotImplementedError as exc:
                print("file-backed add_noise is intentionally unsupported:")
                print(exc)
            """
        ),
        code(
            """
            signal_records = []

            def injection_kwargs(frame, *, auto_bounding):
                return {
                    "path": stg.constant_path(
                        f_start=frame.get_frequency(frame.fchans // 2),
                        drift_rate=0.25 * frame.unit_drift_rate,
                    ),
                    "t_profile": stg.constant_t_profile(level=25),
                    "f_profile": stg.gaussian_f_profile(width=8 * frame.df),
                    "bp_profile": stg.constant_bp_profile(level=1),
                    "auto_bounding": auto_bounding,
                    "truncate_below": 1e-3,
                }

            def affected_fchans_from_signal(signal):
                nonzero = np.flatnonzero(np.any(signal != 0, axis=0))
                return int(nonzero.size)

            def eager_injection(auto_bounding):
                frame = stg.Frame(waterfall=source_path)
                signal = frame.add_signal(**injection_kwargs(frame, auto_bounding=auto_bounding))
                output = OUT / f"profile_eager_injected_auto_{auto_bounding}.h5"
                frame.save_hdf5(output)
                return {
                    "output": output,
                    "affected_fchans": affected_fchans_from_signal(signal),
                    "returned_signal_shape": signal.shape,
                }

            def file_backed_injection(auto_bounding):
                output = OUT / f"profile_file_backed_injected_auto_{auto_bounding}.h5"
                with stg.Frame.open_copy(
                    source_path,
                    output,
                    overwrite=True,
                    max_chunk_bytes=PROFILE_CHUNK_BYTES,
                ) as backed:
                    result = backed.add_signal(
                        **injection_kwargs(backed, auto_bounding=auto_bounding),
                        max_chunk_bytes=PROFILE_CHUNK_BYTES,
                    )
                return {"output": output, "result": result}

            for auto_bounding in (False, True):
                eager_result = profile_case(
                    f"eager injection auto_bounding={auto_bounding}",
                    lambda auto_bounding=auto_bounding: eager_injection(auto_bounding),
                    signal_records,
                    operation="signal injection",
                    storage="eager",
                    auto_bounding=auto_bounding,
                )
                signal_records[-1]["affected_fchans"] = eager_result["affected_fchans"]
                signal_records[-1]["returned_signal_shape"] = eager_result["returned_signal_shape"]

                backed_result = profile_case(
                    f"file-backed injection auto_bounding={auto_bounding}",
                    lambda auto_bounding=auto_bounding: file_backed_injection(auto_bounding),
                    signal_records,
                    operation="signal injection",
                    storage="file-backed",
                    auto_bounding=auto_bounding,
                )
                fb_result = backed_result["result"]
                signal_records[-1]["affected_fchans"] = fb_result.frequency_slice.stop - fb_result.frequency_slice.start
                signal_records[-1]["time_chunks"] = fb_result.time_chunks
                signal_records[-1]["max_chunk_shape"] = fb_result.max_chunk_shape

            signal_table = show_table(signal_records)
            """
        ),
        code(
            """
            all_tables = {
                "read": read_table,
                "write": write_table,
                "noise": noise_table,
                "signal": signal_table,
            }

            for name, table in all_tables.items():
                path = OUT / f"profile_{name}_table.csv"
                table.to_csv(path, index=False)
                print("wrote", path)

            summary = pd.concat(
                [table.assign(section=name) for name, table in all_tables.items()],
                ignore_index=True,
                sort=False,
            )
            display(summary[[
                "section",
                "case",
                "elapsed_s",
                "rss_delta_mib",
                "python_peak_mib",
                "affected_fchans",
                "fchans_read",
                "materializes_full_frame",
            ]].round({
                "elapsed_s": 4,
                "rss_delta_mib": 2,
                "python_peak_mib": 2,
            }))
            """
        ),
        markdown(
            """
            ## Analysis

            The main question is not whether file-backed access is universally
            faster. It should be comparable to eager loading when the whole
            observation must be materialized. The expected win is when the
            scientific operation is local in frequency: narrowband reads,
            local noise/SNR estimation, and bounded signal injection should
            touch only the relevant channels.
            """
        ),
        code(
            """
            def one(table, case):
                match = table.loc[table["case"] == case]
                if len(match) != 1:
                    raise ValueError(f"Expected one row for {case!r}, found {len(match)}")
                return match.iloc[0]

            def speedup(reference, candidate):
                return float(reference["elapsed_s"] / candidate["elapsed_s"])

            def memory_ratio(reference, candidate, column):
                ref = float(reference[column])
                cand = float(candidate[column])
                if cand == 0:
                    return np.inf
                return ref / cand

            read_eager = one(read_table, "eager full load: Frame(waterfall=...)")
            read_bounded = one(read_table, "file-backed bounded read_frame")
            read_full_backed = one(read_table, "file-backed full read_frame")

            write_eager = one(write_table, "eager load + save_hdf5")
            write_copy = one(write_table, "file-backed open_copy")

            noise_eager = one(noise_table, "eager full load + local noise stats")
            noise_backed = one(noise_table, "file-backed local noise stats")

            sig_eager_unbounded = one(signal_table, "eager injection auto_bounding=False")
            sig_backed_unbounded = one(signal_table, "file-backed injection auto_bounding=False")
            sig_eager_bounded = one(signal_table, "eager injection auto_bounding=True")
            sig_backed_bounded = one(signal_table, "file-backed injection auto_bounding=True")

            bounded_read_fraction = read_bounded["fchans_read"] / PROFILE_FCHANS
            bounded_signal_fraction = sig_backed_bounded["affected_fchans"] / PROFILE_FCHANS

            analysis_rows = [
                {
                    "comparison": "bounded file-backed read vs eager full load",
                    "wall_time_speedup": speedup(read_eager, read_bounded),
                    "python_peak_ratio": memory_ratio(read_eager, read_bounded, "python_peak_mib"),
                    "rss_delta_ratio": memory_ratio(read_eager, read_bounded, "rss_delta_mib"),
                    "channels_touched_fraction": bounded_read_fraction,
                },
                {
                    "comparison": "full file-backed read vs eager full load",
                    "wall_time_speedup": speedup(read_eager, read_full_backed),
                    "python_peak_ratio": memory_ratio(read_eager, read_full_backed, "python_peak_mib"),
                    "rss_delta_ratio": memory_ratio(read_eager, read_full_backed, "rss_delta_mib"),
                    "channels_touched_fraction": 1.0,
                },
                {
                    "comparison": "file-backed open_copy vs eager load + save",
                    "wall_time_speedup": speedup(write_eager, write_copy),
                    "python_peak_ratio": memory_ratio(write_eager, write_copy, "python_peak_mib"),
                    "rss_delta_ratio": memory_ratio(write_eager, write_copy, "rss_delta_mib"),
                    "channels_touched_fraction": 1.0,
                },
                {
                    "comparison": "file-backed local noise stats vs eager local noise stats",
                    "wall_time_speedup": speedup(noise_eager, noise_backed),
                    "python_peak_ratio": memory_ratio(noise_eager, noise_backed, "python_peak_mib"),
                    "rss_delta_ratio": memory_ratio(noise_eager, noise_backed, "rss_delta_mib"),
                    "channels_touched_fraction": (PROFILE_CONTEXT_WIDTH * 2 + 32) / PROFILE_FCHANS,
                },
                {
                    "comparison": "file-backed bounded injection vs eager bounded injection",
                    "wall_time_speedup": speedup(sig_eager_bounded, sig_backed_bounded),
                    "python_peak_ratio": memory_ratio(sig_eager_bounded, sig_backed_bounded, "python_peak_mib"),
                    "rss_delta_ratio": memory_ratio(sig_eager_bounded, sig_backed_bounded, "rss_delta_mib"),
                    "channels_touched_fraction": bounded_signal_fraction,
                },
                {
                    "comparison": "file-backed unbounded injection vs eager unbounded injection",
                    "wall_time_speedup": speedup(sig_eager_unbounded, sig_backed_unbounded),
                    "python_peak_ratio": memory_ratio(sig_eager_unbounded, sig_backed_unbounded, "python_peak_mib"),
                    "rss_delta_ratio": memory_ratio(sig_eager_unbounded, sig_backed_unbounded, "rss_delta_mib"),
                    "channels_touched_fraction": 1.0,
                },
                {
                    "comparison": "file-backed bounded injection vs file-backed unbounded injection",
                    "wall_time_speedup": speedup(sig_backed_unbounded, sig_backed_bounded),
                    "python_peak_ratio": memory_ratio(sig_backed_unbounded, sig_backed_bounded, "python_peak_mib"),
                    "rss_delta_ratio": memory_ratio(sig_backed_unbounded, sig_backed_bounded, "rss_delta_mib"),
                    "channels_touched_fraction": bounded_signal_fraction,
                },
            ]

            analysis_table = pd.DataFrame(analysis_rows)
            display(analysis_table.round({
                "wall_time_speedup": 2,
                "python_peak_ratio": 2,
                "rss_delta_ratio": 2,
                "channels_touched_fraction": 5,
            }))

            analysis_path = OUT / "profile_analysis_table.csv"
            analysis_table.to_csv(analysis_path, index=False)
            print("wrote", analysis_path)
            """
        ),
        code(
            """
            from IPython.display import Markdown

            def fmt_speed(value):
                return f"{value:.1f}x"

            def fmt_pct(value):
                return f"{100 * value:.3f}%"

            read_speed = speedup(read_eager, read_bounded)
            noise_speed = speedup(noise_eager, noise_backed)
            copy_speed = speedup(write_eager, write_copy)
            bounded_signal_speed = speedup(sig_eager_bounded, sig_backed_bounded)
            unbounded_signal_speed = speedup(sig_eager_unbounded, sig_backed_unbounded)
            file_backed_bound_speed = speedup(sig_backed_unbounded, sig_backed_bounded)

            interpretation = f'''
            ### Interpretation

            - Bounded file-backed reads touched {int(read_bounded["fchans_read"]):,} of
              {PROFILE_FCHANS:,} channels ({fmt_pct(bounded_read_fraction)}) and were
              {fmt_speed(read_speed)} faster than eager full loading in this run.

            - Full file-backed reads were intentionally similar to eager full loading
              ({fmt_speed(speedup(read_eager, read_full_backed))} relative speed).
              File backing is not magic when the operation truly needs every channel;
              the advantage appears when the read/write region is narrow.

            - `Frame.open_copy(...)` avoided materializing the full observation and was
              {fmt_speed(copy_speed)} faster than eager load plus `save_hdf5(...)`.
              This is the right pattern for safe copy-backed mutation before injection.

            - File-backed local noise estimation produced the same noise statistics as
              eager local noise estimation while avoiding the full-frame load. In this
              run it was {fmt_speed(noise_speed)} faster. This is the SNR-estimation
              path we care about scientifically because local context is needed around
              the signal track.

            - Full-frame file-backed synthetic noise mutation is intentionally not
              implemented. Adding noise to an entire observation is a global synthetic
              operation and would require writing the whole file; for that workflow, use
              an eager synthetic frame or a future dedicated full-file generation path.

            - With `auto_bounding=True`, file-backed injection touched
              {int(sig_backed_bounded["affected_fchans"]):,} of {PROFILE_FCHANS:,}
              channels ({fmt_pct(bounded_signal_fraction)}) and was
              {fmt_speed(bounded_signal_speed)} faster than eager bounded injection.
              It was also {fmt_speed(file_backed_bound_speed)} faster than file-backed
              unbounded injection.

            - With `auto_bounding=False`, file-backed injection still avoids returning
              a full signal array, but it must read and write all frequency channels in
              chunks. That mode is mostly a correctness fallback; bounded injection is
              the performance-critical path for narrowband signals.

            - RSS deltas and wall times are machine- and cache-dependent. The most
              stable reporting quantities are the channel fraction touched, whether the
              operation materializes the full frame, and Python peak allocation from
              `tracemalloc`.
            '''

            display(Markdown(interpretation))
            """
        ),
    ]),
    "07_one_gib_file_backed_profile.ipynb": notebook([
        markdown(
            """
            # One-GiB File-Backed Performance Profile

            This notebook repeats the file-backed/eager comparisons with a
            large synthetic HDF5 source. By default it targets a 1 GiB
            materialized dynamic spectrum:

            - 128 time bins
            - 1,048,576 frequency channels
            - float64 in-memory frame representation

            The HDF5 file size is reported after writing because HDF5 filters
            and datatype details affect the exact on-disk size. The source file
            is kept under `generated/`; temporary benchmark outputs are deleted
            by default so the notebook does not leave several extra GiB behind.
            """
        ),
        code(SETUP),
        code(
            """
            import gc
            import os
            import time
            import tracemalloc

            import pandas as pd
            import psutil

            PROCESS = psutil.Process()
            FAST_LARGE_PROFILE = os.environ.get("SETIGEN_FAST_LARGE_PROFILE") == "1"

            if FAST_LARGE_PROFILE:
                PROFILE_TCHANS = 16
                PROFILE_FCHANS = 2**15
                PROFILE_CHUNK_BYTES = 512 * 1024
                PROFILE_CONTEXT_WIDTH = 2048
                PROFILE_GUARD_WIDTH = 64
                PROFILE_BOUND_HALF_WIDTH = 256
                SOURCE_STEM = "one_gib_profile_fast"
            else:
                PROFILE_TCHANS = 128
                PROFILE_FCHANS = 2**20
                PROFILE_CHUNK_BYTES = 64 * 1024**2
                PROFILE_CONTEXT_WIDTH = 8192
                PROFILE_GUARD_WIDTH = 128
                PROFILE_BOUND_HALF_WIDTH = 2048
                SOURCE_STEM = "one_gib_profile"

            PROFILE_DF = 1 * u.Hz
            PROFILE_DT = 1 * u.s
            PROFILE_FCH1 = (6e9 + PROFILE_FCHANS - 1) * u.Hz

            REBUILD_SOURCE = False
            KEEP_LARGE_OUTPUTS = False
            RUN_UNBOUNDED_SIGNAL = True

            source_path = OUT / f"{SOURCE_STEM}_source.h5"

            def rss_mib():
                return PROCESS.memory_info().rss / 1024**2

            def materialized_gib(dtype=np.float64):
                bytes_ = PROFILE_TCHANS * PROFILE_FCHANS * np.dtype(dtype).itemsize
                return bytes_ / 1024**3

            def remove_if_temporary(path):
                path = Path(path)
                if not KEEP_LARGE_OUTPUTS and path.exists():
                    path.unlink()

            def profile_case(label, func, records, **metadata):
                gc.collect()
                rss_before = rss_mib()
                tracemalloc.start()
                t0 = time.perf_counter()
                result = func()
                elapsed = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rss_after = rss_mib()

                record = {
                    "case": label,
                    "elapsed_s": elapsed,
                    "rss_delta_mib": rss_after - rss_before,
                    "python_peak_mib": peak / 1024**2,
                }
                record.update(metadata)
                records.append(record)
                return result

            def show_table(records):
                table = pd.DataFrame(records)
                display(table.round({
                    "elapsed_s": 4,
                    "rss_delta_mib": 2,
                    "python_peak_mib": 2,
                    "source_size_mib": 2,
                    "output_size_mib": 2,
                }))
                return table

            print("fast validation mode:", FAST_LARGE_PROFILE)
            print("profile shape:", (PROFILE_TCHANS, PROFILE_FCHANS))
            print("materialized float64 size, GiB:", materialized_gib())
            print("chunk budget, MiB:", PROFILE_CHUNK_BYTES / 1024**2)
            print("source path:", source_path)
            """
        ),
        code(
            """
            if REBUILD_SOURCE or not source_path.exists():
                print("creating source; this can take time and several GiB of transient RAM in full mode")
                source = stg.Frame(
                    tchans=PROFILE_TCHANS,
                    fchans=PROFILE_FCHANS,
                    df=PROFILE_DF,
                    dt=PROFILE_DT,
                    fch1=PROFILE_FCH1,
                    ascending=False,
                    seed=71,
                    source_name="One-GiB performance profile source",
                )
                source.add_noise(10, noise_type="chi2")
                source.save_hdf5(source_path)
                del source
                gc.collect()

            source_size_mib = source_path.stat().st_size / 1024**2
            print("source HDF5 size, MiB:", source_size_mib)
            print("source HDF5 size, GiB:", source_size_mib / 1024)
            """
        ),
        code(
            """
            read_records = []
            f0 = PROFILE_FCHANS // 2 - PROFILE_BOUND_HALF_WIDTH
            f1 = PROFILE_FCHANS // 2 + PROFILE_BOUND_HALF_WIDTH

            def eager_full_load():
                return stg.Frame(waterfall=source_path)

            def file_backed_bounded_read():
                with stg.Frame.open(source_path, mode="r", max_chunk_bytes=PROFILE_CHUNK_BYTES) as backed:
                    return backed.read_frame(
                        f_index_range=(f0, f1),
                        t_index_range=(0, backed.tchans),
                    )

            def file_backed_full_read():
                with stg.Frame.open(source_path, mode="r", max_chunk_bytes=PROFILE_CHUNK_BYTES) as backed:
                    return backed.read_frame()

            eager = profile_case(
                "eager full load: Frame(waterfall=...)",
                eager_full_load,
                read_records,
                operation="read",
                fchans_read=PROFILE_FCHANS,
                tchans_read=PROFILE_TCHANS,
                source_size_mib=source_size_mib,
            )
            bounded = profile_case(
                "file-backed bounded read_frame",
                file_backed_bounded_read,
                read_records,
                operation="read",
                fchans_read=f1 - f0,
                tchans_read=PROFILE_TCHANS,
                source_size_mib=source_size_mib,
            )
            full_backed = profile_case(
                "file-backed full read_frame",
                file_backed_full_read,
                read_records,
                operation="read",
                fchans_read=PROFILE_FCHANS,
                tchans_read=PROFILE_TCHANS,
                source_size_mib=source_size_mib,
            )

            print("bounded shape:", bounded.shape)
            print("full file-backed read matches eager:", np.allclose(full_backed.data, eager.data))
            read_table = show_table(read_records)
            del eager, bounded, full_backed
            gc.collect()
            """
        ),
        code(
            """
            write_records = []

            def eager_rewrite():
                output = OUT / f"{SOURCE_STEM}_eager_rewrite.h5"
                frame = stg.Frame(waterfall=source_path)
                frame.save_hdf5(output)
                output_size_mib = output.stat().st_size / 1024**2
                del frame
                remove_if_temporary(output)
                return output_size_mib

            def file_backed_copy():
                output = OUT / f"{SOURCE_STEM}_file_backed_copy.h5"
                with stg.Frame.open_copy(source_path, output, overwrite=True, max_chunk_bytes=PROFILE_CHUNK_BYTES):
                    pass
                output_size_mib = output.stat().st_size / 1024**2
                remove_if_temporary(output)
                return output_size_mib

            eager_size = profile_case(
                "eager load + save_hdf5",
                eager_rewrite,
                write_records,
                operation="write/copy",
                materializes_full_frame=True,
                source_size_mib=source_size_mib,
            )
            write_records[-1]["output_size_mib"] = eager_size

            copy_size = profile_case(
                "file-backed open_copy",
                file_backed_copy,
                write_records,
                operation="write/copy",
                materializes_full_frame=False,
                source_size_mib=source_size_mib,
            )
            write_records[-1]["output_size_mib"] = copy_size

            write_table = show_table(write_records)
            """
        ),
        code(
            """
            noise_records = []
            noise_config = stg.NoiseEstimationConfig(
                method="sigma_clip",
                context_width=PROFILE_CONTEXT_WIDTH,
                guard_width=PROFILE_GUARD_WIDTH,
            )

            def local_noise_kwargs(frame):
                return {
                    "path": stg.constant_path(
                        f_start=frame.get_frequency(frame.fchans // 2),
                        drift_rate=0.25 * frame.unit_drift_rate,
                    ),
                    "f_profile": stg.gaussian_f_profile(width=8 * frame.df),
                    "auto_bounding": True,
                    "truncate_below": 1e-3,
                    "config": noise_config,
                }

            def eager_noise_estimation():
                frame = stg.Frame(waterfall=source_path)
                stats = frame.estimate_noise_stats(**local_noise_kwargs(frame))
                del frame
                return stats

            def file_backed_noise_estimation():
                with stg.Frame.open(source_path, mode="r", max_chunk_bytes=PROFILE_CHUNK_BYTES) as backed:
                    return backed.estimate_noise_stats(**local_noise_kwargs(backed))

            eager_noise = profile_case(
                "eager full load + local noise stats",
                eager_noise_estimation,
                noise_records,
                operation="noise estimation",
                materializes_full_frame=True,
                source_size_mib=source_size_mib,
            )
            backed_noise = profile_case(
                "file-backed local noise stats",
                file_backed_noise_estimation,
                noise_records,
                operation="noise estimation",
                materializes_full_frame=False,
                source_size_mib=source_size_mib,
            )

            print("eager noise stats:", eager_noise)
            print("file-backed noise stats:", backed_noise)
            noise_table = show_table(noise_records)
            """
        ),
        code(
            """
            try:
                output = OUT / f"{SOURCE_STEM}_noise_mutation_attempt.h5"
                with stg.Frame.open_copy(
                    source_path,
                    output,
                    overwrite=True,
                    max_chunk_bytes=PROFILE_CHUNK_BYTES,
                ) as backed:
                    backed.add_noise(10, noise_type="chi2")
            except NotImplementedError as exc:
                print("file-backed add_noise is intentionally unsupported:")
                print(exc)
            finally:
                remove_if_temporary(output)
            """
        ),
        code(
            """
            signal_records = []
            FIXED_SIGNAL_LEVEL = 25
            TARGET_SIGNAL_SNR = 25

            def signal_base_kwargs(frame, *, auto_bounding):
                return {
                    "path": stg.constant_path(
                        f_start=frame.get_frequency(frame.fchans // 2),
                        drift_rate=0.25 * frame.unit_drift_rate,
                    ),
                    "f_profile": stg.gaussian_f_profile(width=8 * frame.df),
                    "bp_profile": stg.constant_bp_profile(level=1),
                    "auto_bounding": auto_bounding,
                    "truncate_below": 1e-3,
                }

            def signal_noise_kwargs(signal_kwargs):
                return {
                    "path": signal_kwargs["path"],
                    "f_profile": signal_kwargs["f_profile"],
                    "auto_bounding": True,
                    "truncate_below": signal_kwargs["truncate_below"],
                    "config": noise_config,
                }

            def injection_kwargs(frame, *, auto_bounding, level_mode):
                kwargs = signal_base_kwargs(frame, auto_bounding=auto_bounding)
                stats = None
                target_snr = np.nan

                if level_mode == "fixed-level":
                    level = FIXED_SIGNAL_LEVEL
                elif level_mode == "target-snr":
                    stats = frame.estimate_noise_stats(**signal_noise_kwargs(kwargs))
                    target_snr = TARGET_SIGNAL_SNR
                    level = frame.get_intensity(snr=TARGET_SIGNAL_SNR, noise_stats=stats)
                else:
                    raise ValueError(f"Unknown level mode: {level_mode}")

                context_fchans = 0
                excluded_fchans = 0
                if stats is not None:
                    context_fchans = stats.context_bounds[1] - stats.context_bounds[0]
                    excluded_fchans = stats.excluded_bounds[1] - stats.excluded_bounds[0]

                kwargs["t_profile"] = stg.constant_t_profile(level=level)
                metadata = {
                    "level_mode": level_mode,
                    "uses_local_noise_stats": stats is not None,
                    "target_snr": target_snr,
                    "resolved_level": level,
                    "noise_mean": np.nan if stats is None else stats.mean,
                    "noise_std": np.nan if stats is None else stats.std,
                    "noise_samples": 0 if stats is None else stats.n_samples,
                    "noise_auto_bounding": False if stats is None else True,
                    "noise_context_fchans": context_fchans,
                    "noise_excluded_fchans": excluded_fchans,
                    "noise_context_bounds": None if stats is None else stats.context_bounds,
                    "noise_excluded_bounds": None if stats is None else stats.excluded_bounds,
                }
                return kwargs, metadata

            def affected_fchans_from_signal(signal):
                nonzero = np.flatnonzero(np.any(signal != 0, axis=0))
                return int(nonzero.size)

            def eager_injection(auto_bounding, level_mode):
                output = OUT / f"{SOURCE_STEM}_eager_{level_mode}_injected_auto_{auto_bounding}.h5"
                frame = stg.Frame(waterfall=source_path)
                kwargs, metadata = injection_kwargs(
                    frame,
                    auto_bounding=auto_bounding,
                    level_mode=level_mode,
                )
                signal = frame.add_signal(**kwargs)
                affected = affected_fchans_from_signal(signal)
                signal_shape = signal.shape
                frame.save_hdf5(output)
                output_size_mib = output.stat().st_size / 1024**2
                del frame, signal
                remove_if_temporary(output)
                return {
                    "output_size_mib": output_size_mib,
                    "affected_fchans": affected,
                    "returned_signal_shape": signal_shape,
                    "metadata": metadata,
                }

            def file_backed_injection(auto_bounding, level_mode):
                output = OUT / f"{SOURCE_STEM}_file_backed_{level_mode}_injected_auto_{auto_bounding}.h5"
                with stg.Frame.open_copy(
                    source_path,
                    output,
                    overwrite=True,
                    max_chunk_bytes=PROFILE_CHUNK_BYTES,
                ) as backed:
                    kwargs, metadata = injection_kwargs(
                        backed,
                        auto_bounding=auto_bounding,
                        level_mode=level_mode,
                    )
                    result = backed.add_signal(
                        **kwargs,
                        max_chunk_bytes=PROFILE_CHUNK_BYTES,
                    )
                output_size_mib = output.stat().st_size / 1024**2
                remove_if_temporary(output)
                return {
                    "output_size_mib": output_size_mib,
                    "result": result,
                    "metadata": metadata,
                }

            auto_bounding_values = [True]
            if RUN_UNBOUNDED_SIGNAL:
                auto_bounding_values.insert(0, False)

            for level_mode in ("fixed-level", "target-snr"):
                label_prefix = "" if level_mode == "fixed-level" else "target-SNR "
                for auto_bounding in auto_bounding_values:
                    eager_result = profile_case(
                        f"eager {label_prefix}injection auto_bounding={auto_bounding}",
                        lambda auto_bounding=auto_bounding, level_mode=level_mode: eager_injection(
                            auto_bounding,
                            level_mode,
                        ),
                        signal_records,
                        operation="signal injection",
                        storage="eager",
                        auto_bounding=auto_bounding,
                        level_mode=level_mode,
                        source_size_mib=source_size_mib,
                    )
                    signal_records[-1]["affected_fchans"] = eager_result["affected_fchans"]
                    signal_records[-1]["returned_signal_shape"] = eager_result["returned_signal_shape"]
                    signal_records[-1]["output_size_mib"] = eager_result["output_size_mib"]
                    signal_records[-1].update(eager_result["metadata"])

                    backed_result = profile_case(
                        f"file-backed {label_prefix}injection auto_bounding={auto_bounding}",
                        lambda auto_bounding=auto_bounding, level_mode=level_mode: file_backed_injection(
                            auto_bounding,
                            level_mode,
                        ),
                        signal_records,
                        operation="signal injection",
                        storage="file-backed",
                        auto_bounding=auto_bounding,
                        level_mode=level_mode,
                        source_size_mib=source_size_mib,
                    )
                    fb_result = backed_result["result"]
                    signal_records[-1]["affected_fchans"] = fb_result.frequency_slice.stop - fb_result.frequency_slice.start
                    signal_records[-1]["time_chunks"] = fb_result.time_chunks
                    signal_records[-1]["max_chunk_shape"] = fb_result.max_chunk_shape
                    signal_records[-1]["output_size_mib"] = backed_result["output_size_mib"]
                    signal_records[-1].update(backed_result["metadata"])

            signal_table = show_table(signal_records)
            """
        ),
        code(
            """
            all_tables = {
                "read": read_table,
                "write": write_table,
                "noise": noise_table,
                "signal": signal_table,
            }

            for name, table in all_tables.items():
                path = OUT / f"{SOURCE_STEM}_{name}_table.csv"
                table.to_csv(path, index=False)
                print("wrote", path)

            def one(table, case):
                match = table.loc[table["case"] == case]
                if len(match) != 1:
                    raise ValueError(f"Expected one row for {case!r}, found {len(match)}")
                return match.iloc[0]

            def speedup(reference, candidate):
                return float(reference["elapsed_s"] / candidate["elapsed_s"])

            def ratio(reference, candidate, column):
                ref = float(reference[column])
                cand = float(candidate[column])
                if cand == 0:
                    return np.inf
                return ref / cand

            read_eager = one(read_table, "eager full load: Frame(waterfall=...)")
            read_bounded = one(read_table, "file-backed bounded read_frame")
            read_full_backed = one(read_table, "file-backed full read_frame")
            write_eager = one(write_table, "eager load + save_hdf5")
            write_copy = one(write_table, "file-backed open_copy")
            noise_eager = one(noise_table, "eager full load + local noise stats")
            noise_backed = one(noise_table, "file-backed local noise stats")
            sig_eager_bounded = one(signal_table, "eager injection auto_bounding=True")
            sig_backed_bounded = one(signal_table, "file-backed injection auto_bounding=True")
            sig_eager_target_bounded = one(signal_table, "eager target-SNR injection auto_bounding=True")
            sig_backed_target_bounded = one(signal_table, "file-backed target-SNR injection auto_bounding=True")

            bounded_read_fraction = read_bounded["fchans_read"] / PROFILE_FCHANS
            bounded_signal_fraction = sig_backed_bounded["affected_fchans"] / PROFILE_FCHANS
            bounded_target_signal_fraction = sig_backed_target_bounded["affected_fchans"] / PROFILE_FCHANS

            def bounds_fraction(bounds):
                if bounds is None:
                    return np.nan
                return (bounds[1] - bounds[0]) / PROFILE_FCHANS

            analysis_rows = [
                {
                    "comparison": "bounded file-backed read vs eager full load",
                    "wall_time_speedup": speedup(read_eager, read_bounded),
                    "python_peak_ratio": ratio(read_eager, read_bounded, "python_peak_mib"),
                    "rss_delta_ratio": ratio(read_eager, read_bounded, "rss_delta_mib"),
                    "channels_touched_fraction": bounded_read_fraction,
                },
                {
                    "comparison": "full file-backed read vs eager full load",
                    "wall_time_speedup": speedup(read_eager, read_full_backed),
                    "python_peak_ratio": ratio(read_eager, read_full_backed, "python_peak_mib"),
                    "rss_delta_ratio": ratio(read_eager, read_full_backed, "rss_delta_mib"),
                    "channels_touched_fraction": 1.0,
                },
                {
                    "comparison": "file-backed open_copy vs eager load + save",
                    "wall_time_speedup": speedup(write_eager, write_copy),
                    "python_peak_ratio": ratio(write_eager, write_copy, "python_peak_mib"),
                    "rss_delta_ratio": ratio(write_eager, write_copy, "rss_delta_mib"),
                    "channels_touched_fraction": 1.0,
                },
                {
                    "comparison": "file-backed local noise stats vs eager local noise stats",
                    "wall_time_speedup": speedup(noise_eager, noise_backed),
                    "python_peak_ratio": ratio(noise_eager, noise_backed, "python_peak_mib"),
                    "rss_delta_ratio": ratio(noise_eager, noise_backed, "rss_delta_mib"),
                    "channels_touched_fraction": (PROFILE_CONTEXT_WIDTH * 2 + 32) / PROFILE_FCHANS,
                },
                {
                    "comparison": "file-backed bounded injection vs eager bounded injection",
                    "wall_time_speedup": speedup(sig_eager_bounded, sig_backed_bounded),
                    "python_peak_ratio": ratio(sig_eager_bounded, sig_backed_bounded, "python_peak_mib"),
                    "rss_delta_ratio": ratio(sig_eager_bounded, sig_backed_bounded, "rss_delta_mib"),
                    "channels_touched_fraction": bounded_signal_fraction,
                },
                {
                    "comparison": "file-backed bounded target-SNR injection vs eager bounded target-SNR injection",
                    "wall_time_speedup": speedup(sig_eager_target_bounded, sig_backed_target_bounded),
                    "python_peak_ratio": ratio(sig_eager_target_bounded, sig_backed_target_bounded, "python_peak_mib"),
                    "rss_delta_ratio": ratio(sig_eager_target_bounded, sig_backed_target_bounded, "rss_delta_mib"),
                    "channels_touched_fraction": bounded_target_signal_fraction,
                    "noise_context_fraction": bounds_fraction(sig_backed_target_bounded["noise_context_bounds"]),
                },
            ]

            if RUN_UNBOUNDED_SIGNAL:
                sig_eager_unbounded = one(signal_table, "eager injection auto_bounding=False")
                sig_backed_unbounded = one(signal_table, "file-backed injection auto_bounding=False")
                sig_eager_target_unbounded = one(signal_table, "eager target-SNR injection auto_bounding=False")
                sig_backed_target_unbounded = one(signal_table, "file-backed target-SNR injection auto_bounding=False")
                analysis_rows.extend([
                    {
                        "comparison": "file-backed unbounded injection vs eager unbounded injection",
                        "wall_time_speedup": speedup(sig_eager_unbounded, sig_backed_unbounded),
                        "python_peak_ratio": ratio(sig_eager_unbounded, sig_backed_unbounded, "python_peak_mib"),
                        "rss_delta_ratio": ratio(sig_eager_unbounded, sig_backed_unbounded, "rss_delta_mib"),
                        "channels_touched_fraction": 1.0,
                    },
                    {
                        "comparison": "file-backed bounded injection vs file-backed unbounded injection",
                        "wall_time_speedup": speedup(sig_backed_unbounded, sig_backed_bounded),
                        "python_peak_ratio": ratio(sig_backed_unbounded, sig_backed_bounded, "python_peak_mib"),
                        "rss_delta_ratio": ratio(sig_backed_unbounded, sig_backed_bounded, "rss_delta_mib"),
                        "channels_touched_fraction": bounded_signal_fraction,
                    },
                    {
                        "comparison": "file-backed unbounded target-SNR injection vs eager unbounded target-SNR injection",
                        "wall_time_speedup": speedup(sig_eager_target_unbounded, sig_backed_target_unbounded),
                        "python_peak_ratio": ratio(sig_eager_target_unbounded, sig_backed_target_unbounded, "python_peak_mib"),
                        "rss_delta_ratio": ratio(sig_eager_target_unbounded, sig_backed_target_unbounded, "rss_delta_mib"),
                        "channels_touched_fraction": 1.0,
                        "noise_context_fraction": bounds_fraction(sig_backed_target_unbounded["noise_context_bounds"]),
                    },
                    {
                        "comparison": "file-backed bounded target-SNR injection vs file-backed unbounded target-SNR injection",
                        "wall_time_speedup": speedup(sig_backed_target_unbounded, sig_backed_target_bounded),
                        "python_peak_ratio": ratio(sig_backed_target_unbounded, sig_backed_target_bounded, "python_peak_mib"),
                        "rss_delta_ratio": ratio(sig_backed_target_unbounded, sig_backed_target_bounded, "rss_delta_mib"),
                        "channels_touched_fraction": bounded_target_signal_fraction,
                        "noise_context_fraction": bounds_fraction(sig_backed_target_bounded["noise_context_bounds"]),
                    },
                ])

            analysis_table = pd.DataFrame(analysis_rows)
            display(analysis_table.round({
                "wall_time_speedup": 2,
                "python_peak_ratio": 2,
                "rss_delta_ratio": 2,
                "channels_touched_fraction": 6,
                "noise_context_fraction": 6,
            }))

            analysis_path = OUT / f"{SOURCE_STEM}_analysis_table.csv"
            analysis_table.to_csv(analysis_path, index=False)
            print("wrote", analysis_path)
            """
        ),
        code(
            """
            from IPython.display import Markdown

            def fmt_speed(value):
                return f"{value:.1f}x"

            def fmt_pct(value):
                return f"{100 * value:.4f}%"

            read_speed = speedup(read_eager, read_bounded)
            noise_speed = speedup(noise_eager, noise_backed)
            copy_speed = speedup(write_eager, write_copy)
            bounded_signal_speed = speedup(sig_eager_bounded, sig_backed_bounded)
            bounded_target_signal_speed = speedup(sig_eager_target_bounded, sig_backed_target_bounded)
            target_context_fraction = bounds_fraction(sig_backed_target_bounded["noise_context_bounds"])

            unbounded_sentence = ""
            if RUN_UNBOUNDED_SIGNAL:
                file_backed_bound_speed = speedup(sig_backed_unbounded, sig_backed_bounded)
                file_backed_target_bound_speed = speedup(
                    sig_backed_target_unbounded,
                    sig_backed_target_bounded,
                )
                unbounded_sentence = (
                    f" Bounded file-backed injection was {fmt_speed(file_backed_bound_speed)} "
                    "faster than file-backed unbounded injection. Bounded target-SNR "
                    f"injection was {fmt_speed(file_backed_target_bound_speed)} faster "
                    "than file-backed unbounded target-SNR injection."
                )

            interpretation = f'''
            ### One-GiB Interpretation

            - The source target is {materialized_gib():.2f} GiB as a materialized
              float64 frame. The actual HDF5 file written here is
              {source_size_mib / 1024:.2f} GiB.

            - Bounded file-backed reads touched {int(read_bounded["fchans_read"]):,}
              of {PROFILE_FCHANS:,} channels ({fmt_pct(bounded_read_fraction)}) and
              were {fmt_speed(read_speed)} faster than eager full loading.

            - Full file-backed reads are expected to be close to eager full loading
              because both paths must materialize the full observation.

            - File-backed `open_copy(...)` was {fmt_speed(copy_speed)} faster than
              eager load plus `save_hdf5(...)` and did not materialize the full frame
              in Python.

            - File-backed local noise/SNR context estimation was
              {fmt_speed(noise_speed)} faster than the eager path while producing the
              same statistics. This is the scientifically relevant mode for narrowband
              injections because SNR should be estimated locally around the signal
              track.

            - Target-SNR injection rows estimate local sigma-clipped noise statistics
              before mutation, then call `get_intensity(snr=..., noise_stats=...)`.
              The bounded file-backed target-SNR row used
              {int(sig_backed_target_bounded["noise_samples"]):,} noise samples from
              {fmt_pct(target_context_fraction)} of the frequency axis and resolved
              SNR {sig_backed_target_bounded["target_snr"]:.1f} to an intensity level
              of {sig_backed_target_bounded["resolved_level"]:.4g}. These timings
              intentionally include the local noise-estimation cost.

            - Bounded file-backed signal injection touched
              {int(sig_backed_bounded["affected_fchans"]):,} of {PROFILE_FCHANS:,}
              channels ({fmt_pct(bounded_signal_fraction)}) and was
              {fmt_speed(bounded_signal_speed)} faster than eager bounded injection.
              {unbounded_sentence}

            - Bounded file-backed target-SNR injection touched
              {int(sig_backed_target_bounded["affected_fchans"]):,} of
              {PROFILE_FCHANS:,} channels ({fmt_pct(bounded_target_signal_fraction)})
              and was {fmt_speed(bounded_target_signal_speed)} faster than eager
              bounded target-SNR injection.

            - Timings are still local wall-clock measurements. For reporting, prefer
              pairing speedups with the source file size, materialized frame size,
              channel fraction touched, and whether the operation materialized the
              whole frame.
            '''

            display(Markdown(interpretation))
            """
        ),
    ]),
}


def main() -> None:
    for name, nb in NOTEBOOKS.items():
        path = ROOT / name
        path.write_text(json.dumps(nb, indent=1) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
