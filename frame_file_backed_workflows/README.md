# Frame And File-Backed Workflows

These notebooks exercise the frame, signal-injection, file-backed I/O,
cadence, and derived-product behavior under active development in
`/Users/bryanbrzycki/Code/seti/setigen`.

They expect `setigen` to be installed in the active `seti` conda environment,
preferably as an editable install from `/Users/bryanbrzycki/Code/seti/setigen`:

```bash
conda run -n seti python -m pip install -e /Users/bryanbrzycki/Code/seti/setigen
```

Run the first setup cell in each notebook before any plotting cells. It selects
the inline matplotlib backend before importing `matplotlib.pyplot` or `setigen`,
forces the inline backend if the live notebook kernel is still on plain `Agg`,
and prints the active backend. A backend of `Agg` in a live notebook usually
means a startup hook or environment variable has overridden Jupyter's inline
display path.

Plot cells explicitly call `IPython.display.display(fig)` and then close the
figure instead of relying on `plt.show()`, which avoids the non-interactive
`Agg.show()` warning path.

Suggested order:

- `01_small_frame_signal_injection.ipynb`
- `02_file_write_load_loops.ipynb`
- `03_file_backed_large_frequency_ranges.ipynb`
- `04_cadence_signal_workflow.ipynb`
- `05_spectrum_timeseries_products.ipynb`
- `06_file_backed_performance_profile.ipynb`
- `07_one_gib_file_backed_profile.ipynb`

Notebook `06` is the reporting-oriented profile notebook. It writes
`profile_read_table.csv`, `profile_write_table.csv`, `profile_noise_table.csv`,
`profile_signal_table.csv`, and `profile_analysis_table.csv` under
`generated/` for consistent reuse.

Notebook `07` repeats the profiling workflow with a 1 GiB materialized synthetic
frame target by default. It keeps the source HDF5 file and deletes temporary
benchmark outputs unless `KEEP_LARGE_OUTPUTS = True` is set in the notebook.
Its signal-injection table includes both fixed-level rows, which isolate
render/write cost, and target-SNR rows, which first estimate local sigma-clipped
noise statistics around the signal track and then compute the injected intensity
with `Frame.get_intensity(..., noise_stats=...)`.
The smoke validator sets `SETIGEN_FAST_LARGE_PROFILE=1` so validation does not
create a 1 GiB file.

Generated files are written under `generated/` when notebooks are executed.
That directory is intentionally ignored because the artifacts are reproducible
from the notebooks.

The notebooks can be regenerated and smoke-tested from
`setigen_development` with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-seti conda run -n seti python frame_file_backed_workflows/make_notebooks.py
MPLCONFIGDIR=/tmp/matplotlib-seti conda run -n seti python frame_file_backed_workflows/validate_notebooks.py
```
