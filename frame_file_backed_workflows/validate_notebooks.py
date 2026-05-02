from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    "01_small_frame_signal_injection.ipynb",
    "02_file_write_load_loops.ipynb",
    "03_file_backed_large_frequency_ranges.ipynb",
    "04_cadence_signal_workflow.ipynb",
    "05_spectrum_timeseries_products.ipynb",
    "06_file_backed_performance_profile.ipynb",
    "07_one_gib_file_backed_profile.ipynb",
]


def executable_source(source: str) -> str:
    """Return code-cell source suitable for plain Python execution.

    Args:
        source: Notebook code-cell source.

    Returns:
        Source with IPython-only magic lines removed.
    """
    lines = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        lines.append(line)
    return "".join(lines)


def execute_notebook(path: Path) -> None:
    namespace: dict[str, object] = {
        "__file__": str(path),
        "__name__": "__notebook__",
    }
    notebook = json.loads(path.read_text())
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = executable_source("".join(cell.get("source", [])))
        if not source.strip():
            continue
        print(f"  cell {index}")
        exec(compile(source, str(path), "exec"), namespace)


def main() -> None:
    matplotlib.use("Agg")
    os.environ.setdefault("SETIGEN_FAST_LARGE_PROFILE", "1")
    os.chdir(ROOT)
    for name in NOTEBOOKS:
        path = ROOT / name
        print(f"executing {path.name}")
        execute_notebook(path)


if __name__ == "__main__":
    main()
