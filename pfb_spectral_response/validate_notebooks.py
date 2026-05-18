from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


NOTEBOOK_DIR = Path(__file__).resolve().parent
NOTEBOOKS = [
    "01_ideal_pfb_response.ipynb",
    "02_noise_response_overlay.ipynb",
    "03_tone_response_and_snr.ipynb",
    "04_edge_detected_intensity.ipynb",
]


def main() -> None:
    for name in NOTEBOOKS:
        path = NOTEBOOK_DIR / name
        print(f"Validating {path.name}...")
        with open(path, encoding="utf-8") as fh:
            notebook = json.load(fh)
        namespace: dict[str, object] = {"__name__": "__notebook_validation__"}
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            source = "\n".join(
                line for line in source.splitlines() if not line.lstrip().startswith("%")
            )
            exec(compile(source, str(path), "exec"), namespace)
    print("All PFB spectral response notebook code cells ran successfully.")


if __name__ == "__main__":
    main()
