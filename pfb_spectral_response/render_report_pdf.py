from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "pfb_response_report_fallback.pdf"
RESULTS = ROOT / "response_experiment_results.json"


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str], *, footer: str = "") -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", va="top")
    y = 0.88
    for paragraph in paragraphs:
        wrapped = fill(paragraph, width=92)
        fig.text(0.08, y, wrapped, fontsize=10.5, va="top", family="monospace")
        y -= 0.028 * (wrapped.count("\n") + 2.2)
    if footer:
        fig.text(0.08, 0.04, footer, fontsize=8, color="0.35")
    pdf.savefig(fig)
    plt.close(fig)


def add_figure_page(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    image = plt.imread(image_path)
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold", va="top")
    ax = fig.add_axes([0.08, 0.30, 0.84, 0.56])
    ax.imshow(image)
    ax.axis("off")
    fig.text(0.08, 0.24, fill(caption, width=100), fontsize=10.5, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    with open(RESULTS, encoding="utf-8") as fh:
        results = json.load(fh)
    edge = results["edge_relative"]
    bandpass = results["bandpass_summary"]

    with PdfPages(REPORT) as pdf:
        add_text_page(
            pdf,
            "PFB Scalloping, Noise Bandpass, and Narrowband Signal Intensity",
            [
                "Rendering note: this is a Matplotlib fallback PDF for machines "
                "without a working TeX engine. The primary report source is "
                "pfb_response_report.tex, rendered as pfb_response_report.pdf.",
                "Core result: a deterministic narrowband tone and white Gaussian "
                "voltage noise pass through the same PFB, but they do not produce "
                "the same frequency-dependent transfer curve in the final spectrogram.",
                f"At the exact coarse-channel edge, the single-branch tone prediction is "
                f"{edge['single_branch_prediction']:.4f} of center, while the modeled "
                f"noise bandpass is {edge['noise_bandpass_model']:.4f} of center.",
                f"The noisy peak excess above a modeled bandpass is "
                f"{edge['modeled_peak_excess']:.4f} +/- {edge['modeled_peak_sem']:.4f} "
                "of the center-bin excess.",
                f"The clean voltage simulation matches the PFB-window tone prediction "
                f"with maximum absolute error {results['signal_prediction_max_abs_error']:.3g}.",
            ],
            footer="See pfb_response_report.tex for the full LaTeX source.",
        )
        add_text_page(
            pdf,
            "Signal-Processing Explanation",
            [
                "Let h[n] be the PFB prototype coefficients, B the number of branches, "
                "and delta the tone offset from a coarse-channel center in units of one "
                "coarse-channel width. A tone in one analysis branch has power transfer "
                "R_tone(delta) = |G(delta)|^2 / |G(0)|^2, where "
                "G(delta) = sum_n h[n] exp(2 pi i delta n / B).",
                "White noise is broadband and the PFB is critically decimated. The "
                "fine-channelized noise baseline therefore contains aliased terms: "
                "R_noise(delta) is proportional to sum_m |G(delta + m)|^2. Near a "
                "coarse-channel edge, adjacent alias terms contribute to the noise "
                "baseline but not to the target-branch deterministic tone power.",
                "This is why a bandpass model fit to noise scalloping is not the same "
                "as a narrowband signal-intensity correction. Near the center they are "
                "similar; at the edge they differ by about a factor of two in power for "
                "the tested configuration.",
            ],
        )
        add_figure_page(
            pdf,
            "Tone Transfer vs Noise Bandpass",
            ROOT / "figures" / "transfer_functions.png",
            "The single-branch deterministic tone response falls to about 0.25 at "
            "the edge, while the aliased noise bandpass remains near 0.52.",
        )
        add_figure_page(
            pdf,
            "Exact Prediction From The PFB Window",
            ROOT / "figures" / "signal_prediction.png",
            "The PFB-window response predicts the clean voltage tone simulation. "
            "This shows the effect is expected from the configured PFB, not a "
            "numerical artifact in the experiment.",
        )
        add_figure_page(
            pdf,
            "Noisy Excess Above Modeled Bandpass",
            ROOT / "figures" / "modeled_bandpass_excess.png",
            "Noisy simulations with a modeled PFB baseline reproduce the same edge "
            "suppression in detected signal excess.",
        )
        add_text_page(
            pdf,
            "Implications",
            [
                f"The noise-only modeled bandpass fit has correlation "
                f"{bandpass['correlation']:.3f} with the measured noise spectrum in "
                "the default short run. Longer integrations reduce residual noise.",
                "For setigen voltage synthesis, the current pipeline accurately models "
                "the PFB it implements: the response is predicted directly by the PFB "
                "window coefficients.",
                "For real observations, matching reality requires matching the actual "
                "instrument PFB coefficients, tap count, coarse-channel selection, "
                "quantization/requantization, fine FFT, normalization, and any later "
                "bandpass flattening.",
                "For inference, keep two curves separate: a noise-bandpass response "
                "for local background/scalloping, and a narrowband tone-transfer "
                "response for recovering original signal intensity from detected excess.",
            ],
        )

    print(REPORT)


if __name__ == "__main__":
    main()
