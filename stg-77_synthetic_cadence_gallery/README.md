# Synthetic Cadence Gallery

This folder contains a small, reproducible gallery of synthetic cadence
examples for visually inspecting `setigen` cadence behavior.

## Generate Outputs

Run:

```bash
python generate_cadence_gallery.py
```

The script writes PNG files to `outputs/`.

## What It Generates

- `abacad_multi_signal.png`
  Standard `ABACAD` cadence view with multiple injected synthetic signals.
- `abacad_multi_signal_slew.png`
  The same cadence with vertical spacing scaled by slew time.
- `ababab_crosscheck.png`
  A denser `ABABAB` cadence for comparing alternating pointings.

The examples use the public `Frame`, `OrderedCadence`, `Cadence.by_label`,
`add_signal`, and plotting APIs documented in the main `setigen` docs.
