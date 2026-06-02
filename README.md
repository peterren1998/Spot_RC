Code for running spot fitting of DNA MERFISH spots in RC

## Signal-spot drift alignment

Datasets without fiducial beads can use adjacent-round signal spots for drift:

```bash
SpotDNA ... --drift-method signal
```

In signal mode the default fitted/pool channels are `750,647,561`. If the DAX
frame order differs, pass the actual signal channel order with
`--fish-channels`, for example:

```bash
SpotDNA ... --drift-method signal --fish-channels 750,647,561
```

Signal mode still has to know the full acquired DAX channel order, including
channels that are loaded only to deinterleave frames. By default, non-reference
rounds use `--fish-channels` plus `--fiducial-channel`, and the reference round
also appends `--DAPI-channel`. Override this for unusual acquisitions:

```bash
SpotDNA ... --drift-method signal \
    --fish-channels 750,647,561 \
    --dax-channels 750,647,561,488 \
    --ref-dax-channels 750,647,561,488,405
```

The signal drift estimator fits spots first, pools chromatic-corrected spot
coordinates across the selected channels, estimates each round's drift from
the adjacent processed round, and accumulates drift back to `--ref`. Per-bit
HDF5 groups store `drift_method`, `drift_parent_round`, and a `drift_qc` group
with source/reference spot counts, candidate matches, inliers, and residuals.

If a non-reference round has no entries for the requested `--fish-channels` in
`Color_Usage.csv`, signal mode skips spot fitting for that round. The next
round with requested signal channels is aligned to the nearest previously
processed round.
