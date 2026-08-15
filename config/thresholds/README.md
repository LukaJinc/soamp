# Organism-specific MIC activity thresholds

`organism_thresholds.csv` is the hand-curated `thresholds:organism_specific`
artifact: for each organism (or genus, as a fallback), the MIC breakpoints
that decide whether a peptide is "active" against it. This table is
consumed by `src/soamp/common/thresholds.py`, shared by the Stage 1
labeling pipeline and (later) the Stage 4 benchmarking threshold decider --
it is not regenerated as part of the regular curation/labeling run, only
via `pipeline/labeling/build_threshold_template.py` when the underlying
dataset changes.

## Columns

- `match_key` -- the organism string (species row, e.g. `"Staphylococcus
  aureus"`, matched exactly against the final dataset's `organism` column)
  or the genus alone (genus row, e.g. `"Staphylococcus"`).
- `level` -- `species` or `genus`.
- `ncbi_taxon_id_if_available` -- populated for species rows where the
  curation pipeline resolved one; blank for genus rows.
- `n_dataset_rows` -- how many rows in `data/final_mic_regression_dataset.csv`
  this key covers, as of the last regeneration. Use it to prioritize which
  organisms to fill in first.
- `active_threshold_uM` / `inactive_threshold_uM` -- **the dual breakpoint
  to fill in.** Units are micromolar (uM), matching the dataset's canonical
  `mic_value_uM` column. A peptide is labeled `active` if its MIC is at or
  below `active_threshold_uM`, `inactive` if at or above
  `inactive_threshold_uM`, and `uncertain` if strictly between the two --
  an intentional gray zone (mirrors the susceptible/intermediate/resistant
  shape of CLSI/EUCAST-style breakpoint tables). Set both to the same value
  for a single cutoff with no gray zone. Both columns must be filled
  together -- leave both blank for organisms you haven't sourced values
  for yet (blank rows are simply skipped at lookup time, label =
  `unlabeled`, not an error); filling only one is a validation error.
- `source` -- citation for where the threshold values came from (paper,
  CLSI/EUCAST breakpoint table, etc.). Required whenever the thresholds are
  filled in, so provenance survives independent of this session.
- `notes` -- free text (caveats, alternate values considered, etc.).

## Lookup precedence

An exact `species`-level match on the dataset's `organism` string wins;
otherwise a `genus`-level match on the organism's first word is used as a
fallback; otherwise the row is unmatched. See
`src/soamp/common/thresholds.py::lookup_threshold`.

## Updating

Edit `active_threshold_uM`/`inactive_threshold_uM`/`source`/`notes`
directly in the CSV, then add an entry to `CHANGELOG.md` describing what
changed and why. Re-running `pipeline/labeling/build_threshold_template.py`
after the underlying dataset changes will refresh `n_dataset_rows` and add
any newly-seen organisms without touching values you've already filled in.
