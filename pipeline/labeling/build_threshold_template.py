"""Generate/refresh config/thresholds/organism_thresholds.csv.

Not a numbered pipeline step -- this is a maintenance utility for the
hand-curated `thresholds:organism_specific` artifact, run manually whenever
the underlying dataset changes or the table needs to be (re)scaffolded.

Seeds one 'species'-level row per distinct `organism` string in
data/final_mic_regression_dataset.csv, plus one 'genus'-level row per
distinct genus (first word of `organism`), each with `active_threshold_uM`
and `inactive_threshold_uM` left blank for the user to fill in -- no cutoff
values are invented here. The two columns are a dual breakpoint (at/below
active = active, at/above inactive = inactive, the gap between is an
intentional gray zone -> label 'uncertain'); see
src/soamp/common/thresholds.py for the lookup semantics.

Rerunning is safe: any (level, match_key) already present keeps its
existing active_threshold_uM/inactive_threshold_uM/source/notes
(hand-entered by the user) and only has its n_dataset_rows count
refreshed; new keys are appended with blank thresholds; keys no longer
present in the dataset are kept (not deleted -- the user's hand-entered
value shouldn't vanish silently) but flagged as stale in the log and given
n_dataset_rows=0 so they sort to the bottom.

Every run appends an entry to config/thresholds/CHANGELOG.md describing
what changed, per the table's "versioned independently with changelog"
requirement.
"""
import csv
import re
from collections import Counter
from datetime import date, datetime

from soamp.labeling.config import LabelingConfig
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging

CFG = load_config("config/labeling/base.yaml", LabelingConfig)
FINAL_CSV = CFG.paths.data_dir / "final_mic_regression_dataset.csv"
TABLE_CSV = CFG.paths.thresholds_dir / CFG.threshold_table.filename
CHANGELOG_MD = CFG.paths.thresholds_dir / "CHANGELOG.md"
README_MD = CFG.paths.thresholds_dir / "README.md"

FIELDNAMES = ["match_key", "level", "ncbi_taxon_id_if_available", "n_dataset_rows",
              "active_threshold_uM", "inactive_threshold_uM", "source", "notes"]

README_TEMPLATE = """\
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
"""


def _next_version(changelog_text: str) -> str:
    versions = re.findall(r"^## v(\d+)\.(\d+)\.(\d+)", changelog_text, flags=re.MULTILINE)
    if not versions:
        return "0.1.0"
    major, minor, patch = (int(x) for x in versions[-1])  # last entry = most recent
    return f"{major}.{minor}.{patch + 1}"


def main() -> None:
    log = configure_logging("build_threshold_template")

    with open(FINAL_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    species_counts: Counter[str] = Counter()
    species_taxid: dict[str, str] = {}
    genus_counts: Counter[str] = Counter()
    for row in rows:
        organism = row["organism"].strip()
        species_counts[organism] += 1
        if organism not in species_taxid and row.get("ncbi_taxon_id_if_available"):
            species_taxid[organism] = row["ncbi_taxon_id_if_available"]
        genus = organism.split(" ", 1)[0] if organism else ""
        if genus:
            genus_counts[genus] += 1

    existing: dict[tuple[str, str], dict] = {}
    if TABLE_CSV.exists():
        with open(TABLE_CSV, newline="") as f:
            for row in csv.DictReader(f):
                existing[(row["level"], row["match_key"])] = row

    new_species_keys, new_genus_keys, stale_keys = [], [], []
    out_rows: list[dict] = []

    for organism, count in species_counts.items():
        key = ("species", organism)
        prior = existing.pop(key, None)
        if prior is None:
            new_species_keys.append(organism)
        out_rows.append({
            "match_key": organism,
            "level": "species",
            "ncbi_taxon_id_if_available": species_taxid.get(organism, ""),
            "n_dataset_rows": count,
            "active_threshold_uM": prior["active_threshold_uM"] if prior else "",
            "inactive_threshold_uM": prior["inactive_threshold_uM"] if prior else "",
            "source": prior["source"] if prior else "",
            "notes": prior["notes"] if prior else "",
        })

    for genus, count in genus_counts.items():
        key = ("genus", genus)
        prior = existing.pop(key, None)
        if prior is None:
            new_genus_keys.append(genus)
        out_rows.append({
            "match_key": genus,
            "level": "genus",
            "ncbi_taxon_id_if_available": "",
            "n_dataset_rows": count,
            "active_threshold_uM": prior["active_threshold_uM"] if prior else "",
            "inactive_threshold_uM": prior["inactive_threshold_uM"] if prior else "",
            "source": prior["source"] if prior else "",
            "notes": prior["notes"] if prior else "",
        })

    # Anything left in `existing` is a key no longer present in the dataset --
    # keep it (don't discard hand-entered values) but zero its count so it
    # sorts to the bottom, and flag it in the log.
    for (level, match_key), prior in existing.items():
        stale_keys.append((level, match_key))
        out_rows.append({
            "match_key": match_key,
            "level": level,
            "ncbi_taxon_id_if_available": prior.get("ncbi_taxon_id_if_available", ""),
            "n_dataset_rows": 0,
            "active_threshold_uM": prior["active_threshold_uM"],
            "inactive_threshold_uM": prior["inactive_threshold_uM"],
            "source": prior["source"],
            "notes": prior["notes"],
        })

    out_rows.sort(key=lambda r: -int(r["n_dataset_rows"]))

    CFG.paths.thresholds_dir.mkdir(parents=True, exist_ok=True)
    with open(TABLE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    n_filled = sum(1 for r in out_rows if r["active_threshold_uM"] or r["inactive_threshold_uM"])
    log.info(f"Wrote {TABLE_CSV}: {len(species_counts)} species rows, "
              f"{len(genus_counts)} genus rows, {n_filled} thresholds filled in, "
              f"{len(new_species_keys)} new species keys, {len(new_genus_keys)} new genus keys, "
              f"{len(stale_keys)} stale keys retained")
    if stale_keys:
        log.warning(f"Stale keys no longer in dataset (kept, n_dataset_rows=0): {stale_keys}")

    if not README_MD.exists():
        with open(README_MD, "w") as f:
            f.write(README_TEMPLATE)
        log.info(f"Wrote {README_MD}")

    changelog_text = CHANGELOG_MD.read_text() if CHANGELOG_MD.exists() else ""
    version = _next_version(changelog_text)
    entry = (
        f"## v{version} - {date.today().isoformat()}\n\n"
        f"Regenerated by `pipeline/labeling/build_threshold_template.py` "
        f"({datetime.now().isoformat(timespec='seconds')}).\n\n"
        f"- {len(species_counts)} species rows, {len(genus_counts)} genus rows "
        f"({n_filled} thresholds filled in)\n"
        f"- {len(new_species_keys)} new species keys, {len(new_genus_keys)} new genus keys added\n"
        f"- {len(stale_keys)} stale keys retained (no longer in dataset): {stale_keys}\n\n"
    )
    with open(CHANGELOG_MD, "a") as f:
        f.write(entry)
    log.info(f"Appended v{version} entry to {CHANGELOG_MD}")


if __name__ == "__main__":
    main()
