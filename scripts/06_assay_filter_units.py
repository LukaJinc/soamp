"""
Step 5: assay filtering and unit standardization.

Operates on the union of (a) QMAP-baseline peptides (diff bucket in_both) and
(b) recovered peptides (data/recovered_peptides.csv), reprocessing each peptide's
*raw* DBAASP targetActivities directly -- not QMAP's pre-aggregated per-species
numbers -- so that the uM conversion is uniformly computed from SMILES-derived
molecular weight (RDKit) for every peptide, canonical and non-canonical alike,
per the task's explicit requirement.

Filters applied (each logged with a before/after count):
  1. activityMeasureGroup == 'MIC' exactly (excludes MIC50/MIC90/IC50/MBC/LC/... )
  2. target species domain == Bacteria (via NCBI taxonomy genus classification)
  3. unit in {uM, ug/ml} (excludes missing units and the rare 'nmol/g' entries,
     which aren't convertible to molar concentration without sample-specific
     normalization data DBAASP doesn't provide)
  4. molecular weight computable from the peptide's resolved SMILES (RDKit)

Then: concentration strings are parsed with midpoint-for-ranges /
explicit-bound-for-censored (units.parse_activity, ported from QMAP), ug/mL
values are converted to uM via the peptide's RDKit molecular weight, and
multiple measurements for the same (peptide, species) pair are IQR-outlier-
filtered and averaged (matching QMAP's method, units/get_iqr port).
"""
import sys
import os
import json
import csv
import math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from parse_dbaasp import DBAASPPeptide
from smiles_gen import generate_smiles
from units import parse_activity, classify_censoring, compute_smiles_weight, ug_ml_to_uM, precision
from taxonomy import classify_species
import numpy as np

RAW_JSONL = os.path.join(os.path.dirname(__file__), "..", ".cache", "dbaasp_raw.jsonl")
DIFF_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "dbaasp_vs_qmap_diff.csv")
RECOVERED_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "recovered_peptides.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "step5_standardized_mic.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "step5_assay_unit_log.txt")


def get_iqr(values):
    q1, q3 = np.percentile(values, [25, 75])
    return q1, q3, q3 - q1


def resolve_smiles_for_peptide(p: DBAASPPeptide):
    """Native SMILES if present, else attempt p2smi generation. Returns (smiles, source, reason)."""
    if p.has_native_smiles:
        return p.native_smiles_list[0], "native_dbaasp_smiles", ""
    result = generate_smiles(p.sequence, p.unusual_amino_acids, p.intrachain_bonds,
                               p.nterminus, p.cterminus)
    if result["convertible"]:
        return result["smiles"], f"p2smi_generated({result['constraint_used']})", ""
    return None, None, result["reason"]


def main():
    raw_peptides = {}
    with open(RAW_JSONL) as f:
        for line in f:
            raw = json.loads(line)
            raw_peptides[raw["id"]] = DBAASPPeptide(raw)

    in_both_ids = set()
    with open(DIFF_CSV) as f:
        for row in csv.DictReader(f):
            if row["bucket"] == "in_both":
                in_both_ids.add(int(row["peptide_id"]))

    recovered = {}
    with open(RECOVERED_CSV) as f:
        for row in csv.DictReader(f):
            recovered[int(row["peptide_id"])] = row

    # --- Resolve SMILES + MW once per included peptide ---
    n_qmap_baseline = len(in_both_ids)
    n_recovered = len(recovered)
    included_ids = in_both_ids | set(recovered.keys())

    peptide_info = {}
    n_smiles_unresolved = 0
    n_mw_unresolved = 0
    for pid in included_ids:
        p = raw_peptides.get(pid)
        if p is None:
            continue
        if pid in recovered:
            smiles = recovered[pid]["smiles"]
            source = "recovered"
        else:
            smiles, src, reason = resolve_smiles_for_peptide(p)
            source = "qmap_original"
            if smiles is None:
                n_smiles_unresolved += 1
                peptide_info[pid] = {"peptide": p, "smiles": None, "mw": None,
                                       "source": source, "unresolved_reason": reason}
                continue
        mw = compute_smiles_weight(smiles)
        if mw is None:
            n_mw_unresolved += 1
        peptide_info[pid] = {"peptide": p, "smiles": smiles, "mw": mw, "source": source,
                               "unresolved_reason": "" if mw is not None else "RDKit could not parse resolved SMILES"}

    # --- Walk raw targetActivities, filter + convert ---
    n_activities_total = 0
    assay_group_excluded = Counter()
    n_non_bacteria = Counter()
    n_unit_excluded = Counter()
    n_mw_missing_at_conversion = 0
    n_kept_raw_measurements = 0
    censor_counter = Counter()

    # (peptide_id, species_binomial) -> list of dicts {min, max, censor}
    grouped = defaultdict(list)
    species_domain_cache = {}

    for pid, info in peptide_info.items():
        p = info["peptide"]
        for t in p.target_activities:
            n_activities_total += 1
            group = (t.get("activityMeasureGroup") or {}).get("name")
            if group != "MIC":
                assay_group_excluded[group] += 1
                continue

            species_name = (t.get("targetSpecies") or {}).get("name")
            if species_name not in species_domain_cache:
                species_domain_cache[species_name] = classify_species(species_name)
            domain, taxid, binomial = species_domain_cache[species_name]
            if domain != "Bacteria":
                n_non_bacteria[domain] += 1
                continue

            unit = (t.get("unit") or {}).get("name")
            if unit not in ("µM", "µg/ml"):
                n_unit_excluded[unit] += 1
                continue

            if info["smiles"] is None:
                n_mw_missing_at_conversion += 1
                continue

            min_v, max_v = parse_activity(t.get("concentration"))
            if isinstance(min_v, float) and math.isnan(min_v):
                censor_counter["missing"] += 1
                continue

            if unit == "µg/ml":
                if info["mw"] is None:
                    n_mw_missing_at_conversion += 1
                    continue
                min_v, max_v = ug_ml_to_uM(min_v, max_v, info["mw"])

            censor = classify_censoring(min_v, max_v, t.get("concentration"))
            censor_counter[censor] += 1
            n_kept_raw_measurements += 1
            grouped[(pid, binomial)].append({
                "min": min_v, "max": max_v, "censor": censor, "taxid": taxid,
            })

    # --- IQR outlier removal + averaging per (peptide, species) ---
    final_rows = []
    n_groups_multi = 0
    n_groups_outliers_removed = 0
    n_total_outlier_points = 0

    for (pid, binomial), measurements in grouped.items():
        info = peptide_info[pid]
        p = info["peptide"]

        min_acts = [m["min"] for m in measurements if not (isinstance(m["min"], float) and math.isnan(m["min"])) and m["min"] != 0.]
        max_acts = [m["max"] for m in measurements if not (isinstance(m["max"], float) and math.isnan(m["max"])) and m["max"] != float("inf")]
        all_vals = min_acts + max_acts
        if not all_vals:
            continue

        if len(measurements) > 1:
            n_groups_multi += 1

        if len(all_vals) >= 4:
            q1, q3, iqr = get_iqr(all_vals)
            valid = [x for x in all_vals if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]
            if len(valid) < len(all_vals):
                n_groups_outliers_removed += 1
                n_total_outlier_points += (len(all_vals) - len(valid))
        else:
            valid = all_vals

        mean_val = precision(sum(valid) / len(valid))
        taxid = measurements[0]["taxid"]

        if len(measurements) == 1:
            # Collapse the 4-way parse-level classification (exact/censored/ranged/missing)
            # to the 3 categories requested for the final schema: a single two-sided range
            # is treated as 'censored' (bounded-but-not-a-point-value), same as a one-sided
            # '<X'/'>Y' value; only a single exact reported number is 'exact'.
            mic_type = "exact" if measurements[0]["censor"] == "exact" else "censored"
        else:
            mic_type = "averaged"

        bond_types = "|".join(p.bond_types) if p.bond_types else "none"

        final_rows.append({
            "peptide_id": pid,
            "sequence": p.sequence,
            "smiles": info["smiles"],
            "organism": binomial,
            "ncbi_taxon_id": taxid if taxid else "",
            "mic_value_uM": mean_val,
            "mic_type": mic_type,
            "n_raw_measurements": len(measurements),
            "source": info["source"],
            "has_noncanonical": p.has_noncanonical,
            "bond_type": bond_types,
            "molecular_weight": precision(info["mw"], 5) if info["mw"] else "",
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    log_lines = []
    log_lines.append("=== Step 5: assay filtering + unit standardization ===")
    log_lines.append(f"Peptides considered: {len(included_ids)} "
                      f"(QMAP baseline: {n_qmap_baseline}, recovered: {n_recovered})")
    log_lines.append(f"Peptides where SMILES could not be resolved at all (excluded entirely): {n_smiles_unresolved}")
    log_lines.append(f"Peptides where SMILES resolved but RDKit MW computation failed: {n_mw_unresolved}")
    log_lines.append("")
    log_lines.append(f"Total raw targetActivity records examined across included peptides: {n_activities_total}")
    log_lines.append(f"Excluded: non-MIC assay type (activityMeasureGroup != 'MIC'): "
                      f"{sum(assay_group_excluded.values())}")
    log_lines.append("  top excluded assay types:")
    for g, cnt in assay_group_excluded.most_common(15):
        log_lines.append(f"    {g}: {cnt}")
    log_lines.append(f"Excluded: non-bacterial target domain: {sum(n_non_bacteria.values())}")
    for d, cnt in n_non_bacteria.most_common():
        log_lines.append(f"  {d}: {cnt}")
    log_lines.append(f"Excluded: unsupported/missing unit: {sum(n_unit_excluded.values())}")
    for u, cnt in n_unit_excluded.most_common():
        log_lines.append(f"  {u!r}: {cnt}")
    log_lines.append(f"Excluded: MW unavailable at ug/mL->uM conversion time: {n_mw_missing_at_conversion}")
    log_lines.append(f"Excluded: unparseable/missing concentration string: {censor_counter.get('missing', 0)}")
    log_lines.append("")
    log_lines.append(f"Kept raw (peptide, species, measurement) rows after all filters: {n_kept_raw_measurements}")
    log_lines.append("Censoring classification of kept raw measurements (before grouping):")
    for c, cnt in censor_counter.most_common():
        log_lines.append(f"  {c}: {cnt}")
    log_lines.append("")
    log_lines.append(f"Final (peptide, organism) groups: {len(final_rows)}")
    log_lines.append(f"Groups with >1 raw measurement (averaged): {n_groups_multi}")
    log_lines.append(f"Groups where IQR removed >=1 outlier point: {n_groups_outliers_removed}")
    log_lines.append(f"Total individual outlier points removed: {n_total_outlier_points}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
