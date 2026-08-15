# QA report: SMILES-based AMP MIC regression dataset

Generated: 2026-08-15

Independent audit of `scripts/01-08_*.py`'s outputs (`data/*.csv`, `reports/curation_audit.md`). Deliberately avoids importing the pipeline's own filter/unit-conversion logic (`scripts/common/units.py`, `scripts/06_assay_filter_units.py`) so a bug there would actually be caught -- see `scripts/qa/qa_common.py` for the two disclosed exceptions.

## Summary

| # | Section | Status |
|---|---|---|
| 1 | Coverage vs QMAP baseline | **PASS** |
| 2 | SMILES structural integrity | **PASS** |
| 3 | Independent unit-conversion spot-check | **PASS** |
| 4 | Independent MIC/Bacteria filter verification | **PASS** |
| 5 | Train/test split integrity | **PASS** |
| 6 | Row-level sanity & documented-caveat checks | **PASS** |
| 7 | Threshold coverage & labeling correctness | **FLAGGED** |

**Overall: FLAGGED**

## Section 1: Coverage vs QMAP baseline -- PASS

**(a) Full-corpus comparison:** final dataset has **15,904** unique peptides vs QMAP's full published corpus of **18,033** (freshly reparsed from `.cache/qmap_hf/dbaasp.json`, not trusted from `qmap_included.csv`). Delta: -2,129 (-11.8%). Note this is not a clean apples-to-apples number since QMAP's 18,033 includes peptides with zero bacterial-MIC rows -- see (b) for the fair comparison.

**(b) Apples-to-apples comparison:** of QMAP's 18,033 peptides, **13,761** have at least one non-empty `targets` dict (freshly recomputed -- this is QMAP's own "usable" peptide count). This pipeline's final dataset retains **13,680** QMAP-sourced peptides (delta: -81). A negative delta here is expected: this pipeline reprocesses every peptide's MIC values from raw DBAASP data with its own SMILES-derived-MW unit conversion (Step 5) rather than trusting QMAP's pre-aggregated numbers, so a peptide can legitimately end up with zero surviving rows here (e.g. all its measurements needed unit conversion but its resolved SMILES's MW couldn't be computed) even though QMAP counted it as usable.

**(c) Recovery contribution:** 2,870 candidate peptides were recovered from QMAP's exclusion buckets (Step 4); **2,224** of those survive into the final dataset with >=1 bacterial MIC row (gap of 646, matching 646 peptide_ids confirmed absent from step5_standardized_mic.csv).

**Metrics:**

| metric | value |
|---|---|
| final_unique_peptides | 15904 |
| qmap_full_corpus | 18033 |
| delta_a | -2129 |
| pct_a | -11.81 |
| qmap_usable_peptides | 13761 |
| final_qmap_sourced | 13680 |
| delta_b | -81 |
| n_recovered_candidates | 2870 |
| final_recovered_unique | 2224 |
| recovery_gap | 646 |
| recovery_gap_verified | True |

**recovered_via counts: audit-claimed vs actual (recovered_peptides.csv):**

| recovered_via | audit_claimed | actual | match |
|---|---|---|---|
| native_dbaasp_smiles | 2853 | 2853 | True |
| p2smi_generated(linear) | 11 | 11 | True |
| p2smi_generated(SS) | 3 | 3 | True |
| p2smi_generated(HT) | 3 | 3 | True |

**recovery_category counts: audit-claimed vs actual:**

| recovery_category | audit_claimed | actual | match |
|---|---|---|---|
| terminus_based | 1230 | 1230 | True |
| residue_based | 17 | 17 | True |
| bond_based | 6 | 6 | True |
| unexplained_temporal_drift | 1617 | 1617 | True |

**Sample of recovered peptide_ids absent from step5 (legitimately dropped):**

| peptide_id |
|---|
| 21607 |
| 11719 |
| 8325 |
| 23463 |
| 19211 |
| 21709 |
| 24631 |
| 11294 |
| 4237 |
| 15772 |

## Section 2: SMILES structural integrity -- PASS

Checked **15,904** unique (peptide_id, smiles) pairs from `final_mic_regression_dataset.csv`. **15,904** parsed successfully in RDKit (0 did not). Of **15,904** with a stored `molecular_weight` to check against, **0** disagreed with an independently recomputed RDKit MW by more than 0.1%. **0** were parseable but not canonical-round-trip-stable.

**Metrics:**

| metric | value |
|---|---|
| n_unique_smiles_checked | 15904 |
| n_parseable | 15904 |
| n_unparseable | 0 |
| n_mw_checked | 15904 |
| n_mw_mismatches | 0 |
| n_roundtrip_unstable | 0 |

## Section 3: Independent unit-conversion spot-check -- PASS

Of **38,808** step5 rows that are a single exact (non-averaged, non-censored) measurement, **38,570** were uniquely traced to one qualifying raw DBAASP `targetActivity` record with a bare-numeric concentration string (skipping anything requiring `parse_activity`'s range/censoring logic, which is out of scope for this independent check). **238** could not be uniquely traced (0 or >1 qualifying raw records, or unresolvable MW) and were skipped. Of the traced rows, **0** disagreed with the pipeline's stored `mic_value_uM` by more than 1.0% when independently recomputed via `rdkit_mw` + hand-rolled `ug_ml_to_uM_by_hand` (bypassing `units.py` entirely).

**Metrics:**

| metric | value |
|---|---|
| n_candidates | 38808 |
| n_traced | 38570 |
| n_ambiguous_skipped | 238 |
| n_mismatches | 0 |

## Section 4: Independent MIC/Bacteria filter verification -- PASS

Checked all **72,587** unique (peptide_id, organism) pairs in `final_mic_regression_dataset.csv` against a fresh, independent re-scan of each peptide's raw `targetActivities` (assay_group=='MIC' and domain=='Bacteria', domain from `classify_species` -- the same classifier the pipeline uses, checking the filter's *application*, not taxonomy correctness). **0** pairs could not be traced back to any qualifying raw record.

**Metrics:**

| metric | value |
|---|---|
| n_pairs_checked | 72587 |
| n_untraceable | 0 |

## Section 5: Train/test split integrity -- PASS

Train set: **11,114** peptide_ids (audit claims 11,114). Test set: **4,790** peptide_ids (audit claims 4,790). Train/test overlap: **0** peptide_ids. Final-dataset peptides missing a split assignment: **0**. Split peptide_ids absent from the final dataset: **0**. Peptides with row-level split inconsistency: **0**.

**Metrics:**

| metric | value |
|---|---|
| n_train | 11114 |
| n_test | 4790 |
| n_overlap | 0 |
| n_orphan_final_not_split | 0 |
| n_orphan_split_not_final | 0 |
| n_row_level_inconsistent | 0 |

## Section 6: Row-level sanity & documented-caveat checks -- PASS

**Duplicates:** 0 rows share a duplicate (peptide_id, organism) key.
**Value sanity:** 0 non-positive/missing mic_value_uM rows, 0 infinite-value rows.
**mic_type self-consistency:** 0 step5 rows have a mic_type label inconsistent with their own n_raw_measurements count. Independently recomputed averaged-group count (n_raw_measurements > 1): **15,456** (audit claims 15,456).
**mic_type breakdown vs audit:** matches exactly.
**IQR outlier-point counts** (audit claims 2,236 groups / 4,062 points) are reported as-is from the audit, not independently re-derived here -- doing so would require reimplementing `parse_activity`'s range/censoring string parser, which is out of scope for this independent pass (disclosed in `qa_common.py`).
**Manual-review flag (peptide_id=21052):** status = `as_expected` (qmap_included=' LLLRRRRLL', final='LLLRRRRLL'). Expected: qmap_included.csv retains the undocumented-elsewhere whitespace, final dataset is clean because it's sourced from an independently-crawled, already-clean copy -- not because anything was corrected mid-pipeline.

**Metrics:**

| metric | value |
|---|---|
| n_duplicate_rows | 0 |
| n_nonpositive_or_missing | 0 |
| n_infinite | 0 |
| n_mic_type_inconsistent | 0 |
| n_groups_multi_recomputed | 15456 |
| flag_21052_status | as_expected |

## Section 7: Threshold coverage & labeling correctness -- FLAGGED

**Table structure:** 658 rows (3 species + 0 genus breakpoints filled in), 0 duplicate keys, 0 bad levels, 0 unparseable thresholds, 0 partial fills, 0 reversed breakpoints.
**Threshold coverage:** 33,840/72,587 (46.6%) of dataset rows resolve to a breakpoint (breakdown: {'species': 33840, 'genus': 0, 'none': 38747}).
**Label distribution:** {'uncertain': 9210, 'unlabeled': 38747, 'active': 20736, 'inactive': 3894}.
**Independent re-derivation cross-check:** 0 threshold_match_level mismatches, 0 label mismatches out of 72,587 rows.
**Censor-direction recovery:** 18,323/18,323 mic_type='censored' rows in the final dataset successfully recovered bounds (18,323 rows in mic_censor_direction.csv total).

**Metrics:**

| metric | value |
|---|---|
| n_threshold_rows | 658 |
| n_species_filled | 3 |
| n_genus_filled | 0 |
| n_dup_keys | 0 |
| n_bad_level | 0 |
| n_bad_threshold | 0 |
| n_partial_fill | 0 |
| n_reversed | 0 |
| n_covered | 33840 |
| n_total | 72587 |
| coverage_pct | 46.6 |
| n_threshold_match_level_mismatches | 0 |
| n_label_mismatches | 0 |
| n_censored_recovered | 18323 |
| n_censored_final | 18323 |

**Highest-impact species still missing a threshold (top 20):**

| match_key | n_dataset_rows |
|---|---|
| Bacillus subtilis | 3888 |
| Klebsiella pneumoniae | 3780 |
| Staphylococcus epidermidis | 3133 |
| Acinetobacter baumannii | 3098 |
| Enterococcus faecalis | 2497 |
| Salmonella enterica | 2193 |
| Micrococcus luteus | 1338 |
| Enterococcus faecium | 1117 |
| Salmonella typhimurium | 1013 |
| Bacillus cereus | 814 |
| Listeria monocytogenes | 759 |
| Enterobacter cloacae | 618 |
| Bacillus megaterium | 467 |
| Pseudomonas syringae | 408 |
| Streptococcus pyogenes | 379 |
| Proteus mirabilis | 346 |
| Streptococcus pneumoniae | 340 |
| Streptococcus mutans | 333 |
| Klebsiella aerogenes | 330 |
| Stenotrophomonas maltophilia | 263 |

**Follow-ups:**

- Threshold coverage is 46.6% -- 496 species still have no breakpoint filled in. See punch list for the highest-impact organisms to prioritize.

