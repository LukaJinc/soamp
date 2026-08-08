# QA follow-up: split-assignment gap and caveat-drift fixes

Generated: 2026-08-07

Addendum to `reports/qa_report.md` (the original independent QA pass) and `reports/curation_audit.md`. Documents the diagnosis and fix for the two issues that pass found: a hard FAIL in Section 5 (split integrity) and a FLAGGED result in Section 6 (documented-caveat drift).

## Issue 1: 1,414 peptides missing a train/test split assignment

### What was broken

`data/split_indices.json`'s `train_peptide_ids` (11,114) and `test_peptide_ids` (3,376) covered only 14,490 of the final dataset's 15,904 unique peptides. 1,414 peptides (8.9%) had no split assignment at all. Train and test themselves did not overlap, and no split-assigned peptide was missing from the final dataset -- the gap was strictly one-directional: final dataset -> split assignment.

### Root cause

`scripts/07_build_final_and_split.py` calls `qmap.toolkit.train_test_split(sequences, peptide_ids, threshold=0.60, test_size=0.2, random_state=42, verbose=True)` without setting `post_filtering`, which defaults to `True` in the installed library (`qmap/toolkit/split/train_test_split.py:15`). After clustering all 15,904 peptides into train/test via Leiden community detection (a step that covers every peptide with no gaps), `post_filtering=True` runs a leakage-prevention step (`qmap._pwiden_engine.filter_out`) that **removes any train-assigned peptide with a similarity edge (>=60% identity) to a test peptide**. Test itself is never touched by this filter. The removed peptides were dropped from the function's return value entirely -- `07_build_final_and_split.py` only ever captured the already-filtered `train_ids`/`test_ids` and wrote them straight to `split_indices.json` with no reconciliation against the full peptide set.

Confirmed by direct re-run: the library's own verbose output states *"Removed 1414 samples from the training set due to similarity with the test set"* -- an exact match to the 1,414-peptide gap QA found. Missing peptides skewed short (mean length 10.9 vs 18.1 for peptides that kept an assignment) and were somewhat more often duplicate-sequence peptides (19.4% vs 15.3%), consistent with short/near-duplicate sequences being more likely to trip a 60%-identity edge to an unrelated test peptide. The gap was **not** caused by a stale/earlier snapshot -- file timestamps showed the final CSV and split were computed in one continuous run.

This also meant one of the audit report's own caveats had gone from stale to **actively false**: "if `train_test_split` failed... that is called out explicitly in the Step 6 log" was true only for *total* failure (the existing `try`/`except`); this *partial* silent-drop failure mode was never anticipated, and the Step 6 log printed `15,904` unique peptides next to `11,114`/`3,376` train/test with no reconciliation -- the gap was invisible unless you did the arithmetic yourself.

### Fix applied

`scripts/07_build_final_and_split.py`:
- After the `train_test_split(...)` call (parameters unchanged: `threshold=0.60`, `test_size=0.2`, `random_state=42` -- same methodology), computes `dropped_ids = set(peptide_ids) - set(train_ids) - set(test_ids)` and appends them to `test_ids` -- not train, since `filter_out` only ever removes *from* train, so putting these peptides in test cannot reintroduce the leakage the filter exists to prevent.
- `split_indices.json` now includes `"post_filtering": true`, a `"post_filtering_note"` explaining the mechanism, and a new `"leakage_filter_reassigned_to_test_peptide_ids"` field (length 1,414) so the reassigned subset is separately identifiable, not silently merged into `test_peptide_ids` indistinguishably.
- `reports/step6_final_split_log.txt` now prints the pre-reassignment train/test counts, the reassigned count, and an explicit reconciliation check (`train + test == unique peptides?`) plus an overlap check, every run -- this is what makes the "called out explicitly" claim actually true going forward.
- `scripts/08_audit_report.py`'s split-failure caveat bullet was rewritten to describe this behavior and record that the gap was found and fixed on 2026-08-07.

### Result

| | Before fix | After fix |
|---|---|---|
| Train peptides | 11,114 | 11,114 (unchanged) |
| Test peptides | 3,376 | **4,790** (+1,414) |
| Train union Test | 14,490 | **15,904** |
| Missing split assignment | 1,414 | **0** |
| Train/test overlap | 0 | 0 |

The reassigned 1,414 peptides landed **entirely in test** by construction of the fix (they were peptides `filter_out` flagged as too similar to some test peptide, so they could not safely go to train) -- this is not a residual bias to flag, it's the direct, intended consequence of the chosen fix. Test is now ~30.1% of unique peptides instead of the originally targeted 20% (`test_size=0.2` is the input to the clustering step, not a guarantee on the post-filtered/reassigned output).

## Issue 2: peptide_id=21052 whitespace caveat

### What was flagged

QA found peptide_id=21052's sequence clean (`'LLLRRRRLL'`) in the final dataset, contradicting the audit report's caveat that its whitespace (`' LLLRRRRLL'`) was "not corrected here."

### Root cause

Not a code bug. The whitespace **still exists**, unchanged, in `data/qmap_included.csv` (QMAP's own pulled copy -- `scripts/02_pull_qmap.py` only flags it via a diagnostic check, never corrects it). It was never present in the data that reaches the final dataset: `scripts/06_assay_filter_units.py` sources every row's `sequence` field from `DBAASPPeptide` objects built off `.cache/dbaasp_raw.jsonl` (this project's own independent DBAASP REST crawl), which already had `"sequence": "LLLRRRRLL"` clean at the source (`scripts/common/parse_dbaasp.py` has no `.strip()` anywhere). The two "copies" of this peptide's sequence simply come from two different data sources and were always different -- nothing was silently corrected mid-pipeline. The audit report's original wording conflated "QMAP's copy" with "the final dataset's copy" as if they were the same field.

### Fix applied

- `scripts/08_audit_report.py`'s peptide_id=21052 caveat bullet rewritten to state precisely which copy has the whitespace (QMAP's pulled file only) and why the final dataset is unaffected (independent source, already clean).
- `scripts/qa/section6_sanity_checks.py`'s check rewritten to verify both locations explicitly -- `data/qmap_included.csv` is expected to retain the whitespace, the final dataset is expected to be clean -- so the check can detect real future drift (either copy changing) instead of permanently flagging a now-understood non-issue.

### Other caveats re-verified against current data

| Caveat | Status |
|---|---|
| peptide_id=21052 whitespace | Corrected wording (see above) -- underlying fact unchanged |
| Diff bucket b5 (`b5_excluded_unexplained`, 1,619) | Still accurate, recomputed from `data/dbaasp_vs_qmap_diff.csv` |
| Diff bucket c (`c_in_qmap_not_in_raw_pull`, 0) | Still accurate |
| Residue-code mapping scope | Still accurate -- matches `scripts/common/residue_map.py` |
| Bond-based recovery scope (SS/HT only) | Still accurate -- matches `scripts/common/smiles_gen.py` |
| N-/C-terminal modification scope (ACT/AMD only) | Still accurate -- matches `scripts/common/smiles_gen.py` |
| Split-failure callout | Was **materially false** (silently didn't cover partial drops) -- now fixed, see Issue 1 |

## Re-verification

Re-ran `scripts/qa/run_qa.py` end-to-end after applying both fixes and regenerating `data/final_mic_regression_dataset.csv`, `data/split_indices.json`, `reports/step6_final_split_log.txt`, and `reports/curation_audit.md`:

| Section | Before | After |
|---|---|---|
| 1. Coverage vs QMAP baseline | PASS | PASS |
| 2. SMILES structural integrity | PASS | PASS |
| 3. Independent unit-conversion spot-check | PASS | PASS |
| 4. Independent MIC/Bacteria filter verification | PASS | PASS |
| 5. Train/test split integrity | **FAIL** | **PASS** |
| 6. Row-level sanity & documented-caveat checks | **FLAGGED** | **PASS** |

Section 5 metrics after the fix: `n_train=11114`, `n_test=4790`, `n_overlap=0`, `n_orphan_final_not_split=0`, `n_orphan_split_not_final=0`, `n_row_level_inconsistent=0`.

Section 6 metrics after the fix: `flag_21052_status=as_expected` (`qmap_included=' LLLRRRRLL'`, `final='LLLRRRRLL'`); all other Section 6 checks (duplicates, value sanity, mic_type self-consistency) unchanged at 0/matching, since final dataset row content was not altered by this fix -- only the split assignment and documentation changed.

Sections 1-4 are unaffected by either fix (final dataset content is byte-for-byte the same set of MIC rows; only `data/split_indices.json` and the two caveat bullets in `reports/curation_audit.md` changed), consistent with their PASS status carrying over unchanged.
