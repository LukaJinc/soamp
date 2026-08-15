# Curation audit report: SMILES-based AMP MIC regression dataset

Generated: 2026-08-15

## Overview

This dataset extends QMAP's (Lavertu, Corbeil & Germain, bioRxiv 10.64898/2026.02.03.703041) published AMP MIC regression benchmark by independently re-pulling the full DBAASP corpus via its public REST API, diffing it against QMAP's included set, and recovering non-canonical/cyclic peptides that QMAP's filter excludes but that are structurally resolvable (via native DBAASP SMILES or p2smi-generated SMILES). All MIC values are standardized to uM using molecular weight computed from each peptide's SMILES structure via RDKit -- never an assumed conversion factor.

**Final dataset size: 72587 (peptide, organism) MIC rows, 15904 unique peptides (13680 from QMAP's original set, 2224 newly recovered).**

## Step 1 -- QMAP included dataset (baseline)

Source: `anthol42/qmap_benchmark_2025` on Hugging Face (`dbaasp.json`), which is the direct output of QMAP's own `build_dbaasp_dataset()` pipeline (verified by reading github.com/anthol42/QMAP source directly, not assumed from the paper alone).

```
=== Step 1: QMAP included dataset pull ===
Source: huggingface anthol42/qmap_benchmark_2025 / dbaasp.json
Total peptide records in QMAP dataset: 18033
Peptides with >=1 bacterial MIC target (species) recorded: 13761
Peptides with native DBAASP SMILES present: 14785 (82.0%)
  - of which have >1 SMILES candidate listed: 9702
Peptides flagged has_noncanonical=True: 1964 (10.9%)
Peptides canonical-only: 16069

Non-canonical breakdown (peptide-level, by residue kind combination present):
  canonical: 16069
  D-canonical-residue: 1350
  other_unresolved(has_smiles): 254
  D-canonical-residue+ornithine: 137
  ornithine: 87
  D-canonical-residue+other_unresolved(has_smiles): 70
  dab: 32
  D-canonical-residue+dab: 18
  ornithine+other_unresolved(has_smiles): 5
  D-canonical-residue+dab+ornithine: 4
  dab+ornithine: 3
  dab+other_unresolved(has_smiles): 3
  other_char: : 1

Bond type counts (peptide-level, DSB/AMD only per QMAP filter):
  none: 14661
  DSB: 2358
  AMD: 1214

MIC unit: uM for all 'targets' entries (QMAP's target_base.py converts ug/mL -> uM via computed molecular weight before this file is produced; confirmed by reading QMAP source, no unit field is stored in this flattened export so this is a methodology-level confirmation, not a per-row field).
Unique organisms across all (peptide, organism) MIC rows: 492
Total (peptide, organism) MIC rows written to /Users/lukajin/PycharmProjects/soamp/data/qmap_included.csv: 62729
Peptides with zero bacterial MIC targets (present in QMAP's 18,033 but contribute 0 rows to the flattened regression table -- e.g. hemolysis-only entries or species that didn't pass QMAP's Bacteria/MIC/unit filters): 4272

MANUAL REVIEW FLAGS (1):
  peptide_id=21052: sequence has leading/trailing whitespace: ' LLLRRRRLL'

Top 20 organisms by row count:
  Escherichia coli: 11059
  Staphylococcus aureus: 10252
  Pseudomonas aeruginosa: 7502
  Bacillus subtilis: 3398
  Klebsiella pneumoniae: 3045
  Staphylococcus epidermidis: 2718
  Acinetobacter baumannii: 2399
  Enterococcus faecalis: 2124
  Salmonella enterica: 2019
  Micrococcus luteus: 1272
  Salmonella typhimurium: 968
  Enterococcus faecium: 793
  Listeria monocytogenes: 709
  Bacillus cereus: 708
  Enterobacter cloacae: 558
  Bacillus megaterium: 420
  Pseudomonas syringae: 365
  Streptococcus pyogenes: 352
  Proteus mirabilis: 312
  Streptococcus mutans: 303
```

## Step 2 -- Full DBAASP raw pull (independent, direct REST API)

Endpoint: `GET https://dbaasp.org/peptides/{id}` (confirmed against QMAP's own fetch code and the official `dbaasp_api_helper_libraries` repo -- the `/api/v1/...` and `/api?page=rest` paths suggested by the DBAASP web UI do not serve a REST API directly; they are the Angular SPA shell). IDs 1 through 24,707 were probed (24,207 = QMAP's January-2026 snapshot max id, plus a 500-id margin to catch anything added since).

Crawl mechanics: concurrent requests via a thread pool. An initial run at 50 concurrent workers sustained ~12 req/s cleanly for the first ~11,700 requests, then began timing out on nearly every request (the server appears to have a sustained-load threshold); the crawl was restarted at 20 workers with resume-from-checkpoint support (already-fetched IDs skipped, results appended), which completed the remainder without further incident. This is noted here because it affected wall-clock time, not data completeness -- every ID was eventually fetched or logged as not_found/error.

```
DBAASP crawl summary
IDs attempted: 263 (1..24707)
ok (200): 0
not_found (400, gap/deleted id): 263
error (network/timeout/5xx after retries): 0
elapsed_seconds: 3.4
```

```
TRUE TOTALS (recomputed directly from .cache/dbaasp_raw.jsonl, since the crawl was interrupted once and resumed -- the log above only reflects the final resumed segment's in-process counters, not the full run):
  ID space probed: 1..24707 (24707 ids)
  Unique peptides successfully fetched (200 OK): 24444
  Not-found (400, gap/deleted id): 263 (re-confirmed in the final resumed pass, which re-attempts any id not already saved -- so this figure is a true global count, not segment-local)
  Outstanding errors after retry: 0 (the single remaining timeout, id=8244, was manually re-fetched successfully and appended)
  Check: 24444 + 263 = 24707 (== 24707, full id space accounted for)
```

```
=== Step 2: DBAASP raw pull (direct REST API) ===
Endpoint: GET https://dbaasp.org/peptides/{id}, Accept: application/json
Total entries successfully pulled: 24444
Entries with native SMILES field populated: 20763 (84.9%)
Entries flagged has_noncanonical: 6274 (25.7%)

Complexity breakdown (Monomer / Dimer / Multi peptide -- only Monomer is in scope for a single-chain SMILES regression dataset; non-monomers are logged here and excluded downstream in the diff step):
  Monomer: 23780
  Multimer: 428
  Multi-Peptide: 236

Non-canonical residue codes detected (unusualAminoAcids.modificationType.name), with occurrence counts across peptide entries (629 distinct codes):
  DAB: 1580
  ORN: 1162
  NLys: 604
  S-ALA-4-pen: 602
  AIB: 452
  NLE: 312
  Tic: 272
  ABU: 252
  DHB: 247
  Oic: 228
  DAP: 224
  Nspe: 221
  Npm: 202
  D-DAB: 185
  βALA: 171
  2-NAL: 169
  D-ORN: 166
  D-Allo-ILE: 158
  BIP: 139
  D-Me-Phe: 135
  1-NAL: 129
  βNPhe(F): 126
  PYA: 123
  Me-Leu: 118
  Me-Val: 117
  Sar: 115
  4-HYP: 107
  DHA: 106
  Ac6c: 99
  DIP: 98
  Acm-Cys: 97
  D-1-NAL: 95
  hArg: 86
  DHF: 77
  βNPe(F): 73
  βNPhe: 72
  AEEA: 69
  Me-Phe: 69
  NLeu: 65
  Ahx: 60
  8-Aoc: 60
  D-Allo-Thr: 58
  Me-Ala: 56
  LYS-C8: 56
  Dab(βOH): 56
  Allo-Thr: 54
  LYS-C10: 52
  Nspe(p-Br): 51
  D-NLE: 50
  (S,S)-ACPC: 50
  MET(O): 49
  2-Abz: 47
  NTrp: 47
  R-ALA-7-oct: 46
  Nnm: 45
  Allo-ILE: 42
  Nap: 42
  Cha: 41
  Npm(p-F): 40
  LYS-C16: 39
  Iva: 39
  NVA: 37
  3-Abz: 37
  Nspe(p-Cl): 37
  Me-Thr: 36
  Npm(p-Cl): 36
  βNspe: 35
  LYS-C4: 34
  D-2-NAL: 33
  LYS-C6: 33
  12-NH2-C12: 32
  HSer: 32
  APC: 29
  βNLys: 28
  Npm(p-I): 28
  Npm(p-Br): 28
  CIT: 27
  3-Me-Val: 27
  Me-Ile: 27
  LYS-(CH3)3: 26
  4-F-PHE: 26
  LYS-C12: 25
  Me-LYS: 25
  2-NH2-C12: 25
  GABA: 24
  Cys-N-C12-AcNh: 24
  N-TYR: 24
  DHArg: 24
  Ac5c: 22
  Ndpe: 21
  N5-Ac-N5-OH-ORN: 21
  1(N)-BIP-TRP: 21
  NVal: 20
  LYS-ACT: 19
  Chg: 19
  NNLe: 19
  β-hARG: 18
  Pra: 18
  3-OH-Leu: 18
  AdaGly: 17
  TERT BU PHE: 17
  12-Guan-C12: 17
  3-OH-Asp: 17
  Lys(N3): 16
  Npm(p-CH3): 16
  3-ME-GLU: 16
  P-Ser: 15
  Ada-Ala: 15
  MET(O2): 15
  O-Me-Tyr: 15
  4-Cl-PHE: 15
  4-Me-Pro: 15
  3,5-F-PHE: 15
  6F-LEU: 15
  3-Hyp: 15
  2-Ahx: 14
  DHTrp: 14
  3-Me-Trp: 14
  D-Me-Ser: 14
  Nhe: 14
  βNPhe(CH3): 14
  βNPhe(F3): 14
  βNPhe(CF3): 14
  βNCha: 14
  βNpe: 14
  Me-ARG: 13
  Piz: 13
  5-OH-TRP: 13
  HPhe: 13
  KYN: 13
  4-NH2-Pro: 13
  D-3-OH-ASN: 13
  SER(GlcNAc): 13
  D-3-OH-Leu: 13
  CYS-mPEG2000: 12
  Me-Ser: 12
  TOAC: 12
  5F-PHE: 12
  Pip: 12
  CYS-PEG4-Chol: 12
  Me-4-OH-Phg: 12
  DIPM-CYS: 12
  D-Piz: 11
  D-Me-Leu: 11
  D-4-HYP: 11
  MET-Boro: 11
  Ala-d3: 11
  Me-TYR: 11
  TERT BU TRP: 11
  3,4-Me-Gln: 11
  D-3-OMe-Ala: 11
  Phg: 10
  D-ABU: 10
  CF3-Bpg: 10
  5-Br-TRP: 10
  4-NO2-PHE: 10
  HLeu: 10
  6-F-TRP: 10
  LYS-C18: 10
  Me-Asn: 10
  β-hTrp: 10
  4-3Fm-PHE: 10
  Nae: 10
  Npe: 10
  Eps-LYS: 10
  MIM: 10
  D-Pip: 10
  3-Me-Pro: 10
  cis-3-Hyp: 10
  CYS-mPEG750: 9
  BDZ: 9
  ADMArg: 9
  Se-Cys: 9
  S-tBu-CYS: 9
  N-Me-5-OH-Leu: 9
  PHE(NH2): 9
   ALA(O)S: 9
  D-TERT BU TRP: 9
  AGP-Pro: 9
  βNsche: 9
  tBu-βAc6c: 9
  LYS-C14: 8
  AGP: 8
  NArg: 8
  D-Me-Ala: 8
  D-LAP: 8
  LAP: 8
  D-LYS-C6: 8
  DAA: 8
  Thi-ALA: 8
  D-LYS-C10: 8
  Nai: 8
  Er-Pre-TRP: 8
  2-DMA-TRP: 8
  4-OH-Phg: 8
  Dab(Thr): 8
  D-3-OH-Asp: 8
  DHγAla: 8
  3,5-Abz: 8
  Nspe(p-CH3): 8
  Nspe(p-F): 8
  TYR-Bzl: 7
  Arg(NO2): 7
  Me-CYS: 7
  DAB-C10: 7
  1(N)-Me-TRP: 7
  D-Gly-C8: 7
  7-DMA-TRP: 7
  4-Cl-HPro: 7
  O-Me-Thr: 7
  DOPA: 6
  PFPh: 6
  6-Br-trp: 6
  Asn(GlcNAc) : 6
  Me-OMe-Trp: 6
  D-4-Cl-PHE: 6
  D-DAP: 6
  βLeu: 6
  His((CH2)7C6H5)2: 6
  D-Me-TYR: 6
  Et-Cys: 6
  3-F-TYR: 6
  HPro: 6
  2,6-F-PHE: 6
  OMe-Trp: 6
  Me-GLN: 6
  THR-Cl: 6
  Pen: 6
  3-OH-VAL: 6
  D-Iva: 6
  Me-3-OH-Val: 6
  4-3F-ABU: 6
  D-3-OMe-Tyr: 6
  Fur-Me-Ala: 6
  S-Me-CYS: 6
  DHγPhe: 6
  7-NH2-C7: 5
  3-OH-PHE: 5
  SDMArg: 5
  D-LYS-C4: 5
  Me-Trp: 5
  NIle: 5
  N-DMA-TRP: 5
  PHE(NH2-Bz): 5
  BTA: 5
  S-S-tBu-CYS: 5
  CYS-C16: 5
  Aoda: 5
  3-Me-ASP: 5
  N,O-Me-TYR: 5
  D-DOPA: 5
  3-OH-ASN: 5
  D-3,5-OH-Phg: 5
  D-4-OH-Piz: 5
  D-Me-Asn: 5
  Val(O)S: 5
  D-3-OMe-Tyr-O-GLC: 5
  LYS-Bu: 5
  Me-Nva: 5
  D-3-NH2-C10: 5
  Nphb: 5
  Thr(GalNAc): 4
  D-PYA: 4
  Y-SO3H: 4
  4,5-OH-LYS: 4
  D-HPhe: 4
  CYS-S-CH3: 4
  D-CF3-Bpg: 4
  OMe-Glu: 4
  DEtGly: 4
  N,N-Me-Lys: 4
  AzAla: 4
  D-LYS-C8: 4
  His(chx)2: 4
  Me-Allo-ILE: 4
  DAB-C12: 4
  DAB-C14: 4
  Agb: 4
  4-Br-PHE: 4
  R-ALA-4-pen: 4
  D-TYR-Bzl: 4
  Bn-CYS: 4
  β-BzThi-Ala: 4
  3-NH2-5,6-OH-7-Xyl-C18: 4
  O-Me-D-Tyr: 4
  N,O-Me-Ser: 4
  OH-Ile: 4
  3-OH-N1,N3-Me-His: 4
  4-2F-ABU: 4
  4-F-ABU: 4
  N,3-Me-LEU: 4
  DAP-Bu: 4
  DHγTyr: 4
  Lys-Alloc: 4
  Lys-Pha: 4
  N1,N3-C10-His: 4
  Nprg: 4
  Nbg: 4
  AGL: 3
  D-TIC: 3
  βASP: 3
  2-NH2-C11: 3
  Ac3c: 3
  Ph-TRP: 3
  2-Aoc: 3
  10-Adc: 3
  AHMOD: 3
  HCys: 3
  Thr(GlcNAc): 3
  Dpg: 3
  D-LYS-C14: 3
  D-LYS-C16: 3
  D-LYS-C12: 3
  CYS-S-HexNAc: 3
  D-BIP: 3
  N(5)-OH-ARG: 3
  4-I-PHE: 3
  2-Adc: 3
  Ntridec: 3
  Glu-OAll: 3
  Iaa: 3
  O-Act-Thr: 3
  D-4-OH-Phg: 3
  3-I-Tyr: 3
  CYS-PEG12-Chol: 3
  5-Cl-TRP: 3
  β-AIB: 3
  D-3-NH2-14-Me-C16: 3
  P-Thr: 3
  D-IGln: 3
  D-CYA: 3
  Met(O)S: 3
  Ndec: 3
  D-3-NH2-15-Me-C16: 3
  3-NH2-2-OH-4-Me-15-oxo-C18: 3
  3,5-Br-4-Ome-Phe: 3
  N-Bip-D-Arg: 3
  N-Bip-Arg: 3
  Npm(3,5-CF3): 3
  Npm(3,5-CH3): 3
  α-Me-Phe: 3
  3-Ome-Bn-CYS: 3
  SER(GalNAc): 2
  5-Me-GLU: 2
  Et-GLU: 2
  TERT BU ALA: 2
  5-OH-LYS: 2
  D-FLG: 2
  D-DIP: 2
  D-NVA: 2
  NNar: 2
  DHL: 2
  Aic: 2
  Epa: 2
  LYS-Me-C5: 2
  LYS-Me-C6: 2
  D-LYS-Me-C5: 2
  HTrp: 2
  Api: 2
  D-Pra: 2
  IAA-Cys: 2
  5-NH2-C5: 2
  Hex-Hyp: 2
  LYS-C7: 2
  Lys-C11: 2
  Lys-C9: 2
  FLG: 2
  NHet: 2
  N,4-Me-Glu	: 2
  LYS-C24: 2
  N(Me)Bmt(E): 2
  LYS-PEG8: 2
  Iac: 2
  5-F-Trp: 2
  DAB-C8: 2
  DAB-C4: 2
  DAB-C6: 2
  3-Ud-DHA: 2
  CAM: 2
  βLYS: 2
  Chx-PRO: 2
  2-F-PHE: 2
  mPEG-5000-Mal-CYS: 2
  mPEG-2000-Mal-CYS: 2
  3-NH2-C14: 2
  3-NH2-14-Me-C16: 2
  DIT: 2
  Cbz-ORN: 2
  3-OH-GLN: 2
  HTyr: 2
  CYS-Chol: 2
  iPr-Lys: 2
  D-2-AAA: 2
  MPro-HEMOAH: 2
  5-Me-OPro: 2
  D-3-Me-Asp: 2
  6-Cl-Trp: 2
  4-Cl-KYN: 2
  4-OH-Glu: 2
  D-3-NH2-13-Me-C14: 2
  3-Me-Phe: 2
  OH-GLY: 2
  α-thio-α-(OH)PHE: 2
  Me-α-Thio-α-SER: 2
  Suc-Lys: 2
  19-Guan-3-OH-C19: 2
  OH-VAL: 2
  D-3-O-TRP: 2
  3-Me-Ile: 2
  2,5-Me-(E)2-4-Me-NH2-C6: 2
  3-OH-Ile: 2
  4-2F-5-3F-NVA: 2
  PhSar: 2
  β-hPhe: 2
  13-N,N-Me-Guan-(E)2,4,8-C13: 2
  3F-NLE: 2
  N-DMA-3-OH-TRP: 2
  3-Ome-Phe: 2
  D-MET(O): 2
  C7G: 2
  D-3-NH2-C8: 2
  N-Bn-Arg: 2
  N-Bip-Orn: 2
  N-Bip-Har: 2
  N-Bip-Lys: 2
  NHis: 2
  4-Me-Pip: 2
  β3,3-Pip(C12): 2
  β2,2-Ac6c: 2
  β3,3-Pip(NC12): 2
  3,4-OH-ARG: 1
  4-NH2-D-PRO: 1
  DHV: 1
  LYS-C12-OH: 1
  Ac4c: 1
  (R,R)-ACPC: 1
  Allo-End: 1
  DAP(Ac): 1
  DAB(Ac): 1
  LYS-C5: 1
  LYS-Me-C8: 1
  2-OH-Me-SER: 1
  D-2-Aoc: 1
  AMOD: 1
  TERT BU SER: 1
  15-Guan-3-OH-C15: 1
  LEU-Boro: 1
  Act-Cys: 1
  Aza-gly: 1
  2,4,6-F-PHE: 1
  DAP-C12: 1
  DAP-C14: 1
  DAP-C10: 1
  DAP-C16: 1
  DAB-C16: 1
  His((CH2)3C6H5)2: 1
  His((CH2)7C6H5): 1
  N,N,N-Me-D-Orn: 1
  MOR-D-Orn: 1
  iPr-D-Orn: 1
  N,N-Me-D-Orn: 1
  ACT-D-Orn: 1
  NF-KYN: 1
  Dab(Cys): 1
  Anth-ALA: 1
  D-Me-Trp: 1
  Di-P-Tyr: 1
  LYS-BrC6: 1
  LYS-NH2-C5: 1
  LYS-Ahx: 1
  LYS-NH2-C11: 1
  HCha: 1
  SER-O-BZY: 1
  Leu(O)S: 1
  BisHomo-Pra: 1
  D-AGL: 1
  3-OH-Er-Pre-TRP: 1
  ADMH: 1
  2-F-TYR: 1
  LYS-PEG4: 1
  LYS-C22: 1
  LYS-C20: 1
  CM-Cys: 1
  D-4-F-PHE: 1
  Hex-CYS: 1
  Gly-C8: 1
  N-Me-5-O-Leu: 1
  β-Thi-D-ALA: 1
  THZ-Ala: 1
  PHE-NH2-C5: 1
  D-ORN-NH2-C7: 1
  D-ORN-NH2-C5: 1
  F-Pro: 1
  CH3-S-Pro: 1
  4,4-F-Pro: 1
  CF3-Pro: 1
  Ph-Pro: 1
  Bn-Pro: 1
  PhO-Pro: 1
  Aohda: 1
  Aoeda: 1
  ASU: 1
  D-Me-3-OH-Phe: 1
  3-NH2-12-Me-C14: 1
  2-NH2-8-OH-C10: 1
  D-Me-Br-Tyr: 1
  OH-2-Abz: 1
  EtGly: 1
  3-NH2-14-Me-C15: 1
  Et-VAL: 1
  3-NH2-12-Me-C13: 1
  D-Lys(N3): 1
  Cbz-D-ORN: 1
  Cbz-LYS: 1
  5-NH2-2-OH-C5: 1
  LYS-EPA: 1
  ORN-EPA: 1
  ORN-PA: 1
  Act-HSer: 1
  LYS-PA: 1
  3-NH2-5,6,7-OH-C18: 1
  3,5-I-Tyr: 1
  IST: 1
  CYS-Aea-PEG24-Chol: 1
  4-AcO-DHV: 1
  N(4-Ome-Bn)-3,4-OH-Phg: 1
  4-Abz: 1
  3-OH-5-Me-Pro: 1
  5-Me-PRO: 1
  THR-OH: 1
  D-Dab(Thr): 1
  D-HLeu: 1
  N,S-Me-CYS: 1
  Me-MET: 1
  5-Me-DHTrp: 1
  3,5-Me-Trp: 1
  5-OH-DHTrp: 1
  3-Me-5-OH-Trp: 1
  5-Ome-DHTrp: 1
  3-Me-5-Ome-Trp: 1
  D-3-Me-Trp: 1
  3-OMe-Asp: 1
  D-3-NH2-12-Me-C13: 1
  Dab(βOH)-C10: 1
  Dab(βOH)-C14:1: 1
  Dab(βOH)-C12:1: 1
  Dab(βOH)-C12: 1
  O-EA-4-OH-Phg: 1
  O-EA-Tyr: 1
  D-3-NH2-14-Me-C15: 1
  D-Pen: 1
  3-Me-HIS: 1
  19-Guan-C19: 1
  D-HSer: 1
  11-NH2-C11: 1
  2,6-OH-PHE: 1
  Act-Thr: 1
  3-NH2-2-OH-4,15-Me-(E)14-C19: 1
  3-NH2-2-OH-4-Me-C14: 1
  3-NH2-2-OH-4-Me-C12: 1
  D-4-O-Piz: 1
  D-3-Cl-3-OH-Tyr: 1
  D-Van-Glu-Phg: 1
  3-Cl-3-OH-Tyr: 1
  3,5-OH-Phg: 1
  2-NH2-2,6,8-Me-C10: 1
  D-OMe-Glu: 1
  N1,N2-Me-ORN: 1
  3-OH-Br-Phe: 1
  D-Me-Val: 1
  N,O-Me-Glu: 1
  N,3,3-Me-PHE: 1
  N,3,3-Me-TRP: 1
  3-Me-Leu: 1
  15-N,N-Me-Guan-C15: 1
  5-OH-Phe(O)S: 1
  N1,N3-Me-His: 1
  MET(O2)S: 1
  TYR-O-GLC: 1
  2-OH-Phe: 1
  D-SER-O-OMe-CIN: 1
  D-3-OH-GLN: 1
  D-SER-O-Me-CIN: 1
  SER-O-Me-CIN: 1
  D-PhSar: 1
  D-2-HYP: 1
  D-OH-Leu: 1
  DHDMP-Trp: 1
  2-HYP: 1
  N-Me-4,5-OMe-Leu: 1
  DHLys: 1
  N(ω)-Me-ARG: 1
  Suc-Orn: 1
  1(N)-Me-KYN: 1
  Me-ABU: 1
  SER-O-C4MA: 1
  3-NH2-16-Me-C17: 1
  3-NH2-15-Me-C16: 1
  3-NH2-10-Me-C12: 1
  D-3-NH2-12-Me-C14: 1
  D-3-NH2-10-Me-C12: 1
  4-NH2-2,2-Me-3-oxo-C5: 1
  3-NH2-2-Me-C5: 1
  3-NH2-2,15-OH-4-Me-C18: 1
  ACPB: 1
  3-F-ALA: 1
  DHγleu: 1
  DHγVal: 1
  Ome-BIP: 1
  3',5'-Br-4'-Ome-BIP: 1
  3-Br-4-Ome-Phe: 1
  Lys-Paa: 1
  Lys-Chb: 1
  Lys-Pba: 1
  Lys-IBA: 1
  N1,N3-C7-His: 1
  N1-C10-His: 1
  N1-(E)2-C10-His: 1
  N1,N3-C14-His: 1
  Me-His: 1
  4-ip-Pip: 1
  D-3,5-F-PHE	: 1
  3,4,5-F-PHE: 1
  3-F-PHE: 1
  D-Chg: 1
  D-Cha: 1
  D-3-Me-Bn-CYS: 1
  Tr-CYS: 1
  S-Pr-Cys: 1

Entries with >=1 intrachain bond of a type other than DSB/AMD: 1100
All intrachain bond types observed, with entry counts:
  DSB: 2602
  AMD: 1950
  EST: 440
  TIE: 290
  DCB: 255
  AMN: 38
  CAR: 35
  p-XylB: 23
  (E)-but-2-enyl-B: 19
  ETH: 18
  TRZB: 10
  BisMeBn: 3
  but-2-ynyl-B: 2
  IMN: 2

Non-DSB/AMD bond types specifically, with entry counts:
  EST: 440
  TIE: 290
  DCB: 255
  AMN: 38
  CAR: 35
  p-XylB: 23
  (E)-but-2-enyl-B: 19
  ETH: 18
  TRZB: 10
  BisMeBn: 3
  but-2-ynyl-B: 2
  IMN: 2
```

## Step 3 -- Diff: DBAASP raw pull vs. QMAP's included set

```
=== Step 3: DBAASP raw pull vs QMAP included set -- diff ===
Join key: DBAASP integer peptide id (native to both datasets)
Raw pull peptides: 24444
QMAP included peptides: 18033

Bucket counts:
  b1_excluded_non_monomer: 664
  b2_excluded_unsupported_terminus: 2565
  b3_excluded_residue_no_smiles: 1428
  b4_excluded_nonstandard_bond: 135
  b5_excluded_unexplained: 1619
  c_in_qmap_not_in_raw_pull: 0
  in_both: 18033

IMPORTANT METHODOLOGICAL FINDING (from reading QMAP's build_dataset.py source directly): QMAP excludes a peptide for residue reasons ONLY when an unresolved 'X' placeholder remains AND no native SMILES is available. A non-canonical residue that already has a native DBAASP SMILES is NOT excluded by QMAP for residue reasons -- so 'non-canonical WITH SMILES available' is not a meaningful separate exclusion bucket; those peptides are already in QMAP's baseline (bucket in_both) unless they also fail the terminus or bond-type checks. The real recovery targets are bucket b2 (terminus modifications like lipidation/PEGylation QMAP doesn't support) and b4 (bond types like thioether/lactam/lactone cyclization QMAP excludes unconditionally, regardless of SMILES availability).

Bucket b2 terminus detail (which specific mod caused exclusion):
  nterm:C8: 277
  nterm:C16: 220
  nterm:C12: 209
  nterm:C14: 172
  nterm:C10: 158
  nterm:C18: 68
  nterm:C6: 62
  nterm:3,5 Bis-(Me)Tol: 60
  nterm:C9: 41
  nterm:3-OH-5-Me-C6: 34
  nterm:(CH3)2: 32
  nterm:4Me-GUAN: 30
  nterm:6-Me-C8: 29
  nterm:C4: 29
  nterm:C7: 27
  cterm:PUT: 27
  nterm:FOR: 25
  nterm:3-OH-C14: 25
  cterm:PEA: 25
  nterm:OIle: 23
  nterm:FMOC: 22
  cterm:A(P): 22
  cterm:PHEol: 20
  nterm:4-Me-C6: 20
  nterm:BZA: 19
  nterm: OLeu: 18
  nterm:AdAcA: 16
  cterm:OBzl: 16
  cterm:OMe: 15
  nterm:2-Me-3-OH-C9: 15
  nterm:3-OH-C10: 15
  nterm:C11: 14
  nterm:C5: 14
  nterm:2-Me-CIN: 14
  nterm:6-Me-C7: 13
  cterm:EN: 12
  cterm:C16-NH2: 12
  nterm:2-Me-3-OH-C13: 12
  cterm:Chol: 11
  cterm:LEUol: 10
  nterm:DiMIQ: 10
  nterm:TFA: 10
  cterm:C12H25-NH2: 10
  nterm:I3CA: 10
  nterm:Cl-Th2CA: 10
  nterm:SUC: 9
  nterm:TPA: 9
  cterm:RIM: 9
  nterm:CIN: 8
  nterm:LIN: 8
  nterm:Chol: 8
  nterm:4-BCA: 8
  nterm:(E)2-C7: 8
  nterm:DNS: 7
  nterm:C20: 7
  nterm:OVal: 7
  nterm:D-OVal: 7
  nterm:D-OIle: 7
  nterm:(Z)4-(E)6-2,6,8-Me-2,3-OH-C10: 7
  cterm:TRPol: 6
  nterm:C15: 6
  cterm:Am-Chol: 6
  nterm:3-OH-4-Me-C10: 6
  nterm:6-Pha: 6
  nterm:3-OH-C12: 6
  nterm:TOS: 5
  nterm:HPPA: 5
  nterm:3-OH-11-Me-C13: 5
  nterm:Boc: 5
  nterm:16-OH-C16: 5
  nterm:C13: 5
  nterm:D-OPhe: 5
  nterm:Pyr: 5
  nterm:FLCpOH: 5
  nterm:(4-Cl-Ph)Ac: 5
  nterm:F5PhAc: 5
  nterm:F3PhAc: 5
  nterm:cHexAc: 5
  nterm:(5-Br-indol)Ac: 5
  nterm:pHCA: 4
  nterm:3MeACl: 4
  cterm:DiMIQ: 4
  cterm:BY: 4
  nterm:3-OH-C16: 4
  nterm:2-Me-C10: 4
  nterm:Pic: 4
  nterm:3-OH-7-Me-C8: 4
  cterm:NH-NH2: 4
  nterm:VitE: 4
  nterm:CAA: 4
  nterm:7-Me-C8: 4
  nterm:DiPhAc: 4
  nterm:(E)2,4,6-C8: 4
  nterm:2,4-Me-C8: 4
  nterm:1,3-Me-LUM: 4
  nterm:2-O-C4: 4
  nterm:3-OH-Pic: 4
  cterm:3,5-Br-4-Ome-PEA: 4
  nterm:(E)2-C8: 4
  cterm:LEUal: 4
  nterm:Bn: 3
  cterm:NME: 3
  nterm:B7: 3
  nterm:3-OH-9,11-Me-C13: 3
  cterm:C16: 3
  cterm:Sprd: 3
  nterm:3-OH-6-Me-C8: 3
  nterm:MPA: 3
  nterm: C10(6): 3
  nterm:3-OH-4-Me-C6: 3
  nterm:Cbz: 3
  nterm:13-Me-(E)2-C14: 3
  cterm:PEG4-Chol: 3
  nterm:PEG 5000: 3
  nterm:Pac: 3
  nterm:FeC(O): 3
  nterm:3-OH-4-Me-C8: 3
  nterm:2-ClPhNHCO: 3
  nterm:4-Me-Bz: 3
  nterm:PcC(O)Zn: 3
  nterm:12-Me-C13: 3
  nterm:2-OH-C3: 3
  nterm:3-OH-9-Me-C10: 3
  nterm:2,6-Me-3,5,11-OH-C12: 3
  nterm:MurNAc: 3
  nterm:COC3H6CO: 3
  nterm:CHP: 3
  nterm:C12-CA: 3
  nterm:Ch: 2
  nterm:3-OH-8,10-Me-C12: 2
  nterm:3-OH-C6: 2
  nterm:2MeACl: 2
  cterm:AM: 2
  nterm:C3: 2
  cterm:C3H7-NH2: 2
  nterm:C9B: 2
  nterm:F7: 2
  nterm:F9: 2
  nterm:F9B: 2
  nterm:F11: 2
  nterm:2,2-Me-C4: 2
  nterm:GER: 2
  nterm:C11:1: 2
  nterm:DiFPhAct: 2
  cterm:C16H33-NH2: 2
  nterm:3-OH-4-Me-C5: 2
  cterm:C6-NH2: 2
  nterm:GUAN: 2
  nterm:3-OH-13-Me-C14: 2
  cterm:C6H13-NH2: 2
  cterm:C10H21-NH2: 2
  cterm:C8H17-NH2: 2
  cterm:EthA: 2
  nterm:RuC(O): 2
  nterm:SA: 2
  nterm:4,4-Cl-3-Me-C4: 2
  nterm:12-Me-(Z)3-C13: 2
  nterm:18:1 cis-9: 2
  nterm:3-OH-14-Me-C15: 2
  nterm:5-OH-C14: 2
  nterm:3-OH-C8: 2
  nterm:(E)5-C12: 2
  nterm:Naproxen: 2
  nterm:LA: 2
  nterm:3-OH-C15: 2
  nterm:8-Me-C10:2: 2
  nterm:8-Me-C9:2: 2
  nterm:9-Me-C10: 2
  nterm:14-Me-C15: 2
  nterm:10-Me-C11: 2
  nterm:10-Me-C12: 2
  nterm:3-OH-11-Me-C12: 2
  nterm:D-OLeu: 2
  nterm:Dhoya: 2
  nterm:3-OH-5-Ph-(E)4-C5: 2
  nterm:3-OH-2,4-Me-C12: 2
  nterm: (Z)4-(E)6-2,6,8-Me-3-OH-C10: 2
  nterm:3-OH-6,8-Me-(Z)4-C9: 2
  nterm:4-Chb: 2
  nterm:4-Pba: 2
  nterm:IBA: 2
  nterm:(E)2-C6: 2
  nterm:(E)2,4,6,8,10-HOCP: 2
  nterm:4-F-CIN: 2
  nterm:PEG 3000: 1
  cterm:dc Delta DOPA: 1
  nterm:Fluo: 1
  nterm:PEG 2000: 1
  nterm:7,9-Me-(E)2-C11: 1
  nterm:3-OH-C7: 1
  cterm:DAP: 1
  nterm:3-OH-4-Me-C16: 1
  nterm:3,4-OH-4-Me-C16: 1
  nterm:C20:1: 1
  nterm:4-OMe-C10: 1
  nterm:17-Me-C18:1: 1
  nterm:C17:1: 1
  nterm:Ph-Pr: 1
  nterm:3-OH-4-Me-C15: 1
  cterm:ARGol: 1
  cterm:MEA: 1
  cterm:ETA: 1
  cterm:PEG4-C16: 1
  cterm:PEG12-Chol: 1
  cterm:PEG24-Chol: 1
  cterm:PEG8-Chol: 1
  nterm:HABPA: 1
  nterm:OBU: 1
  nterm:OMe-CIN: 1
  nterm:N-CIN: 1
  nterm:2,3-OMe-CIN: 1
  cterm:C4H9-NH2: 1
  nterm:5-Me-C6: 1
  nterm:6,7-OH-(E)2,4-C8: 1
  nterm:(E)2-C4-Thz-Dtena: 1
  nterm:Oxo-SA: 1
  nterm:3-OH-C13: 1
  nterm:Hmoya: 1
  nterm:Keto-Oya: 1
  nterm:3-OH-C20: 1
  nterm:MeO-Oya-2-ene: 1
  nterm:PIV: 1
  nterm:MeOya: 1
  nterm:3-OH-12-Me-C14: 1
  nterm:MeOAc: 1
  nterm:3-OH-2,4-Me-C10: 1
  nterm:3-OH-2,4,6-Me-C12: 1
  nterm:8-Me-C9: 1
  nterm:2-OH-C8: 1
  nterm:AdCA: 1
  nterm:PyBA: 1
  nterm:Di-BrPh-PY: 1
  nterm:PCPC: 1
  nterm:BCHC: 1
  nterm:PhOBz: 1
  nterm:Ph-4-PhNHCO: 1
  nterm:HIP: 1
  nterm:3-OH-C18: 1
  nterm:2,6-Cl-Bz: 1
  nterm:2-Cl-Bz: 1
  nterm:4-Cl-Bz: 1
  nterm:3-Cl-Bz: 1
  nterm:2,4-Cl-Bz: 1
  nterm:2-Cl-4-F-Bz: 1
  nterm:4-Cl-2-F-Bz: 1
  nterm:3,5-Cl-Bz: 1
  cterm:VALol: 1
  nterm:OsC(O): 1
  nterm:CoC(O): 1
  nterm:PEG 3: 1
  nterm:GlcA: 1
  nterm:2-Me-C4: 1
  nterm:5-OH-2,4-Me-3-oxo-C8: 1
  nterm:QUI: 1
  nterm:3-OH-QUA: 1
  nterm:6-Me-(E)2,4-C8: 1
  nterm:(E)2,4-C8: 1
  nterm:2,6-OMe-Bz: 1
  cterm:(CH3)2N: 1
  nterm:TERT-BU-Me-PyCA: 1
  nterm:12-Me-C14: 1
  cterm:HPA: 1
  nterm:3-OH-2,11-Me-C14: 1
  nterm:4Me-3-OH-(E)8,10-C20: 1
  nterm:PCHC: 1
  nterm:PPCA: 1
  nterm:HBz: 1
  nterm:BPCA: 1
  nterm:9-Me-(Z)3-C10: 1
  nterm:10-Me-(Z)3-C12: 1
  nterm:10-Me-(Z)3-C11: 1
  nterm:3-OH-10-Me-C11: 1
  nterm:3-OH-12-Me-C13: 1
  nterm:3,4-OH-C16: 1
  nterm:3,4-OH-C14: 1
  nterm:3-OH-(Z)5-C12: 1
  nterm:3-OH-11-Me-(Z)5-C12: 1
  nterm:3-OH-11-Me-(Z)7-C12: 1
  nterm:Dhoea: 1
  nterm:Dhoaa: 1
  nterm:OPhe: 1
  nterm:Epd: 1
  nterm:3-OH-12,14-Me-C16: 1
  nterm:3-OH-14-Me-C16: 1
  nterm:2,6-Me-3,5-OH-11-O-C12: 1
  nterm:3-O-C10: 1
  nterm:3-O-C8: 1
  nterm:3-O-C14: 1
  nterm:3-O-C12: 1
  nterm:2-OH-4,5-epoxy-C5: 1
  nterm:3,9-OH-C12: 1
  nterm:OGly(allyl): 1
  nterm:D-XA: 1
  nterm:6,7-Ome-(E)2,4-C8: 1
  nterm:5-COOH-(E)2,4-C6: 1
  nterm:6-Ome-6-O-(E)2,4-C6: 1
  nterm:6-Ome-7-OH-(E)2,4-C8: 1
  nterm:(Z)9-C16: 1
  nterm:N(3Me)-2-Me-3,19,21,23,25,27,30-OH-(E)11-C31: 1
  nterm:2-OH-2-Me-4-O-C5: 1
  cterm:Spr: 1
  nterm:(E)19-3-OH-C28: 1
  nterm:3-OH-C26: 1
  nterm:FA: 1
  nterm:pNBA: 1
  cterm:C14: 1
  cterm:3-Br-4-Ome-PEA: 1
  cterm:C7H15-NH2: 1
  cterm:C9H19-NH2: 1
  cterm:C13H27-NH2: 1
  cterm:C14H29-NH2: 1
  cterm:BIP-Me-NH2: 1
  nterm:C14H30: 1
  nterm:C12H26: 1
  nterm:C16H34: 1
  nterm:3-OH-7-S-(E)4-C7: 1
  nterm:CHA: 1
  nterm:4-C8-Bz: 1
  nterm:4-Br-Bz: 1
  nterm:2-Cl-6-Br-Bz: 1
  nterm:I2CA: 1

Bucket b4 non-standard bond type detail:
  TIE: 66
  p-XylB: 19
  (E)-but-2-enyl-B: 19
  EST: 14
  AMN: 10
  ETH: 9
  CAR: 6
  BisMeBn: 3
  but-2-ynyl-B: 2
```

## Step 4 -- Recovery of non-canonical/cyclic peptides

```
=== Step 4: recovery of non-canonical/cyclic peptides excluded from QMAP ===
Total candidate entries considered (buckets b1-b5): 6411
  of which out of scope (non-monomer, not attempted): 664
Recovered: 2870
Unconvertible (logged, excluded): 3541

Recovered by source:
  native_dbaasp_smiles: 2853
  p2smi_generated(linear): 11
  p2smi_generated(SS): 3
  p2smi_generated(HT): 3

Recovered / attempted by bucket (recovery category):
  b1_excluded_non_monomer (out_of_scope): 0 recovered / 664 total
  b2_excluded_unsupported_terminus (terminus_based): 1230 recovered / 2565 total
  b3_excluded_residue_no_smiles (residue_based): 17 recovered / 1428 total
  b4_excluded_nonstandard_bond (bond_based): 6 recovered / 135 total
  b5_excluded_unexplained (unexplained_temporal_drift): 1617 recovered / 1619 total

Unconvertible reason summary (top-level, truncated):
  position 1: 466
  position 2: 197
  unsupported N-terminal modification 'C8': 119
  position 4: 119
  position 3: 101
  position 5: 94
  unsupported N-terminal modification ': 72
  position 6: 68
  unsupported N-terminal modification '3,5 Bis-: 60
  position 8: 50
  position 7: 49
  bond constraint 'SCSC' not auto-generated in this pass: 43
  unsupported N-terminal modification 'C12': 37
  unsupported N-terminal modification '3-OH-5-Me-C6': 34
  2 intrachain bonds present -- multi-bond: 34
  position 9: 32
  position 16: 27
  position 10: 26
  unsupported C-terminal modification 'PEA': 25
  position 18: 24
  unsupported N-terminal modification 'C10': 24
  unsupported N-terminal modification 'FMOC': 22
  unsupported C-terminal modification 'A: 22
  position 11: 21
  unsupported N-terminal modification 'OIle': 21
  unsupported C-terminal modification 'PHEol': 20
  3 intrachain bonds present -- multi-bond: 19
  unsupported N-terminal modification 'BZA': 18
  unsupported N-terminal modification 'C14': 18
  unrecognized DBAASP cycleType='LCN': 18

Full recovered list: /Users/lukajin/PycharmProjects/soamp/data/recovered_peptides.csv
Full unconvertible list (with per-entry reason): /Users/lukajin/PycharmProjects/soamp/data/unconvertible_peptides.csv
```

## Step 5 -- Assay filtering and unit standardization

```
=== Step 5: assay filtering + unit standardization ===
Peptides considered: 20903 (QMAP baseline: 18033, recovered: 2870)
Peptides where SMILES could not be resolved at all (excluded entirely): 114
Peptides where SMILES resolved but RDKit MW computation failed: 0

Total raw targetActivity records examined across included peptides: 163653
Excluded: non-MIC assay type (activityMeasureGroup != 'MIC'): 47604
  top excluded assay types:
    MBC: 12763
    None: 10082
    IC50: 8290
    LC: 2075
    MIC50: 1572
    MFC: 1198
    MIC90: 1091
    EC50: 927
    LD50: 886
    IC50 I: 770
    LC50: 619
    IC50 REP: 610
    MEC: 587
    90-100% Inhibition: 448
    50% Cell death: 331
Excluded: non-bacterial target domain: 11766
  Fungus: 11612
  Other: 97
  Unknown: 56
  Animal: 1
Excluded: unsupported/missing unit: 23
  None: 23
Excluded: MW unavailable at ug/mL->uM conversion time: 627
Excluded: unparseable/missing concentration string: 4

Kept raw (peptide, species, measurement) rows after all filters: 103629
Censoring classification of kept raw measurements (before grouping):
  exact: 74233
  censored: 22583
  ranged: 6813
  missing: 4

Final (peptide, organism) groups: 72587
Groups with >1 raw measurement (averaged): 15456
Groups where IQR removed >=1 outlier point: 2236
Total individual outlier points removed: 4062
```

## Step 6 -- Final dataset assembly and homology-aware split

```
=== Step 6: final dataset assembly + homology-aware split ===
Final table rows (peptide x organism MIC pairs): 72587
Unique peptides: 15904
  rows from source=qmap_original: 62331
  rows from source=recovered: 10256
  unique peptides source=qmap_original: 13680
  unique peptides source=recovered: 2224

Split (pre leakage-filter reconciliation): 11114 train peptides / 3376 test peptides (threshold=0.6, target test_size=0.2, seed=42)
post_filtering=True removed 1414 peptides from train for having a >= 0.6 identity edge to a test peptide. These are reassigned to test (see leakage_filter_reassigned_to_test_peptide_ids in split_indices.json) rather than left unassigned.
Final split: 11114 train peptides / 4790 test peptides
  train rows: 51722, test rows: 20865
Reconciliation check: train + test == unique peptides? 11114 + 4790 = 15904 (unique peptides = 15904) -> OK
Train/test overlap check: 0 peptide_ids in both -> OK

Final dataset: /Users/lukajin/PycharmProjects/soamp/data/final_mic_regression_dataset.csv
Split indices: /Users/lukajin/PycharmProjects/soamp/data/split_indices.json
```

## Final dataset composition

- Total rows: 72587
- Unique peptides: 15904
- Rows by source: qmap_original=62331, recovered=10256
- Rows flagged has_noncanonical=True: 13710 (3063 unique peptides)
- mic_type breakdown: exact=38808, censored=18323, averaged=15456
- Unique organisms represented: 499

### Per-organism row counts (top 40, no filtering applied per the task's no-organism-filtering requirement -- shown here purely for visibility)

| Organism | Rows |
|---|---|
| Escherichia coli | 12856 |
| Staphylococcus aureus | 12047 |
| Pseudomonas aeruginosa | 8937 |
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
| Proteus vulgaris | 255 |
| Shigella dysenteriae | 236 |
| Serratia marcescens | 233 |
| Erwinia amylovora | 223 |
| Vibrio parahaemolyticus | 208 |
| Mycobacterium tuberculosis | 203 |
| Streptococcus agalactiae | 190 |
| Listeria innocua | 184 |
| Xanthomonas campestris | 180 |
| Aeromonas hydrophila | 167 |
| Xanthomonas vesicatoria | 167 |
| Cutibacterium acnes | 157 |
| Mycobacterium smegmatis | 155 |
| Shigella flexneri | 152 |
| Vibrio alginolyticus | 149 |
| Corynebacterium glutamicum | 146 |
| Lactococcus lactis | 144 |

## Manual-review flags raised during curation

- `data/qmap_included.csv` (QMAP's own pulled copy): one peptide (id=21052) has a leading space in its `sequence` field (`' LLLRRRRLL'`) -- flagged but not corrected in that file (see `scripts/02_pull_qmap.py`'s Step 1 diagnostic check). **This does not affect the final dataset**: `data/final_mic_regression_dataset.csv`'s `sequence` column is sourced entirely from this project's own independent DBAASP REST crawl (`.cache/dbaasp_raw.jsonl`, via `scripts/common/parse_dbaasp.py`), not from `qmap_included.csv`, and that independent source already has this peptide's sequence clean (`'LLLRRRRLL'`, no whitespace) at the origin. A 2026-08-07 QA follow-up traced this precisely; see `reports/qa_followup.md`. Nothing was silently corrected mid-pipeline -- the two copies of this peptide's sequence were simply always different, from two different data sources.
- Diff bucket b5 (`b5_excluded_unexplained`): peptides that pass this project's faithful re-implementation of QMAP's own inclusion filter but are absent from QMAP's published dataset. Most plausible explanation is DBAASP data changes between QMAP's January-2026 snapshot and this pull (curation edits, corrections, or newly added records); see the exact count and examples in the Step 3 log above.
- Diff bucket c (`c_in_qmap_not_in_raw_pull`): peptide IDs present in QMAP's dataset that this project's crawl did not return (network error after retries, or removed/renumbered on DBAASP's side since QMAP's snapshot); see examples in the Step 3 log above.
- Residue-code mapping (DBAASP -> p2smi) is a best-effort curated subset, not exhaustive -- see `scripts/common/residue_map.py` for the verified synonym table and its chemical-formula cross-checks. Any DBAASP non-canonical residue code not covered there falls through to the 'unconvertible: unknown residue code' path in Step 4 and is excluded, not approximated.
- Bond-based recovery is limited to disulfide (SS) and head-to-tail backbone amide (HT) cyclization, using p2smi's constraint system at the exact DBAASP-annotated bond positions (spot-checked against DBAASP's own native SMILES: generated structures matched native molecular weight to <0.001 Da in every spot-check performed during development). Side-chain lactam bridges (SCSC/SCNT/SCCT) and thioether/lactone cyclizations (lanthionine, sactionine, cyclic ester) are not auto-generated -- see `scripts/common/bond_map.py` for the chemistry rationale -- and are logged as unconvertible rather than guessed.
- N-/C-terminal modifications are only auto-applied for ACT (acetylation) and AMD (amidation), verified against known mass deltas (+42 Da / -1 Da). Lipidation (Cn acyl chains), PEGylation, and fluorophore/protecting-group termini are common in the DBAASP corpus but are NOT auto-generated by this pipeline; peptides with these modifications and no native DBAASP SMILES are excluded and logged as unconvertible, not approximated as unmodified peptides.
- `qmap.toolkit.train_test_split`'s `post_filtering=True` (the library default, used here) removes any train-assigned peptide with a similarity edge to a test peptide, to guarantee train/test independence -- but only ever removes from train, and (prior to 2026-08-07) the removed peptides were dropped from the output entirely with no reconciliation, so they silently ended up with no split assignment at all. A QA pass found 1,414 such peptides missing from `data/split_indices.json` on 2026-08-07; the previous version of this caveat claimed any split problem would be "called out explicitly," which was true for total failure (the `try`/`except` around the `train_test_split` call) but not for this partial silent-drop failure mode, which had gone unnoticed. **Fixed 2026-08-07**: `scripts/07_build_final_and_split.py` now explicitly reconciles the returned train/test ids against the full peptide set and reassigns any gap to the test set (not train -- that would reintroduce the leakage `post_filtering` exists to prevent), recorded separately as `leakage_filter_reassigned_to_test_peptide_ids` in `split_indices.json`, with an explicit reconciliation check printed in the Step 6 log every run from now on. See `reports/qa_followup.md` for the full root-cause writeup and before/after numbers.

## Files produced

- `data/qmap_included.csv` -- Step 1 output
- `data/dbaasp_raw_full.csv` -- Step 2 output
- `data/dbaasp_vs_qmap_diff.csv` -- Step 3 output
- `data/recovered_peptides.csv`, `data/unconvertible_peptides.csv` -- Step 4 output
- `data/step5_standardized_mic.csv` -- Step 5 output
- `data/final_mic_regression_dataset.csv`, `data/split_indices.json` -- Step 6 output
- `.cache/dbaasp_raw.jsonl` -- raw native-format DBAASP API responses (full fidelity, pre-flattening)
