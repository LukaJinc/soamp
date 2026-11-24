# MIC recomputation report (dev v2, seven strains)

Input rows in scope: 30568; kept: 30565; dropped: 3.

## Rule
ug/ml rows: the concentration as written. uM rows: uM x RDKit MW / 1000. NA rows: the DBAASP sentinel 1,111,111. Anything else is dropped, not guessed.

## Concentration kinds (all kept rows)
| conc_kind | rows |
|---|---|
| num | 21645 |
| gt | 5987 |
| NA | 1113 |
| pm | 961 |
| range | 753 |
| lt | 99 |
| messy | 5 |
| other | 2 |

## Recomputed vs DBAASP `activity` (rows with a value; NA rows excluded)
Ratio = activity / recomputed. uM rows differ by a small MW offset (DBAASP's weight vs RDKit's); ug/ml rows should match.
|  | rows | within_0p1pct | within_1pct | median_ratio |
|---|---|---|---|---|
| ('gt', 'uM') | 3230 | 41.3 | 91.95 | 1.0012 |
| ('gt', 'ug/ml') | 2757 | 100.0 | 100.0 | 1.0 |
| ('lt', 'uM') | 56 | 21.43 | 83.93 | 1.0018 |
| ('lt', 'ug/ml') | 43 | 100.0 | 100.0 | 1.0 |
| ('messy', 'uM') | 3 | 66.67 | 100.0 | 1.0 |
| ('messy', 'ug/ml') | 2 | 100.0 | 100.0 | 1.0 |
| ('num', 'uM') | 11158 | 42.44 | 96.05 | 1.0013 |
| ('num', 'ug/ml') | 10487 | 99.98 | 99.98 | 1.0 |
| ('other', 'uM') | 2 | 50.0 | 50.0 | 0.9713 |
| ('pm', 'uM') | 632 | 44.78 | 94.15 | 1.0012 |
| ('pm', 'ug/ml') | 329 | 99.09 | 99.39 | 1.0 |
| ('range', 'uM') | 337 | 29.08 | 87.54 | 1.0015 |
| ('range', 'ug/ml') | 416 | 100.0 | 100.0 | 1.0 |

## NA rows
1113 rows, all set to 1,111,111; measures: {'-': 664, '': 449}; with a unit: 0.

## Dropped rows (3)
|  | peptide_id | measure | concentration | unit | recompute_status |
|---|---|---|---|---|---|
| 0 | 9794 | LC | 2.07 | nan | dropped: no unit |
| 1 | 16406 | MIC | 8 | nan | dropped: no unit |
| 2 | 18648 | MIC |  | µM | dropped: empty concentration |

## Bogus molecular weights
Peptides with at least one uM row whose DBAASP activity implies a weight below 100 g/mol: 8 (list in `dbaasp_bogus_mw_peptides.csv`).
|  | peptide_id | dbaaspId | name | rows | implied_mw | rdkit_mw |
|---|---|---|---|---|---|---|
| 0 | 1710 | DBAASPS_1710 | (Nspe-NLys)2-NLys2 -(Nspe-NLys)2-NLys-Nspe2-NLys-Nspe | 1 | 17.03 | 2170.86 |
| 1 | 6745 | DBAASPS_6745 | (Nspe-NLys-Nspe-NLys-NLys-Nspe)2 | 1 | 17.03 | 1753.3 |
| 2 | 20978 | DBAASPS_20978 | Meropenem | 7 | 45.09 | 383.47 |
| 3 | 22876 | DBAASPS_22876 | Globomycin 51 | 1 | 0 | 750.04 |
| 4 | 24091 | DBAASPS_24091 | NLys-Nspe-Nspe-(NLys-Npm-Npm)2-NLys-Npm-Nspe | 4 | 17.03 | 1749.23 |
| 5 | 24340 | DBAASPS_24340 | Orn-tBu-βAc6c | 2 | 1.01 | 430.64 |
| 6 | 24343 | DBAASPS_24343 | Orn-tBu-βAc6c | 2 | 0 | 612.94 |
| 7 | 24346 | DBAASPS_24346 | Orn-tBu-βAc6c | 2 | 0 | 641.99 |
