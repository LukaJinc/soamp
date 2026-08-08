"""
MIC value parsing and unit standardization to uM.

parse_activity() is ported near-verbatim from QMAP's target_base.py
(TargetBase.parse_activity, github.com/anthol42/QMAP) -- it is QMAP's own published
midpoint-for-ranges / explicit-bound-for-censored parser for DBAASP's free-text
concentration strings, reused here for methodological consistency with the paper
this dataset follows. Minor change: kept as a standalone function (no class state).

Unlike QMAP -- which computes molecular weight from a hardcoded 20-canonical-AA
mass table and only falls back to RDKit-on-SMILES for sequences containing "X" --
this project computes molecular weight from the full peptide SMILES via RDKit
whenever *any* SMILES (native DBAASP or p2smi-generated) is available, per the
task's explicit "do not assume unit conversion factors -- compute via molecular
weight from structure" requirement. This is applied uniformly to canonical and
non-canonical peptides alike, not just as a fallback.
"""
import re
import math
from rdkit import Chem
from rdkit.Chem import Descriptors


def precision(value: float, precision: int = 3) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return value
    return float('%s' % float(f'%.{precision}g' % value))


def parse_activity(activity_string: str):
    """
    Parse a DBAASP free-text concentration string into (min, max) bounds, applying
    midpoint-for-ranges and explicit-bound-for-censored-values, matching QMAP's method.

    Returns (min_val, max_val); both float('nan') if unparseable/missing.
    Ported from QMAP TargetBase.parse_activity (data/dbaasp/dbaasp/target_base.py).
    """
    if activity_string is None:
        return float('nan'), float('nan')

    raw_activity_string = activity_string
    if activity_string == '4.5.5':
        activity_string = '4.5-5'
    elif activity_string == "16=128":
        activity_string = '16-128'

    activity_string = (activity_string.replace('–', '-').replace('~', '')
                        .replace('+', '±').replace("E6", "").replace("E", ""))
    if '±0.0' in activity_string:
        activity_string = activity_string.replace('±0.0', '')

    if '-' in activity_string:
        activity_string = (activity_string.replace(">=", "").replace("=<", "")
                            .replace("=", "").replace(" ", ""))
        if activity_string == '-':
            return float('nan'), float('nan')
        minAct = activity_string.split('-')[0]
        maxAct = activity_string.split('-')[1]
        if minAct.startswith('<'):
            minAct = 0.
        else:
            minAct = float(minAct.replace(">", ""))
        if maxAct.startswith('>'):
            maxAct = float('inf')
        else:
            maxAct = float(maxAct)
        return minAct, maxAct
    elif '>=' in activity_string or '≥' in activity_string or '>' in activity_string:
        activity_string = re.sub(r'±.*', '', activity_string.replace('>=', '')
                                  .replace('≥', '').replace(">", '')).replace(" ", "")
        return float(activity_string), float('inf')
    elif '<=' in activity_string or '<' in activity_string:
        activity_string = re.sub(r'±.*', '', activity_string.replace('<=', '').replace("<", ''))
        return 0., float(activity_string)
    elif activity_string == 'NA':
        return float('nan'), float('nan')
    elif '±' in activity_string:
        value, error = activity_string.split('±')
        error = error.replace(",", ".")
        min_act, max_act = float(value) - float(error), float(value) + float(error)
        if min_act < 0:
            min_act = 0.
        return min_act, max_act
    elif activity_string == '':
        return float('nan'), float('nan')
    elif "up to " in activity_string.lower():
        activity_string = activity_string.replace('up to ', '')
        return 0., float(activity_string)
    else:
        try:
            activity_string = activity_string.replace(' ', '').replace(',', '.')
            return float(activity_string), float(activity_string)
        except ValueError:
            return float('nan'), float('nan')


def classify_censoring(min_val: float, max_val: float, raw_string: str) -> str:
    """
    'exact'    : min == max (a single reported value or a symmetric +/- resolved to bounds)
    'censored' : one-sided bound (<X or >X, i.e. min==0 or max==inf) -- true left/right censoring
    'ranged'   : a genuine two-sided reported range (min != max, both finite, not 0/inf)
    'missing'  : unparseable / NA
    """
    if isinstance(min_val, float) and math.isnan(min_val):
        return 'missing'
    if max_val == float('inf') or min_val == 0.:
        return 'censored'
    if min_val == max_val:
        return 'exact'
    return 'ranged'


def compute_smiles_weight(smiles: str):
    """Molecular weight (g/mol) from a SMILES string via RDKit. None if invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


def ug_ml_to_uM(min_val: float, max_val: float, molar_mass: float):
    """
    uM = 1000 * (ug/mL) / (g/mol).
    Dimensional check: ug/mL == mg/L; mg/L / (mg/mmol) = mmol/L = 1000 * umol/L.
    Matches QMAP's TargetBase.convert_to_micromolar formula.
    """
    if molar_mass is None or (isinstance(molar_mass, float) and math.isnan(molar_mass)) or molar_mass <= 0:
        return float('nan'), float('nan')
    mi = precision(1e3 * min_val / molar_mass) if not math.isnan(min_val) else float('nan')
    ma = precision(1e3 * max_val / molar_mass) if not math.isnan(max_val) else float('nan')
    return mi, ma
