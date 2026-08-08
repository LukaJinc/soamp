"""
Organism domain classification (Bacteria/Archaea/Virus/Fungus/Animal/Plant/Other/Unknown)
and NCBI taxon id lookup, via ete4's NCBITaxa -- matches QMAP's own classify_genus
(data/dbaasp/utils.py) methodology exactly, applied to the genus (first word of the
species binomial), extended here to also return the taxon id for the final schema's
ncbi_taxon_id_if_available column.

DBAASP target species names often include strain info (e.g. "Staphylococcus aureus
ATCC 6538P"); QMAP's own `Target.specie` property keeps only the first two
whitespace-separated words (genus + species epithet) before classification, which
we replicate here.
"""
from functools import lru_cache
from ete4 import NCBITaxa

_ncbi = NCBITaxa()


def species_binomial(target_species_name: str) -> str:
    """First two words of the target species name (genus + species epithet),
    matching QMAP's Target.specie property."""
    if not target_species_name:
        return target_species_name
    return " ".join(target_species_name.split(" ")[:2])


@lru_cache(maxsize=None)
def classify_genus(genus: str):
    """Returns (domain: str, taxon_id: int | None). domain matches QMAP's categories."""
    try:
        translation = _ncbi.get_name_translator([genus])
        if genus not in translation:
            return "Unknown", None
        taxid = translation[genus][0]
        lineage = _ncbi.get_lineage(taxid)
        names = set(_ncbi.get_taxid_translator(lineage).values())

        if "Bacteria" in names:
            domain = "Bacteria"
        elif "Archaea" in names:
            domain = "Archaea"
        elif "Viruses" in names:
            domain = "Virus"
        elif "Fungi" in names:
            domain = "Fungus"
        elif "Metazoa" in names:
            domain = "Animal"
        elif "Viridiplantae" in names:
            domain = "Plant"
        else:
            domain = "Other"
        return domain, taxid
    except KeyError:
        return "Unknown", None


@lru_cache(maxsize=None)
def classify_species(target_species_name: str):
    """
    Returns (domain: str, taxon_id: int | None, binomial: str).
    taxon_id here is looked up for the full binomial (genus+species) when possible,
    falling back to the genus-level id (matching classify_genus) if the binomial
    itself isn't a recognized NCBI name (e.g. informal/typo'd species names).
    """
    binomial = species_binomial(target_species_name)
    if not binomial:
        return "Unknown", None, binomial
    genus = binomial.split(" ")[0]
    domain, genus_taxid = classify_genus(genus)

    taxid = genus_taxid
    try:
        translation = _ncbi.get_name_translator([binomial])
        if binomial in translation:
            taxid = translation[binomial][0]
    except KeyError:
        pass

    return domain, taxid, binomial
