from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

import numpy as np

# -------------------------------------------------
# Build PAINS filter once (heavy operation)
# -------------------------------------------------
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog(pains_params)

# -------------------------------------------------
# SA Score import (RDKit contrib)
# -------------------------------------------------
try:
    from rdkit.Chem.QSAR import SA_Score

    def compute_sa(mol):
        return SA_Score.sascorer.calculateScore(mol)

except Exception:
    def compute_sa(mol):
        return None   # fallback if unavailable


# -------------------------------------------------
# Compute drug-likeness properties
# -------------------------------------------------
def compute_drug_properties(mol):
    """Return all major medicinal chemistry properties for a molecule."""
    props = {}

    props["MW"] = Descriptors.MolWt(mol)
    props["logP"] = Crippen.MolLogP(mol)
    props["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
    props["HBD"] = Lipinski.NumHDonors(mol)
    props["HBA"] = Lipinski.NumHAcceptors(mol)
    props["RotatableBonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    props["Rings"] = rdMolDescriptors.CalcNumRings(mol)
    props["QED"] = float(QED.qed(mol))

    # Synthetic Accessibility
    props["SA"] = compute_sa(mol)

    # Lipinski rule-of-5 violations
    violations = 0
    if props["MW"] > 500: violations += 1
    if props["logP"] > 5: violations += 1
    if props["HBD"] > 5: violations += 1
    if props["HBA"] > 10: violations += 1
    props["LipinskiViolations"] = violations

    # PAINS detection
    pains = pains_catalog.GetMatches(mol)
    props["PAINSAlerts"] = [m.GetDescription() for m in pains]

    return props


# -------------------------------------------------
# Composite analog scoring (same logic as Streamlit)
# -------------------------------------------------
def compute_analog_score(tox_probs, props):
    """
    Returns a composite drug-likeness score.
    Higher score = safer, more drug-like, synthesizable.
    """

    mean_tox = float(np.mean(tox_probs))
    qed = props["QED"]
    sa = props["SA"]

    # SA score normalization: 1 (easy) -> score 1, 10 (difficult) -> score 0
    if sa is None:
        sa_term = 0.5
    else:
        sa_norm = min(max((sa - 1.0) / 9.0, 0.0), 1.0)
        sa_term = 1.0 - sa_norm

    lip_penalty = max(0.0, 1.0 - props["LipinskiViolations"] / 2.0)
    pains_penalty = 0.5 if props["PAINSAlerts"] else 0.0

    score = (
        0.4 * qed +                 # drug-likeness
        0.4 * (1 - mean_tox) +      # toxicity
        0.15 * sa_term +            # synthetic accessibility
        0.05 * lip_penalty          # lipinski friendliness
    ) - pains_penalty               # penalize PAINS

    return float(score)


# -------------------------------------------------
# Pack analog output
# -------------------------------------------------
def package_analog(smiles, mol, tox_probs):
    props = compute_drug_properties(mol)
    score = compute_analog_score(tox_probs, props)

    return {
        "smiles": smiles,
        "score": score,
        "mean_toxicity": float(np.mean(tox_probs)),
        "QED": props["QED"],
        "SA": props["SA"],
        "LipinskiViolations": props["LipinskiViolations"],
        "PAINSAlerts": props["PAINSAlerts"],
    }
