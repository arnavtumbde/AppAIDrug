import random
from rdkit import Chem
import selfies as sf


def generate_selfies_analogs(seed_smiles, model=None, n_candidates=20, max_mut=2):
    seed_sf = sf.encoder(seed_smiles)
    tokens = list(sf.split_selfies(seed_sf))

    alphabet = list(sf.get_semantic_robust_alphabet())

    results = set()
    attempts = 0

    while len(results) < n_candidates and attempts < n_candidates * 20:
        attempts += 1
        t = tokens.copy()

        for _ in range(random.randint(1, max_mut)):
            idx = random.randrange(len(t))
            t[idx] = random.choice(alphabet)

        smi = sf.decoder("".join(t))
        if not smi:
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        results.add(Chem.MolToSmiles(mol))

    return list(results)


# --- NEW: Wrapper required by FastAPI (main.py) ---
def generate_analogs(seed_smiles, model=None, n=10):
    """
    FastAPI endpoint imports this.
    Internally uses the SELFIES generator above.
    """
    return generate_selfies_analogs(
        seed_smiles,
        model=model,
        n_candidates=n,
        max_mut=2
    )
