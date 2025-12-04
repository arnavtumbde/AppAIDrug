import torch


def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        atom.GetImplicitValence(),
        int(atom.GetHybridization()),
    ], dtype=torch.float)


def bond_features(bond):
    return torch.tensor([
        bond.GetBondTypeAsDouble(),
        int(bond.GetIsConjugated()),
        int(bond.GetStereo()),
        int(bond.IsInRing()),
    ], dtype=torch.float)
