import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from rdkit import Chem

from app.features import atom_features, bond_features


tox21_tasks = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]


class GIN_edge(torch.nn.Module):
    def __init__(self, num_tasks=12):
        super().__init__()

        from torch.nn import Linear, ReLU, Sequential
        from torch_geometric.nn import GINEConv, global_mean_pool

        nn1 = Sequential(
            Linear(7, 64),
            ReLU(),
            Linear(64, 64),
        )
        nn2 = Sequential(
            Linear(64, 64),
            ReLU(),
            Linear(64, 64),
        )

        self.conv1 = GINEConv(nn1, edge_dim=4)
        self.conv2 = GINEConv(nn2, edge_dim=4)
        self.fc = Linear(64, num_tasks)
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.pool(x, batch)
        return self.fc(x)


def load_model(device="cpu"):
    model = GIN_edge().to(device)
    state = torch.load("app/tox21_gnn.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def mol_to_graph(mol):
    if mol is None:
        return None

    import torch
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])

    edge_index = []
    edge_attr = []

    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        f = bond_features(b)

        edge_index.append([i, j])
        edge_attr.append(f)
        edge_index.append([j, i])
        edge_attr.append(f)

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t(),
        edge_attr=torch.stack(edge_attr)
    )


def predict_smiles(smiles, model, device="cpu"):
    mol = Chem.MolFromSmiles(smiles)
    graph = mol_to_graph(mol)
    batch = Batch.from_data_list([graph]).to(device)

    with torch.no_grad():
        out = model(batch)
        probs = torch.sigmoid(out)[0].cpu().numpy()

    return mol, probs


def explain_molecule(smiles, model, task_id=0, device="cpu"):
    mol = Chem.MolFromSmiles(smiles)
    graph = mol_to_graph(mol)

    batch = Batch.from_data_list([graph]).to(device)
    batch.x.requires_grad_(True)

    output = model(batch)[0, task_id]
    model.zero_grad()
    output.backward()

    grad = batch.x.grad.abs().sum(dim=1).cpu().numpy()

    grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    return mol, grad_norm
