from fastapi import FastAPI
from pydantic import BaseModel

from app.model_loader import load_model, predict_smiles, explain_molecule
from app.analogs import generate_selfies_analogs


app = FastAPI(title="Tox21 GNN Backend")

model = load_model()   # Load once at startup


class SMILESInput(BaseModel):
    smiles: str
    task: int = 0
    n_analogs: int = 10


@app.get("/")
def root():
    return {"message": "Tox21 GNN Backend is running!"}


@app.post("/predict")
def predict_endpoint(data: SMILESInput):
    mol, probs = predict_smiles(data.smiles, model)
    return {
        "smiles": data.smiles,
        "probabilities": probs.tolist()
    }


@app.post("/explain")
def explain_endpoint(data: SMILESInput):
    mol, importance = explain_molecule(data.smiles, model, task_id=data.task)
    return {
        "smiles": data.smiles,
        "importance": importance.tolist()
    }


@app.post("/analogs")
def analogs_endpoint(data: SMILESInput):
    analog_list = generate_selfies_analogs(
        seed_smiles=data.smiles,
        model=model,
        n_candidates=data.n_analogs
    )
    return {
        "input": data.smiles,
        "n_analogs": len(analog_list),
        "analogs": analog_list
    }
