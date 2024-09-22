import os

models_path = "transformers/src/transformers/models"

modelnames = [
    d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))
]


ModelFiles = dict[str, str]
models: dict[str, ModelFiles] = {}

for m in modelnames:
    models[m] = {}
    for p in os.listdir(os.path.join(models_path, m)):
        filepath = os.path.join(models_path, m, p)
        if os.path.isdir(filepath):
            continue
        with open(filepath) as f:
            models[m][p] = f.read()
