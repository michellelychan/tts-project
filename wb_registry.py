import wandb
import os
from pathlib import Path

# Initialize a W&B run
wandb.init(
    project="speecht5-tts-finetuning",
    name="model-registry-upload",
    job_type="upload"
)

# Path to your model
model_path = "speecht5_finetuned_voxpopuli_nl"

# Create a W&B Artifact
model_artifact = wandb.Artifact(
    name="speecht5_finetuned_voxpopuli_nl",
    type="model",
    description="Fine-tuned SpeechT5 model on VoxPopuli Dutch dataset"
)

# Add the model directory to the artifact
model_artifact.add_dir(model_path)

# Log the artifact to W&B
wandb.log_artifact(model_artifact)

# Optional: Add metadata about your model
model_artifact.metadata = {
    "base_model": "microsoft/speecht5_tts",
    "dataset": "facebook/voxpopuli",
    "language": "nl",
    "fine_tuning_steps": 4000,
    "learning_rate": 1e-5,
}

print(f"Model artifact '{model_artifact.name}' uploaded to W&B")

# Finish the W&B run
wandb.finish()