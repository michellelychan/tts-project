import os
from datetime import datetime
from transformers.integrations import WandbCallback
import wandb
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.tracking.client import MlflowClient
import sys


class DualLoggingCallback(WandbCallback):
    """
    Custom callback that logs to both W&B and MLflow simultaneously
    """
    def __init__(self, model, processor, dataset, vocoder, mlflow_run_name=None, mlflow_tracking_uri=None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.vocoder = vocoder
        
        # Setup MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Start MLflow run - using the same run name as W&B for consistency
        self.mlflow_run_name = mlflow_run_name or f"speecht5_nl_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=self.mlflow_run_name)
        
        # Log MLflow params to match W&B config
        for key, value in wandb.config.items():
            mlflow.log_param(key, value)
        
        # Initialize MLflow client for artifact logging
        self.client = MlflowClient()
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log metrics to both W&B and MLflow"""
        super().on_log(args, state, control, model, logs, **kwargs)
        
        if logs:
            # Filter out non-numeric values that MLflow can't handle
            numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            
            # Log step metrics to MLflow
            mlflow.log_metrics(numeric_logs, step=state.global_step)
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Log evaluation metrics and audio samples"""
        super().on_evaluate(args, state, control, model, **kwargs)
        
        # Only log audio samples every 1000 steps
        if state.global_step % 1000 == 0:
            # Use the model passed in for audio generation
            current_model = model if model is not None else self.model
            
            # Log audio samples to W&B
            self._log_audio_samples(
                model=current_model,
                processor=self.processor,
                dataset=self.dataset,
                vocoder=self.vocoder,
                num_samples=3,
                step=state.global_step
            )
    
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Handle end of training - log final metrics and models"""
        super().on_train_end(args, state, control, model, tokenizer, **kwargs)
        
        # Get final model
        final_model = model if model is not None else self.model
        
        # Log final audio samples
        self._log_audio_samples(
            model=final_model,
            processor=self.processor,
            dataset=self.dataset,
            vocoder=self.vocoder,
            num_samples=5,
            step=state.global_step,
            is_final=True
        )
        
        # Create W&B artifact for the model
        model_artifact_name = f"speecht5_finetuned_nl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_artifact = wandb.Artifact(
            name=model_artifact_name,
            type="model",
            description="Fine-tuned SpeechT5 model on VoxPopuli Dutch dataset"
        )
        
        # Add the model directory to W&B artifact
        model_artifact.add_dir(args.output_dir)
        
        # Add metadata to W&B artifact
        model_artifact.metadata = {
            "base_model": "microsoft/speecht5_tts",
            "dataset": "facebook/voxpopuli",
            "language": "nl",
            "fine_tuning_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "final_loss": state.log_history[-1].get("eval_loss", "N/A"),
            "best_model_checkpoint": state.best_model_checkpoint
        }
        
        # Log and register the W&B artifact
        wandb.log_artifact(model_artifact)
        print(f"✅ Model artifact '{model_artifact.name}' uploaded to W&B model registry")
        
        # Log the model to MLflow model registry
        mlflow.pytorch.log_model(
            final_model, 
            "model",
            registered_model_name=model_artifact_name
        )
        
        # Log the processor to MLflow
        processor_output_path = os.path.join(args.output_dir, "processor")
        os.makedirs(processor_output_path, exist_ok=True)
        self.processor.save_pretrained(processor_output_path)
        
        # Log processor as an MLflow artifact
        mlflow.log_artifact(processor_output_path, "processor")
        
        # End the MLflow run
        mlflow.end_run()
        print(f"✅ Model and processor logged to MLflow model registry")
    
    def _log_audio_samples(self, model, processor, dataset, vocoder, num_samples=3, step=0, is_final=False):
        """Generate and log audio samples to both W&B and MLflow"""
        print(f"Generating audio samples for logging (step {step})...")
        
        # Get texts from test set for visualization
        examples = [dataset["test"][i] for i in range(min(num_samples, len(dataset["test"])))]
        
        # Move vocoder to the same device as the model
        device = model.device
        vocoder = vocoder.to(device)
        
        # Prepare audio sample directory for MLflow
        audio_dir = f"audio_samples_step_{step}"
        os.makedirs(audio_dir, exist_ok=True)
        
        # Get original texts if available
        original_texts = []
        for i, example_idx in enumerate(range(min(num_samples, len(dataset["test"])))):
            try:
                original_text = dataset["test"][example_idx]["normalized_text"]
                original_texts.append(original_text)
            except:
                original_texts.append(f"Sample {i+1}")
        
        for i, example in enumerate(examples):
            # Setup inputs
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0).to(device)
            
            # Generate speech
            with torch.no_grad():
                speech = model.generate_speech(
                    input_ids, 
                    speaker_embeddings, 
                    vocoder=vocoder
                )
            
            # Convert to numpy for logging
            audio_array = speech.cpu().numpy()
            
            # Create caption
            caption = original_texts[i] if i < len(original_texts) else f"Generated speech sample {i+1}"
            prefix = "final_" if is_final else ""
            
            # Log to W&B
            wandb.log({
                f"{prefix}audio_sample_{i+1}_step_{step}": wandb.Audio(
                    audio_array, 
                    sample_rate=16000,
                    caption=caption
                )
            })
            
            # Save audio file for MLflow
            audio_path = os.path.join(audio_dir, f"{prefix}sample_{i+1}.wav")
            self._save_wav(audio_array, audio_path, sample_rate=16000)
            
            # Create text file with caption
            caption_path = os.path.join(audio_dir, f"{prefix}sample_{i+1}_text.txt")
            with open(caption_path, "w") as f:
                f.write(caption)
            
            print(f"Logged audio sample {i+1} at step {step}")
        
        # Log audio directory to MLflow
        mlflow.log_artifacts(audio_dir, f"audio_samples/step_{step}")
        
        # Clean up local directory
        import shutil
        shutil.rmtree(audio_dir)
    
    def _save_wav(self, audio_array, file_path, sample_rate=16000):
        """Save audio array as WAV file"""
        import scipy.io.wavfile as wav
        
        # Ensure audio is properly scaled for 16-bit WAV
        if audio_array.max() <= 1.0 and audio_array.min() >= -1.0:
            audio_array = np.int16(audio_array * 32767)
        
        wav.write(file_path, sample_rate, audio_array)


def setup_dual_logging(model, processor, dataset, vocoder, mlflow_tracking_uri="http://localhost:5000"):
    """
    Helper function to setup dual logging to both W&B and MLflow
    
    Args:
        model: The SpeechT5ForTextToSpeech model
        processor: The SpeechT5Processor
        dataset: The dataset dictionary containing 'test' split
        vocoder: SpeechT5HifiGan vocoder
        mlflow_tracking_uri: URI for MLflow tracking server
    
    Returns:
        DualLoggingCallback instance
    """
    # Generate a unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"speecht5_nl_run_{timestamp}"
    
    # Create the dual logging callback
    return DualLoggingCallback(
        model=model,
        processor=processor,
        dataset=dataset,
        vocoder=vocoder,
        mlflow_run_name=run_name,
        mlflow_tracking_uri=mlflow_tracking_uri
    )


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dual_logging import DualLoggingCallback, setup_dual_logging

# Configure your MLflow tracking URI - replace with your server's address
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Your MLflow server address

# Then replace your existing training section with this modified version:

"""
Training the model with dual logging to W&B and MLflow
"""

from transformers import SpeechT5ForTextToSpeech
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from datetime import datetime
import os

# Generate a unique run name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"speecht5_nl_run_{timestamp}"
output_dir = f"speecht5_finetuned_voxpopuli_nl_{timestamp}"

# Initialize wandb with more metadata
wandb.init(
    project="speecht5-tts-finetuning",
    name=run_name,
    config={
        "model_checkpoint": "microsoft/speecht5_tts",
        "dataset": "facebook/voxpopuli",
        "language": "nl",
        "batch_size": 4,
        "learning_rate": 1e-5,
        "warmup_steps": 500,
        "max_steps": 4000,
        "output_dir": output_dir,
        "timestamp": timestamp,
    }
)

model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
model.config.use_cache = False

# Load the vocoder for audio sample generation
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

data_collator = TTSDataCollatorWithPadding(processor=processor)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    run_name=run_name,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard", "wandb"],  # Keep both for compatibility
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

# Setup dual logging callback
dual_logging_callback = setup_dual_logging(
    model=model,
    processor=processor, 
    dataset=dataset,
    vocoder=vocoder,
    mlflow_tracking_uri=MLFLOW_TRACKING_URI
)

# Use the trainer with our dual logging callback
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    callbacks=[dual_logging_callback],  # Use our dual logging callback
)

# Generate initial audio samples before training
# This function now logs to both W&B and MLflow
dual_logging_callback._log_audio_samples(model, processor, dataset, vocoder, num_samples=2, step=0)

# Train the model
trainer.train()

# Save the processor
processor_output_path = os.path.join(output_dir, "processor")
os.makedirs(processor_output_path, exist_ok=True)
processor.save_pretrained(processor_output_path)

# Push to Hugging Face Hub
trainer.push_to_hub()

# Log a final message about where to find the model
print(f"Training complete! Model saved to: {output_dir}")
print(f"Model uploaded to W&B model registry as: speecht5_finetuned_nl_{timestamp}")
print(f"Model registered in MLflow model registry as: speecht5_finetuned_nl_{timestamp}")
print(f"To use this model, load it from W&B/MLflow or from the local directory: {output_dir}")

# Close wandb run when done
wandb.finish()

