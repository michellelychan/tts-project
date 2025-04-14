from transformers import pipeline
import scipy
from datasets import load_dataset, Audio
from transformers import SpeechT5Processor
from collections import defaultdict
import matplotlib.pyplot as plt
import mlflow
import os
import torch
from datetime import datetime

# Create necessary directories
os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs/images", exist_ok=True)
os.makedirs("../outputs/audio", exist_ok=True)

""" 
Preprocess the data 
""" 

# Initialize MLflow at the beginning
# Set tracking URI to local MLflow server (running on same machine)
mlflow.set_tracking_uri("http://localhost:5000")  # Update this if your server has a different port

# Configure MLflow to batch logged metrics
# This greatly improves performance by reducing network calls
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'false'  # Disable system metrics
os.environ['MLFLOW_BATCH_LOGGING_MIN_FLUSH_INTERVAL_SECONDS'] = '20'  # Only flush every 20 seconds

experiment_name = "speecht5-tts-finetuning"

# Create or get the experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# Start a new MLflow run with a descriptive name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"speecht5_nl_run_{timestamp}"
run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)

# Log all parameters at once to reduce API calls
mlflow.log_params({
    "model_checkpoint": "microsoft/speecht5_tts",
    "dataset": "facebook/voxpopuli",
    "language": "nl",
    "batch_size": 4,
    "learning_rate": 1e-5,
    "warmup_steps": 500,
    "max_steps": 4000,
    "training_timestamp": timestamp,
})

dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
print(len(dataset))
# Log the initial dataset size
mlflow.log_metric("initial_dataset_size", len(dataset))

# synthesiser = pipeline("text-to-speech", "suno/bark-small")
# speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"do_sample": True})
# scipy.io.wavfile.write("../outputs/audio/bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
tokenizer = processor.tokenizer

def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
unsupported_chars = dataset_vocab - tokenizer_vocab
print(f"dataset_vocab - tokenizer_vocab {unsupported_chars}")

# Log unsupported characters as a text file
# But defer actual logging to reduce overhead during startup
with open("../outputs/unsupported_chars.txt", "w") as f:
    for char in unsupported_chars:
        f.write(f"{char}\n")
# We'll log this artifact later with other artifacts in batches

replacements = [
    ("à", "a"),
    ("ç", "c"),
    ("è", "e"),
    ("ë", "e"),
    ("í", "i"),
    ("ï", "i"),
    ("ö", "o"),
    ("ü", "u"),
]

def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs

dataset = dataset.map(cleanup_text)

speaker_counts = defaultdict(int)

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1
    
plt.figure()
hist = plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Examples")
plt.savefig('../outputs/images/speaker_examples_histogram.png')

# We'll log this histogram later with other artifacts in batches

def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400

dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
num_speakers = len(set(dataset["speaker_id"]))
len_dataset = len(dataset)

print(f"no. of speakers: {num_speakers}")
print(f"length of dataset: {len_dataset}")

# Log dataset statistics after filtering
mlflow.log_metrics({
    "num_speakers": num_speakers,
    "filtered_dataset_size": len_dataset
})

""" 
Clean the data 
""" 
import os
import torch
from speechbrain.inference.classifiers import EncoderClassifier
from accelerate.test_utils.testing import get_backend


spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

processed_example = prepare_dataset(dataset[0])
list_keys = list(processed_example.keys())
print(f"list keys: {list_keys}")
print(f"speaker embedding shape: {processed_example['speaker_embeddings'].shape}")

plt.figure()
plt.imshow(processed_example["labels"].T)
plt.savefig("../outputs/images/log-mel_spectrogram_example.png")

# We'll log this spectrogram later with other artifacts in batches

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200

dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
print(f"length of filtered dataset: {len(dataset)}")

# Log dataset size after length filtering
mlflow.log_metric("length_filtered_dataset_size", len(dataset))

dataset = dataset.train_test_split(test_size=0.1)
mlflow.log_metrics({
    "train_set_size": len(dataset["train"]),
    "test_set_size": len(dataset["test"])
})

"""
Data collator
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
            target_lengths = target_lengths.new(
                [length - length % model.config.reduction_factor for length in target_lengths]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

"""
Audio Generation Function
"""
def generate_audio_samples(model, processor, dataset, vocoder, num_samples=3):
    """
    Generate audio samples from the model and save them locally
    
    Args:
        model: The SpeechT5ForTextToSpeech model
        processor: The SpeechT5Processor
        dataset: The dataset dictionary containing 'test' split
        vocoder: SpeechT5HifiGan vocoder for converting spectrograms to audio
        num_samples: Number of samples to generate
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Generating audio samples...")
    
    # Get texts from test set for visualization
    examples = [dataset["test"][i] for i in range(min(num_samples, len(dataset["test"])))]
    
    # Move vocoder to the same device as model
    device = model.device
    vocoder = vocoder.to(device)
    
    # Store original texts for captions
    original_texts = []
    for i, example_idx in enumerate(range(min(num_samples, len(dataset["test"])))):
        try:
            original_text = dataset["test"].features["normalized_text"][example_idx]
            original_texts.append(original_text)
        except:
            original_texts.append(f"Sample {i+1}")
    
    audio_files = []
    artifact_paths = []
    
    # Create a temporary directory for batching artifacts
    tmp_dir = f"../outputs/audio/tmp_{timestamp}"
    os.makedirs(tmp_dir, exist_ok=True)
    
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
        
        # Convert to numpy for saving
        audio_array = speech.cpu().numpy()
        
        # Save audio locally
        audio_path = f"{tmp_dir}/sample_{i+1}.wav"
        scipy.io.wavfile.write(audio_path, rate=16000, data=audio_array)
        audio_files.append(audio_path)
        artifact_paths.append(audio_path)
        
        # Add caption info in a text file
        caption_path = f"{tmp_dir}/sample_{i+1}_caption.txt"
        caption = original_texts[i] if i < len(original_texts) else f"Generated speech sample {i+1}"
        with open(caption_path, "w") as f:
            f.write(caption)
        artifact_paths.append(caption_path)
        
        print(f"Generated audio sample {i+1}")
    
    # Log all artifacts at once to reduce overhead
    mlflow.log_artifacts(tmp_dir, artifact_path=f"audio_samples_{timestamp}")
    print(f"Logged {len(audio_files)} audio samples to MLflow")
    
    return audio_files

"""
Training the model with MLflow model registry
"""

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import os
import time
from datetime import datetime

# Now batch log all the artifacts we've generated so far
artifact_dir = "../outputs"
print("Logging initial artifacts to MLflow...")
mlflow.log_artifacts(artifact_dir, artifact_path="preprocessing_artifacts")

# Generate a unique run name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"speecht5_nl_run_{timestamp}"
output_dir = f"../models/speecht5_finetuned_voxpopuli_nl_{timestamp}"

# Log more metadata
mlflow.set_tag("run_name", run_name)
mlflow.set_tag("output_dir", output_dir)
mlflow.set_tag("timestamp", timestamp)

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
    # Increase steps between evaluations to reduce overhead
    save_steps=2000,  # Save less frequently 
    eval_steps=1000,  
    # Log less frequently for better performance
    logging_steps=100,  # Log every 100 steps instead of 25
    report_to=["tensorboard"],  # Removed wandb, kept tensorboard
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

# Custom callback for MLflow logging and audio sample generation
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import TrainerCallback

class MLflowAndAudioCallback(TrainerCallback):
    def __init__(self, model, processor, dataset, vocoder):
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.vocoder = vocoder
        self.metrics_buffer = {}
        self.last_log_time = time.time()
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Log metrics to MLflow - but batch them for efficiency
        if metrics:
            step = state.global_step
            metrics_to_log = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_to_log[key] = value
            
            # Log all metrics at once rather than individually
            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log, step=step)
        
        # Only generate audio samples at specific intervals
        # First time, then every 2000 steps to avoid slowing down training
        if state.global_step == 0 or (state.global_step % 2000 == 0 and state.global_step > 0):
            model = kwargs.get('model', self.model)
            
            # Generate fewer samples during training to reduce overhead
            num_samples = 1 if state.global_step < args.max_steps // 2 else 2
            
            generate_audio_samples(
                model=model,
                processor=self.processor,
                dataset=self.dataset,
                vocoder=self.vocoder,
                num_samples=num_samples
            )
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log metrics to MLflow - but only every 25 steps to reduce overhead
        if logs and state.global_step % 25 == 0:
            step = state.global_step
            metrics_to_log = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    metrics_to_log[key] = value
            
            # Log all metrics at once rather than individually
            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log, step=step)
    
    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get('model', self.model)
        
        print("Training complete! Generating final audio samples...")
        # Generate final audio samples
        audio_files = generate_audio_samples(
            model=model,
            processor=self.processor,
            dataset=self.dataset,
            vocoder=self.vocoder,
            num_samples=3  # Fewer samples to reduce overhead
        )
        
        print("Registering model in MLflow model registry...")
        # Log the final model to MLflow model registry - but optimize for performance
        # First, save model and processor locally
        model_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(model_path)
        
        processor_path = os.path.join(output_dir, "processor")
        processor.save_pretrained(processor_path)
        
        # Log metadata about the run first
        mlflow.set_tag("final_eval_loss", state.log_history[-1].get("eval_loss", "N/A"))
        mlflow.set_tag("best_model_checkpoint", state.best_model_checkpoint)
        
        # Then log model to MLflow model registry
        print("Uploading model to MLflow registry (this may take a while)...")
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "processor": processor
            },
            artifact_path="model",
            registered_model_name=f"speecht5_finetuned_nl_{timestamp}"
        )
        
        # Log only essential artifacts to reduce upload time
        # Instead of uploading all checkpoints, just log the final and best models
        print("Logging final model artifacts...")
        mlflow.log_artifact(model_path, artifact_path="final_model")
        best_checkpoint = state.best_model_checkpoint
        if best_checkpoint and os.path.exists(best_checkpoint):
            mlflow.log_artifact(best_checkpoint, artifact_path="best_model_checkpoint")
        
        print(f"✅ Model registered in MLflow model registry as: speecht5_finetuned_nl_{timestamp}")

# Use the standard trainer with our custom callback
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    callbacks=[
        TensorBoardCallback(),  # Standard TensorBoard logging
        MLflowAndAudioCallback(model, processor, dataset, vocoder)  # Our custom MLflow callback
    ],
)

# Generate initial audio samples before training - but only one to save time
print("Generating initial audio sample...")
initial_samples = generate_audio_samples(model, processor, dataset, vocoder, num_samples=1)

# Train the model
trainer.train()

# Save the processor
processor_output_path = os.path.join(output_dir, "processor")
os.makedirs(processor_output_path, exist_ok=True)
processor.save_pretrained(processor_output_path)

# Push to Hugging Face Hub
if training_args.push_to_hub:
    trainer.push_to_hub()

# Log a final message about where to find the model
print(f"Training complete! Model saved to: {output_dir}")
print(f"Model registered in MLflow model registry as: speecht5_finetuned_nl_{timestamp}")
print(f"To use this model, load it from MLflow model registry or from the local directory: {output_dir}")

# End the MLflow run
mlflow.end_run()