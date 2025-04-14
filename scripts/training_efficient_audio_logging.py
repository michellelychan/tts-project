from transformers import pipeline, SpeechT5Processor, SpeechT5HifiGan, SpeechT5ForTextToSpeech
from datasets import load_dataset, Audio
from collections import defaultdict
import matplotlib.pyplot as plt
import wandb
import os
import torch
import scipy
import shutil
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.integrations import WandbCallback
from speechbrain.inference.classifiers import EncoderClassifier

# Generate a unique run name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"speecht5_nl_run_{timestamp}"
output_dir = f"../models/speecht5_finetuned_voxpopuli_nl_{timestamp}"

# Cleanup old runs if needed (keep only last 3)
def cleanup_old_runs(base_dir="../models/", keep=3):
    """Remove older model directories, keeping only the most recent ones."""
    try:
        dirs = [d for d in os.listdir(base_dir) if d.startswith("speecht5_finetuned_voxpopuli_nl_")]
        if len(dirs) <= keep:
            return
        
        # Sort by creation time (oldest first)
        dirs.sort(key=lambda x: os.path.getctime(os.path.join(base_dir, x)))
        
        # Delete oldest dirs
        for old_dir in dirs[:-keep]:
            old_path = os.path.join(base_dir, old_dir)
            print(f"Cleaning up old directory: {old_path}")
            shutil.rmtree(old_path, ignore_errors=True)
    except Exception as e:
        print(f"Error cleaning up old runs: {e}")

# Call cleanup before starting new run
cleanup_old_runs()

# Initialize wandb (only once)
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

"""
Preprocess the data - OPTIMIZED WITH STREAMING
"""
print("Loading and preprocessing dataset...")

# Load dataset with streaming to reduce memory usage
dataset = load_dataset("facebook/voxpopuli", "nl", split="train", streaming=True)
# Cache a small sample for analysis
sample_dataset = dataset.take(1000)
sample_dataset = list(sample_dataset)

# Log the initial dataset info
wandb.log({"initial_dataset_sample_size": len(sample_dataset)})

# Initialize processor early to use in filtering
checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
tokenizer = processor.tokenizer

# Analyze vocab once on sample
def extract_all_chars(texts):
    all_text = " ".join(texts)
    vocab = list(set(all_text))
    return vocab

# Extract vocabulary from sample
sample_texts = [item["normalized_text"] for item in sample_dataset]
dataset_vocab = set(extract_all_chars(sample_texts))
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
unsupported_chars = dataset_vocab - tokenizer_vocab
print(f"Unsupported characters: {unsupported_chars}")

# Log unsupported characters
wandb_unsupported_table = wandb.Table(columns=["character"])
for char in unsupported_chars:
    wandb_unsupported_table.add_data(char)
wandb.log({"unsupported_characters": wandb_unsupported_table})

# Character replacements
replacements = [
    ("à", "a"), ("ç", "c"), ("è", "e"), ("ë", "e"),
    ("í", "i"), ("ï", "i"), ("ö", "o"), ("ü", "u"),
]

# Initialize speaker model - move to CPU when not actively used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": "cpu"},  # Start on CPU
    savedir=os.path.join("/tmp", spk_model_name),
)

# Analyze speaker distribution from sample to set filtering thresholds
print("Analyzing speaker distribution...")
speaker_counts = defaultdict(int)
for item in sample_dataset:
    speaker_counts[item["speaker_id"]] += 1

plt.figure()
hist = plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Examples")
os.makedirs('../outputs/images/', exist_ok=True)
plt.savefig('../outputs/images/speaker_examples_histogram.png')
wandb.log({"speaker_examples_histogram": wandb.Image(plt)})
plt.close()  # Close the plot to free memory

# Define optimal speaker range based on distribution
min_examples = 100
max_examples = 400

# Function to create speaker embedding with proper resource management
def create_speaker_embedding(waveform):
    # Move model to GPU only during processing
    speaker_model.to(device)
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform).to(device))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    # Move model back to CPU to free GPU memory
    speaker_model.to("cpu")
    torch.cuda.empty_cache()  # Clear CUDA cache
    return speaker_embeddings

# Combined preprocessing function that does cleaning and filtering in one pass
def preprocess_item(example):
    # Check if speaker should be included
    if speaker_counts.get(example["speaker_id"], 0) < min_examples or speaker_counts.get(example["speaker_id"], 0) > max_examples:
        return None
    
    # Clean text
    text = example["normalized_text"]
    for src, dst in replacements:
        text = text.replace(src, dst)
    
    # Resample audio
    audio = example["audio"]
    if not isinstance(audio, dict):
        try:
            # Handle streaming dataset format
            audio = {"array": audio["array"], "sampling_rate": audio["sampling_rate"]}
        except:
            # Skip problematic examples
            return None
    
    if "sampling_rate" not in audio or audio["sampling_rate"] != 16000:
        try:
            # Resample if needed
            audio = {"array": audio["array"], "sampling_rate": 16000}
        except:
            return None
    
    # Process with the tokenizer and feature extractor
    try:
        processed = processor(
            text=text,
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )
        
        # Check if input is too long (filter early)
        if len(processed["input_ids"][0]) >= 200:
            return None
            
        # Strip off batch dimension
        processed["labels"] = processed["labels"][0]
        
        # Add speaker embeddings
        processed["speaker_embeddings"] = create_speaker_embedding(audio["array"])
        
        # Add the original text for reference
        processed["original_text"] = text
        
        return processed
    except Exception as e:
        print(f"Error processing item: {e}")
        return None

# Define batch processing to speed up dataset preprocessing
def process_batch(batch, batch_size=16):
    results = []
    for i in range(0, len(batch), batch_size):
        sub_batch = batch[i:i+batch_size]
        
        # Process items in batch
        processed_items = []
        for item in sub_batch:
            processed = preprocess_item(item)
            if processed is not None:
                processed_items.append(processed)
        
        results.extend(processed_items)
    return results

# Process dataset with batching and convert to regular dataset
print("Processing full dataset...")
processed_dataset = []

# Process in batches of 1000 for memory efficiency
batch_size = 1000
dataset_batches = dataset.iter(batch_size)

max_examples = 50000  # Limit total examples to process
total_processed = 0

for batch in dataset_batches:
    if total_processed >= max_examples:
        break
        
    processed_batch = process_batch(batch)
    processed_dataset.extend(processed_batch)
    
    total_processed += len(processed_batch)
    print(f"Processed {total_processed} examples")
    
    # Clear memory periodically
    if total_processed % 5000 == 0:
        torch.cuda.empty_cache()

# Convert to HF dataset
from datasets import Dataset
dataset = Dataset.from_dict({
    key: [item[key] for item in processed_dataset] 
    for key in processed_dataset[0].keys()
})

print(f"Final dataset size: {len(dataset)}")
wandb.log({"processed_dataset_size": len(dataset)})

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)
wandb.log({
    "train_set_size": len(dataset["train"]),
    "test_set_size": len(dataset["test"])
})

# Log a sample spectrogram
print("Logging sample spectrogram...")
if len(dataset["train"]) > 0:
    plt.figure()
    plt.imshow(dataset["train"][0]["labels"].T)
    plt.savefig("../outputs/images/log-mel_spectrogram_example.png")
    wandb.log({"spectrogram_example": wandb.Image(plt)})
    plt.close()  # Close the plot to free memory

"""
Data collator - updated for better memory management
"""
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
Audio Logging Function - Optimized for memory
"""
def log_audio_samples(model, processor, dataset, vocoder, num_samples=3):
    """Generate and log audio samples from the model to Weights & Biases"""
    print("Generating audio samples for logging...")
    
    # Get a few examples from test set
    examples = [dataset["test"][i] for i in range(min(num_samples, len(dataset["test"])))]
    
    # Move everything to the same device and then back
    device = model.device
    vocoder = vocoder.to(device)
    
    # Get original texts for captions
    original_texts = [example.get("original_text", f"Sample {i+1}") 
                     for i, example in enumerate(examples)]
    
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
        
        # Move to CPU immediately
        audio_array = speech.cpu().numpy()
        
        # Log to wandb
        caption = original_texts[i] if i < len(original_texts) else f"Generated speech sample {i+1}"
        wandb.log({
            f"audio_sample_{i+1}": wandb.Audio(
                audio_array, 
                sample_rate=16000,
                caption=caption
            )
        })
        
        # Clear memory after each sample
        torch.cuda.empty_cache()
        
    # Move vocoder back to CPU when done
    vocoder = vocoder.cpu()
    torch.cuda.empty_cache()

"""
Custom Callback with Better Memory Management
"""
class ModelRegistryAndAudioCallback(WandbCallback):
    def __init__(self, model, processor, dataset, vocoder):
        super().__init__()
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.vocoder = vocoder
        
    def on_evaluate(self, args, state, control, **kwargs):
        # Call parent method
        super().on_evaluate(args, state, control)
        
        # Get the model
        model = kwargs.get('model', self.model)
        
        # Log audio less frequently (every 1000 steps)
        if state.global_step % 1000 == 0:
            log_audio_samples(
                model=model,
                processor=self.processor,
                dataset=self.dataset,
                vocoder=self.vocoder,
                num_samples=2  # Reduced number of samples
            )
            
            # Explicitly clear cache
            torch.cuda.empty_cache()
            
    def on_train_end(self, args, state, control, **kwargs):
        super().on_train_end(args, state, control)
        
        # Get the model
        model = kwargs.get('model', self.model)
        
        # Log final audio samples
        log_audio_samples(
            model=model,
            processor=self.processor,
            dataset=self.dataset,
            vocoder=self.vocoder,
            num_samples=3  # Reduced from 5
        )
        
        # Create an artifact for the model
        model_artifact = wandb.Artifact(
            name=f"speecht5_finetuned_nl_{timestamp}",
            type="model",
            description="Fine-tuned SpeechT5 model on VoxPopuli Dutch dataset"
        )
        
        # Only add essential files to the artifact
        model_dirs_to_keep = ["config.json", "generation_config.json", "pytorch_model.bin"]
        for file_or_dir in model_dirs_to_keep:
            path = os.path.join(args.output_dir, file_or_dir)
            if os.path.exists(path):
                if os.path.isdir(path):
                    model_artifact.add_dir(path)
                else:
                    model_artifact.add_file(path)
        
        # Add metadata
        model_artifact.metadata = {
            "base_model": "microsoft/speecht5_tts",
            "dataset": "facebook/voxpopuli",
            "language": "nl",
            "fine_tuning_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "final_loss": state.log_history[-1].get("eval_loss", "N/A"),
            "best_model_checkpoint": state.best_model_checkpoint
        }
        
        # Log artifact
        wandb.log_artifact(model_artifact)
        print(f"✅ Model artifact uploaded to W&B model registry")
        
        # Clean up memory
        torch.cuda.empty_cache()

"""
Main Training Loop
"""
print("Initializing model and training...")

# Load model
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
model.config.use_cache = False

# Load vocoder (keep on CPU until needed)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").cpu()

# Create data collator
data_collator = TTSDataCollatorWithPadding(processor=processor)

# Training arguments with reduced checkpoint frequency
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
    save_steps=2000,       # Reduced from 1000
    save_total_limit=2,    # Only keep 2 checkpoints
    eval_steps=1000,
    logging_steps=50,      # Increased from 25
    report_to=["tensorboard", "wandb"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

# Use our custom callback
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    callbacks=[ModelRegistryAndAudioCallback(model, processor, dataset, vocoder)],
)

# Generate initial audio samples (fewer samples)
log_audio_samples(model, processor, dataset, vocoder, num_samples=1)

# Train the model
try:
    trainer.train()
    
    # Save the processor
    processor_output_path = os.path.join(output_dir, "processor")
    os.makedirs(processor_output_path, exist_ok=True)
    processor.save_pretrained(processor_output_path)
    
    # Push to Hub
    if training_args.push_to_hub:
        trainer.push_to_hub()
        
    print(f"Training complete! Model saved to: {output_dir}")
    
except Exception as e:
    print(f"Error during training: {e}")
finally:
    # Always clean up
    torch.cuda.empty_cache()
    
    # Close wandb run
    wandb.finish()