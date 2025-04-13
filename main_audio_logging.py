from transformers import pipeline
import scipy
from datasets import load_dataset, Audio
from transformers import SpeechT5Processor, SpeechT5HifiGan
from collections import defaultdict
import matplotlib.pyplot as plt
import wandb  
import os
import torch


""" 
Preprocess the data 
""" 

# Initialize wandb at the beginning of your script
wandb.init(
    project="speecht5-tts-finetuning",  # name this whatever you want
    config={
        "model_checkpoint": "microsoft/speecht5_tts",
        "dataset": "facebook/voxpopuli",
        "language": "nl",
        "batch_size": 4,
        "learning_rate": 1e-5,
        "warmup_steps": 500,
        "max_steps": 4000,
    }
)

dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
print(len(dataset))
# Log the initial dataset size
wandb.log({"initial_dataset_size": len(dataset)})

# synthesiser = pipeline("text-to-speech", "suno/bark-small")
# speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"do_sample": True})
# scipy.io.wavfile.write("bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])

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

# Log unsupported characters as a Table
wandb_unsupported_table = wandb.Table(columns=["character"])
for char in unsupported_chars:
    wandb_unsupported_table.add_data(char)
wandb.log({"unsupported_characters": wandb_unsupported_table})

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
plt.savefig('speaker_examples_histogram.png')

# Log the histogram to wandb
wandb.log({"speaker_examples_histogram": wandb.Image(plt)})

def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400

dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
num_speakers = len(set(dataset["speaker_id"]))
len_dataset = len(dataset)

print(f"no. of speakers: {num_speakers}")
print(f"length of dataset: {len_dataset}")

# Log dataset statistics after filtering
wandb.log({
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
plt.savefig("log-mel_spectrogram_example.png")

# Log example spectrogram to wandb
wandb.log({"spectrogram_example": wandb.Image(plt)})

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200

dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
print(f"length of filtered dataset: {len(dataset)}")

# Log dataset size after length filtering
wandb.log({"length_filtered_dataset_size": len(dataset)})

dataset = dataset.train_test_split(test_size=0.1)
wandb.log({
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
Audio Logging Function
"""
def log_audio_samples(model, processor, dataset, vocoder, num_samples=3):
    """
    Generate and log audio samples from the model to Weights & Biases
    
    Args:
        model: The SpeechT5ForTextToSpeech model
        processor: The SpeechT5Processor
        dataset: The dataset dictionary containing 'test' split
        vocoder: SpeechT5HifiGan vocoder for converting spectrograms to audio
        num_samples: Number of samples to generate and log
    """
    print("Generating audio samples for logging...")
    
    # Get texts from test set for visualization
    examples = [dataset["test"][i] for i in range(min(num_samples, len(dataset["test"])))]
    
    # Move vocoder to the same device as the model
    device = model.device
    vocoder = vocoder.to(device)
    
    # Store original texts for captions
    original_texts = []
    for i, example_idx in enumerate(range(min(num_samples, len(dataset["test"])))):
        # Get original text if available in the dataset metadata
        try:
            original_text = dataset["test"].features["normalized_text"][example_idx]
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
        
        # Convert to numpy for wandb
        audio_array = speech.cpu().numpy()
        
        # Create a caption - use original text if available
        caption = original_texts[i] if i < len(original_texts) else f"Generated speech sample {i+1}"
        
        # Log to wandb
        wandb.log({
            f"audio_sample_{i+1}": wandb.Audio(
                audio_array, 
                sample_rate=16000,  # SpeechT5 uses 16kHz
                caption=caption
            )
        })
        
        print(f"Logged audio sample {i+1}")

"""
Training the model with W&B model registry upload
"""

import wandb
from transformers import SpeechT5ForTextToSpeech
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers.integrations import WandbCallback
import os
from datetime import datetime

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
    report_to=["tensorboard", "wandb"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

# Create a custom W&B callback to register the model and log audio samples
class ModelRegistryAndAudioCallback(WandbCallback):
    def __init__(self, model, processor, dataset, vocoder):
        super().__init__()
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.vocoder = vocoder
        
    def on_evaluate(self, args, state, control, **kwargs):
        # Call the parent method correctly with only the expected arguments
        super().on_evaluate(args, state, control)
        
        # Get the model from kwargs or use self.model
        model = kwargs.get('model', self.model)
        
        # Only log audio samples every 1000 steps to avoid too many samples
        if state.global_step % 1000 == 0:
            log_audio_samples(
                model=model,
                processor=self.processor,
                dataset=self.dataset,
                vocoder=self.vocoder,
                num_samples=3
            )
            
    def on_train_end(self, args, state, control, **kwargs):
        super().on_train_end(args, state, control)
        
        # Get the model from kwargs or use self.model
        model = kwargs.get('model', self.model)
        
        # Log final audio samples
        log_audio_samples(
            model=model,
            processor=self.processor,
            dataset=self.dataset,
            vocoder=self.vocoder,
            num_samples=5  # Log more samples at the end
        )
        
        # Create an artifact for the model
        model_artifact = wandb.Artifact(
            name=f"speecht5_finetuned_nl_{timestamp}",
            type="model",
            description="Fine-tuned SpeechT5 model on VoxPopuli Dutch dataset"
        )
        
        # Add the model directory to the artifact
        model_artifact.add_dir(args.output_dir)
        
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
        
        # Log and register the artifact
        wandb.log_artifact(model_artifact)
        print(f"✅ Model artifact '{model_artifact.name}' uploaded to W&B model registry")
        
# Use the standard trainer with our custom callback
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    callbacks=[ModelRegistryAndAudioCallback(model, processor, dataset, vocoder)],  # Add our custom callback
)

# Generate initial audio samples before training
log_audio_samples(model, processor, dataset, vocoder, num_samples=2)

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
print(f"To use this model, load it from W&B or from the local directory: {output_dir}")

# Close wandb run when done
wandb.finish()