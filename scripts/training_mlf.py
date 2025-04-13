from transformers import pipeline
import scipy
from datasets import load_dataset, Audio
from transformers import SpeechT5Processor
from collections import defaultdict
import matplotlib.pyplot as plt
import wandb  # Existing wandb import
import mlflow  # Add mlflow import
import os
from mlflow.tracking import MlflowClient

""" 
Preprocess the data 
"""

# Set up MLflow tracking
# For local tracking server, set the tracking URI to your local server
mlflow.set_tracking_uri("http://localhost:5000")  # Adjust if your server is on a different port
# Create or get an experiment
experiment_name = "speecht5-tts-finetuning"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run(run_name="speecht5_voxpopuli_nl") as run:
    run_id = run.info.run_id
    client = MlflowClient()
    
    # Log parameters (same as those in your wandb config)
    params = {
        "model_checkpoint": "microsoft/speecht5_tts",
        "dataset": "facebook/voxpopuli",
        "language": "nl",
        "batch_size": 4,
        "learning_rate": 1e-5,
        "warmup_steps": 500,
        "max_steps": 4000,
    }
    mlflow.log_params(params)
    
    # Initialize wandb as before
    wandb.init(
        project="speecht5-tts-finetuning",
        config=params
    )

    dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
    print(len(dataset))
    # Log the initial dataset size
    wandb.log({"initial_dataset_size": len(dataset)})
    mlflow.log_metric("initial_dataset_size", len(dataset))

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

    # Log unsupported characters
    with open("unsupported_chars.txt", "w") as f:
        for char in unsupported_chars:
            f.write(f"{char}\n")
    mlflow.log_artifact("unsupported_chars.txt")

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

    # Log the histogram to mlflow
    mlflow.log_artifact("speaker_examples_histogram.png")

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
    plt.savefig("log-mel_spectrogram_example.png")

    # Log example spectrogram to mlflow
    mlflow.log_artifact("log-mel_spectrogram_example.png")

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

    def is_not_too_long(input_ids):
        input_length = len(input_ids)
        return input_length < 200

    dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
    print(f"length of filtered dataset: {len(dataset)}")

    # Log dataset size after length filtering
    wandb.log({"length_filtered_dataset_size": len(dataset)})
    mlflow.log_metric("length_filtered_dataset_size", len(dataset))

    dataset = dataset.train_test_split(test_size=0.1)
    wandb.log({
        "train_set_size": len(dataset["train"]),
        "test_set_size": len(dataset["test"])
    })
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

    data_collator = TTSDataCollatorWithPadding(processor=processor)

    """
    Training Model with MLflow tracking
    """
    from transformers import SpeechT5ForTextToSpeech
    from transformers import Seq2SeqTrainingArguments
    from transformers import Seq2SeqTrainer
    import numpy as np
    from typing import Dict
    
    # Create a custom callback for MLflow logging
    class MLflowCallback:
        def __init__(self, mlflow_run_id):
            self.mlflow_run_id = mlflow_run_id
            self.client = MlflowClient()
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for k, v in logs.items():
                    if isinstance(v, (int, float, np.int32, np.int64, np.float32, np.float64)):
                        mlflow.log_metric(k, v, step=state.global_step)

        def on_save(self, args, state, control, **kwargs):
            # Log model checkpoints as artifacts
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(checkpoint_dir):
                mlflow.log_artifact(checkpoint_dir, artifact_path="checkpoints")

    model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
    model.config.use_cache = False

    # Log model architecture info to MLflow
    model_info = {
        "model_type": model.config.model_type,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "encoder_layers": model.config.encoder_layers,
        "decoder_layers": model.config.decoder_layers
    }
    mlflow.log_params(model_info)

    training_args = Seq2SeqTrainingArguments(
        output_dir="speecht5_finetuned_voxpopuli_nl",  
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
        report_to=["tensorboard"],   
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        push_to_hub=True,
    )

    # Create custom trainer with MLflow callback
    class MLflowTrainer(Seq2SeqTrainer):
        def __init__(self, *args, mlflow_run_id=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.mlflow_run_id = mlflow_run_id
            self._add_callback(MLflowCallback(mlflow_run_id))
            
        def log(self, logs: Dict[str, float]) -> None:
            """
            Override log method to also log to MLflow
            """
            super().log(logs)  # Call the original log method
            
            # Log to MLflow
            logs_to_record = {}
            for k, v in logs.items():
                if isinstance(v, (int, float, np.int32, np.int64, np.float32, np.float64)):
                    logs_to_record[k] = v
            
            if logs_to_record:
                mlflow.log_metrics(logs_to_record, step=self.state.global_step)

    trainer = MLflowTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=processor,
        mlflow_run_id=run_id,
    )

    # Train the model
    trainer.train()
    
    # Log the final model and processor
    output_dir = "final_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    mlflow.log_artifact(output_dir, artifact_path="final_model")
    
    # Push to hub if needed
    processor.save_pretrained("michellelychan/speecht5_finetuned_voxpopuli_nl")
    trainer.push_to_hub()
    
    # End the MLflow run
    mlflow.end_run()