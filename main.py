from transformers import pipeline
import scipy
from datasets import load_dataset, Audio
from transformers import SpeechT5Processor
from collections import defaultdict
import matplotlib.pyplot as plt

""" 
Preprocess the data 
""" 

dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
print(len(dataset))

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
plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Examples")
plt.show()

def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400

dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
num_speakers = len(set(dataset["speaker_id"]))
len_dataset = len(dataset)

print(f"no. of speakers: {num_speakers}")
print(f"length of dataset: {len_dataset}")

""" 
Clean the data 
""" 
import os
import torch
from speechbrain.inference.classifiers import EncoderClassifier
from accelerate.test_utils.testing import get_backend

