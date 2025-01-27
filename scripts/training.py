#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# In[2]:


from datasets import load_dataset, DatasetDict

data_dict = load_dataset("parquet", data_files="../data/fr.parquet")


# In[3]:


data_dict = data_dict['train'].train_test_split(0.2)


# In[4]:


data_dict


# In[5]:


from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="fr", task="transcribe"
)


# In[6]:


data_dict["train"].features


# In[7]:


def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    # Lenght in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example


# In[8]:


data_dict = data_dict.map(
    prepare_dataset, remove_columns=data_dict.column_names["train"], num_proc=1
)


# In[9]:


max_input_length = 30.0

def is_audio_in_length_range(length):
    return length < max_input_length


# In[10]:


def count_samples(data_dict: DatasetDict) -> int:
    train_count = data_dict['train'].num_rows
    test_count = data_dict['test'].num_rows
    return train_count + test_count


# In[11]:


before = count_samples(data_dict)
print(f"Before filter: {before}")

data_dict["train"] = data_dict["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

after = count_samples(data_dict)
print(f"After filter: {after}")


# In[12]:


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyway
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# In[13]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# In[14]:


import evaluate

metric = evaluate.load("wer")


# In[15]:


from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


# In[16]:


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)


# In[17]:


from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="fr", task="transcribe", use_cache=True
)


# In[18]:


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="../models/whisper-small-fr",
    logging_dir="../logs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


# In[19]:


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data_dict["train"],
    eval_dataset=data_dict["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)


# In[ ]:


trainer.train()


# In[ ]:




