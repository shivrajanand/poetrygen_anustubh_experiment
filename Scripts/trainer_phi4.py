from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import train_on_responses_only
import json
from transformers import EarlyStoppingCallback
import random

import os
os.environ["HF_HUB_OFFLINE"] = "1"

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

HYPERPARAMS = {
    "MODEL_NAME": "unsloth/phi-4",
    "MAX_LEN": 1024, # max token length calculated for entire text in chat template comes out as 852 for slp1 and 968 for DEV
    "LOAD_IN_4BIT": True,
    "BATCH_SIZE": 8,
    "GRAD_ACC": 2,
    "EPOCHS": 10,
    "LR": 2e-4,
    "LOG_STEPS": 50,
    "SAVE_STEPS": 200,
    "SAVE_LIMIT": 3,
    "EVAL_STEPS": 200,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.03,

    "LORA_R": 16,
    "LORA_ALPHA": 32,
    "LORA_DROPOUT": 0.05,
    
    "ES_THRESHOLD": 0.001,
    "ES_PATIENCE": 5,

    "OUTPUT_DIR": "Phi4-14B-DEV",

}


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=HYPERPARAMS["MODEL_NAME"],
    max_seq_length=HYPERPARAMS["MAX_LEN"],
    load_in_4bit=HYPERPARAMS["LOAD_IN_4BIT"]
)

model = FastLanguageModel.get_peft_model(
    model,
    r=HYPERPARAMS["LORA_R"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=HYPERPARAMS["LORA_ALPHA"],
    lora_dropout=HYPERPARAMS["LORA_DROPOUT"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

ds = load_dataset('csv', data_files="anustubh_hn_sa_train.csv")["train"]

ANUSHTUP_INSTRUCTION = """The goal is to generate Sanskrit verse that follows the anushtup meter rules for the given input text.
RULES:
Verse Rules:
The verse contains 32 syllables/akshara and 4 padas in total.
The verse is divided into 2 lines, each containing 16 syllables.
Each line is divided into 2 padas (quartets), each containing exactly 8 syllables.
The fifth syllable of every pada must be LAGHU or short.
The sixth syllable of every pada must be GURU or long.
The seventh syllable of the second and fourth pada must be HRASVA.
The seventh syllable of the first and third pada must be DEERGHA.

Syllable Rules:
LAGHU vowels: अ, इ, उ, ऋ, ऌ
GURU vowels: आ, ई, ऊ, ॠ, ॡ, ए, ऐ, ओ, औ
HRASVA vowels: अ, इ, उ, ऋ, ऌ
DEERGHA vowels: आ, ई, ऊ, ॠ, ॡ, ए, ऐ, ओ, औ

Syllable classification rules:
- A syllable is marked Laghu/Guru and Hrasva/Deergha based on the vowel it contains.
- Any syllable containing anusvāra (ं) or visarga (ः) is always Guru.
- Any syllable followed by a conjunct consonant (saṁyuktākṣara) is always Guru.
Now convert the given Hindi text into a Sanskrit Anushtup verse in Devanagari:"""


def format_and_tokenize(batch):
    texts = []
    
    for hi, sa in zip(batch["hi"], batch["clean_text"]):
        convo = [
            {"role": "system", "content": ANUSHTUP_INSTRUCTION},
            {"role": "user", "content": hi},
            {"role": "assistant", "content": sa},
        ]
        
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        
        texts.append(text)
    
    return {"text": texts}

tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

ds = ds.map(format_and_tokenize, batched=True)

print("--------------------------------------------------------------------------------------")
for key in ds[0].keys():
    print(key, ds[0][key], sep="\n\n")
print("--------------------------------------------------------------------------------------")

split_datasets = ds.train_test_split(test_size=0.1, seed=42)

# 3. Access the resulting train and test datasets
train_ds = split_datasets['train']
eval_ds = split_datasets['test']

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=HYPERPARAMS["MAX_LEN"],
    packing=False,
    dataset_num_proc=1,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=HYPERPARAMS["ES_PATIENCE"], early_stopping_threshold = HYPERPARAMS["ES_THRESHOLD"])],
    args=SFTConfig(
        output_dir=HYPERPARAMS["OUTPUT_DIR"],

        per_device_train_batch_size=HYPERPARAMS["BATCH_SIZE"],
        gradient_accumulation_steps=HYPERPARAMS["GRAD_ACC"],
        num_train_epochs=HYPERPARAMS["EPOCHS"],

        learning_rate=HYPERPARAMS["LR"],
        lr_scheduler_type="linear",
        warmup_ratio=HYPERPARAMS["WARMUP_RATIO"],
        weight_decay=HYPERPARAMS["WEIGHT_DECAY"],

        logging_steps=HYPERPARAMS["LOG_STEPS"],
        logging_strategy="steps",
        logging_dir=HYPERPARAMS["OUTPUT_DIR"] + "/runs",
        report_to="tensorboard",

        save_steps=HYPERPARAMS["SAVE_STEPS"],
        save_total_limit=HYPERPARAMS["SAVE_LIMIT"],

        eval_strategy="steps",
        eval_steps=HYPERPARAMS["EVAL_STEPS"],


        fp16=False,
        bf16=True,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # max_steps=30,
        optim="adamw_8bit",
        seed=3407,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

trainer.train()

print("BEST MODEL STATS")
print(trainer.state.best_model_checkpoint)
print(trainer.state.best_metric)

essential_config = {
    "HYPER-PARAMETERS": HYPERPARAMS,
    "TRAIN_DATASET_LEN": len(train_ds),
    "VAL_DATASET_LEN": len(eval_ds),
    "best_model": {"best_model_checkpoint": trainer.state.best_model_checkpoint,
    "best_model_metric": trainer.state.best_metric}
}

with open(HYPERPARAMS["OUTPUT_DIR"]+"/essential_config.json", "w", encoding="utf-8") as f:
    json.dump(essential_config, f, indent=4, default=str)