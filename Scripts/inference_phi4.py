import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm

# =========================
# PATHS
# =========================
BASE_MODEL = "unsloth/phi-4"
LORA_PATH = "Phi4-14B-DEV/checkpoint-3400"
INPUT_CSV = "anustubh_hn_sa_test.csv"
OUTPUT_CSV = "anustubh_poetry_phi4_DEV_sampling_plus3shot.csv"
MAX_NEW_TOKENS = 110
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)

model.to(device)
model.eval()

# =========================
# INSTRUCTION
# =========================
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


# =========================
# GENERATION
# =========================
def generate(hi_text):
    messages = [
        {"role": "system", "content": ANUSHTUP_INSTRUCTION},
        {"role": "user", "content": "यह समय वाणीकी पहुँचके परे था उसका वर्णन करना कठिन था उस समय कोई भूपाल वहाँ इस विषयमें कुछ भी न बोल सके मौन रह गये वे बारचार केवल श्रीकृष्णके मुखकी ओर देखते रहे ॥"},
        {"role": "assistant", "content": "ततः केचिन्महीपाला नानुवंस्तत्र किंचन अतीतवाक्पथे काले प्रेक्षमाणा जनार्दनम् ॥"},
        {"role": "user", "content": "फिर तो उसने एक दूसरे भयंकर शत्रुको वहाँ आया हुआ देखा, जो सरकण्डेके फूलके समान भूरे रंगका था वह धरतीमें विवर बनाकर उसके भीतर सोया करता था"},
        {"role": "assistant", "content": "अपश्यदपरं घोरमात्मनः शत्रुमागतम् शरप्रसूनसङ्काशं महीविवरशायिनम्॥"},
        {"role": "user", "content": "जो मनुष्य पाण्डुनन्दन अर्जुनके इस चरित्रको प्रतिदिन सुनता है, उसके मनमै पापपूर्ण विषयभोगोंकी इच्छा नहीं होती ॥"},
        {"role": "assistant", "content": "इदं यः शृणुयाद् वृत्तं नित्यं पाण्डुसुतस्य थे न तस्य कामः कामेषु पापकेषु प्रवर्तते ॥"},
        {"role": "user", "content": hi_text},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.6, 
            top_p=0.9, 
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# =========================
# RUN
# =========================
df = pd.read_csv(INPUT_CSV)
size = df.shape[0]
print("Total Samples = ", size)


df["model_out"] = ""
for i in tqdm(range(size)):
    pred = generate(df.iloc[i]["hi"])
    df.loc[i, "model_out"] = pred

    if i % 50 == 0:
        df.to_csv(OUTPUT_CSV, index=False)

df.to_csv(OUTPUT_CSV, index=False)
