import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True

import unsloth
import pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from tqdm import tqdm

# =========================
# PATHS
# =========================
BASE_MODEL = "unsloth/Qwen3.5-9B"
LORA_PATH = "Model_Qwen3dot_9B/checkpoint-3400"
INPUT_CSV = "Dataset/anustubh_hn_sa_test.csv"
OUTPUT_CSV = "OUTPUTS/OUTPUTS-Qwen/anustubh_poetry_qwen3d5-9B_DEV_untrained_greedy_0shot.csv"
MAX_NEW_TOKENS = 80
device = "cuda" if torch.cuda.is_available() else "cpu"

import sys
lora = int(sys.argv[1].strip())

if lora==1:
    BASE_MODEL = LORA_PATH
    
# =========================
# LOAD MODEL
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=768,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="cuda",
)
print(model.config._name_or_path, "LOADED")

has_lora = any("lora" in name.lower() for name, _ in model.named_modules())
print("LoRA loaded:", has_lora)

model = FastLanguageModel.for_inference(model)
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
        {"role": "system", "content": [{"type": "text", "text": ANUSHTUP_INSTRUCTION}]},
        # {"role": "user", "content":  [{"type": "text", "text": "यह समय वाणीकी पहुँचके परे था उसका वर्णन करना कठिन था उस समय कोई भूपाल वहाँ इस विषयमें कुछ भी न बोल सके मौन रह गये वे बारचार केवल श्रीकृष्णके मुखकी ओर देखते रहे ॥"}]},
        # {"role": "assistant", "content":  [{"type": "text", "text": "ततः केचिन्महीपाला नानुवंस्तत्र किंचन अतीतवाक्पथे काले प्रेक्षमाणा जनार्दनम् ॥"}]},
        # {"role": "user", "content":  [{"type": "text", "text": "फिर तो उसने एक दूसरे भयंकर शत्रुको वहाँ आया हुआ देखा, जो सरकण्डेके फूलके समान भूरे रंगका था वह धरतीमें विवर बनाकर उसके भीतर सोया करता था"}]},
        # {"role": "assistant", "content": [{"type": "text", "text": "अपश्यदपरं घोरमात्मनः शत्रुमागतम् शरप्रसूनसङ्काशं महीविवरशायिनम्॥"}] },
        # {"role": "user", "content":  [{"type": "text", "text": "जो मनुष्य पाण्डुनन्दन अर्जुनके इस चरित्रको प्रतिदिन सुनता है, उसके मनमै पापपूर्ण विषयभोगोंकी इच्छा नहीं होती ॥"}]},
        # {"role": "assistant", "content":  [{"type": "text", "text": "इदं यः शृणुयाद् वृत्तं नित्यं पाण्डुसुतस्य थे न तस्य कामः कामेषु पापकेषु प्रवर्तते ॥"}]},
        # {"role": "user", "content":  [{"type": "text", "text": "मनुष्य बुद्धिबलके सिवा और किसी उपायसे सैकड़ों आघात करके भी आनेवाले अनर्थको नहीं रोक सकते"}]},
        # {"role": "assistant", "content":  [{"type": "text", "text": "नागामिनमनर्थं हि प्रतिघातशतैरपि शक्नुवन्ति प्रतिव्योढुमृते बुद्धिबलान्नराः ॥"}]},
        # {"role": "user", "content":  [{"type": "text", "text": "तदनन्तर ये दोनों क्रोधमें भरकर वारंवार सर्पाकार बाणद्वारा एक दूसरेको घायल करने लगे उस समय उन दोनोंकी बड़ी शोमा होने लगी ॥"}]},
        # {"role": "assistant", "content":  [{"type": "text", "text": "ततस्तौ तत्र संरब्धौ राजमानौ मुहुर्मुहुः शरैराशीविषाकारैस्ततक्षाते परस्परम् ॥"}]},
        # {"role": "user", "content":  [{"type": "text", "text": "मन्दबुद्धि मनुष्य ही अप्रिय वस्तुकी प्राप्ति और प्रिय वस्तुका वियोग होनेपर मनहीमन दु:खी होते हैं ॥"}]},
        # {"role": "assistant", "content":  [{"type": "text", "text": "अनिष्टसम्प्रयोगाच्च विप्रयोगात् प्रियस्य च मनुष्या मानसैर्दुःखैर्युज्यन्ते स्वल्पबुद्धयः॥"}]},
        {"role": "user", "content":  [{"type": "text", "text": hi_text}]},
    ]


    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    input_len = inputs.shape[-1]
    attention_mask = (inputs != tokenizer.pad_token_id).long()

    inputs = inputs.to(device)
    attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            # temperature=0.7, 
            # top_p=0.8, 
            # top_k=20, 
            # min_p=0.0,
            # repetition_penalty=1.0
        )

    response = tokenizer.decode(outputs[0][input_len:],skip_special_tokens=True )

    response = response.strip()

    return response

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
