# TOKEN LENGTH STATS

Token Length of the sanskrit verses in test set

| STATS  | unsloth/gemma-4-E4B-it | unsloth/Phi-4 |
|--------|------------------------|---------------|
|  min   |  12                    |      34       |
|  50p   |  33                    |      95       |
|  75p   |  35                    |      101      |
|  90p   |  37                    |      136      |
|  95p   |  38                    |      109      |
|  99p   |  41                    |      115      |
|  max   |  47                    |      136      |

### MODELS LOCATION
PHI4-DEV: https://huggingface.co/shivrajanand/poetry_gen_hi_sa_anutubh_experiment/tree/main/Models
PHI4-SLP1: https://huggingface.co/shivrajanand/poetry_gen_hi_sa_anutubh_experiment/tree/main/Models
GEMMA-4-E4B: Sanskrit Server


| Model                   |Training  | Input → Output + Shots    |Sampling|Full (%)|Partial (%)|Invalid 32 (%)|Semantic (%)|File Name                                    |
|-------------------------|----------|---------------------------|--------|--------|-----------|--------------|------------|---------------------------------------------|
| Chandomitra (Phi-4)     |Finetuned | EN → SA                   |—       |57.42   |75.01      |17.59         |67.29       |                                             |
| Ground Truth            |—         | DEV Sanskrit + Hindi Prose|—       |99.51   |99.51      |0.00          |74.04       |                                             |
| unsloth/phi-4           |Finetuned | SLP1 → SLP1 / SLP1 `[*]`  |greedy  |24.76   |62.44      |37.68         |73.23       |anustubh_poetry_phi4_SLP1                    |
| unsloth/phi-4           |Finetuned | DEV → DEV + 0-shot        |greedy  |43.08   |66.99      |23.91         |73.76       |anustubh_poetry_phi4_DEV                     |
| unsloth/phi-4           |Finetuned | DEV → DEV + 3-shot        |greedy  |50.97   |72.81      |21.84         |73.21       |anustubh_poetry_phi4_DEV_3shot               |
| unsloth/phi-4           |Finetuned | DEV → DEV + 6-shot        |greedy  |46.72   |69.05      |22.33         |72.93       |anustubh_poetry_phi4_DEV_6shot               |
| unsloth/phi-4           |Finetuned | DEV → DEV + 0-shot        |sampling|42.90   |64.68      |21.78         |72.71       |anustubh_poetry_phi4_DEV_sampling_true       |
| unsloth/phi-4           |Finetuned | DEV → DEV + 3-shot        |sampling|51.70   |72.51      |20.81         |72.05       |anustubh_poetry_phi4_DEV_sampling_plus3shot  |
| unsloth/gemma-4-E4B-it  |UNTRAINED | DEV → DEV + 3-shot        |greedy  |0.00    |8.25       |8.25          |77.78       |anustubh_poetry_gemma4-8B_untrained_0shot    |
| unsloth/gemma-4-E4B-it  |Finetuned | DEV → DEV + 0-shot        |greedy  |0.73    |34.35      |33.62         |72.26       |anustubh_poetry_gemma4-8B_DEV_sampling_0shot |
| unsloth/gemma-4-E4B-it  |Finetuned | DEV → DEV + 3-shot        |sampling|1.15    |33.37      |32.22         |71.32       |anustubh_poetry_gemma4-8B_DEV_sampling_3shot |
| unsloth/Qwen3.5-9B      |UNTRAINED | DEV → DEV + 0-shot        |greedy  |        |           |              |            |                                             |
| unsloth/Qwen3.5-9B      |UNTRAINED | DEV → DEV + 3-shot        |greedy  |        |           |              |            |                                             |
| unsloth/Qwen3.5-9B      |Finetuned | DEV → DEV + 0-shot        |greedy  |        |           |              |            |                                             |
| unsloth/Qwen3.5-9B      |Finetuned | DEV → DEV + 3-shot        |greedy  |        |           |              |            |                                             |

**BOLD**: MAX
*ITALIC*: Second Max
### Notes
- **Full (%)**: Percentage of outputs with 32 syllables and perfect anustubh pattern. 
- **Partial (%)**: Percentage of outputs with 32 syllables  
- **Invalid 32 (%)**: 32-syllable outputs that violate metrical constraints *(Partial − Full)*  
- `[*]` Evaluation performed in Devanagari (SLP1 outputs converted to DEV)
- All verses are obtained via greedy decoding unless stated otherwise
