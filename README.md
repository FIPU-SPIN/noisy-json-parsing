# noisy-json-parsing
While large language models (LLMs) demonstrate high accuracy in parsing natural language into structured data formats (e.g., JSON), current evaluations largely rely on clean, synthetic datasets. Real-world human-computer interaction, however, is characterized by spontaneous corrections, colloquialisms, and structural noise. This study investigates the robustness threshold at which prompt engineering fails and adapter-based fine-tuning (PEFT/LoRA) becomes necessary for semantic parsing tasks. We introduce a novel evaluation framework by injecting three escalating tiers of noise—typographical, conversational (fillers and hesitations), and multi-intent (mid-sentence corrections and negations)—into a baseline dataset of 1,000 retail-cart commands. Using selected open-weight LLMs, we compare the performance of optimized zero-shot and few-shot prompts against models fine-tuned via Low-Rank Adaptation (LoRA). Our evaluation tracks Exact Match Rate (EMR), schema adherence, and intent-correction accuracy. The findings aim to quantify the fragility of prompt-based extraction under real-world conditions and provide an empirical cost-benefit analysis of deploying PEFT to handle out-of-distribution user input in applied retail assistants.

RQ1 (The Threshold): At what level of input noise (typos vs. conversational fillers vs. complex intent corrections) does prompt engineering suffer a statistically significant drop in JSON extraction accuracy?

RQ2 (The Intervention): Does adapter-based fine-tuning (LoRA) on a small subset of noisy data restore or exceed the baseline accuracy achieved by few-shot prompting?

RQ3 (Error Typology): When models fail on noisy data, do they fail by hallucinating the JSON structure (syntax error) or by extracting the wrong entity/quantity (semantic error)?


Tier 0 (Baseline): Clean command. "Add two bottles of whole milk."

Tier 1 (Lexical Noise): Typos, missing punctuation, and slang. "put 2 botles of whol milk in cart rn"

Tier 2 (Conversational Noise): Fillers, hesitations, and disjointed phrasing. "Uhm, I think I need... let's add two bottles of whole milk, please."

Tier 3 (Cognitive/Intent Noise): Spontaneous corrections, negations, and conditional logic. "Add two bottles of whole milk, wait actually make it three, but only if it's the organic one."
