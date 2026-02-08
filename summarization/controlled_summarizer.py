"""
controlled_summarizer.py
Demonstrates how generation parameters affect summary length and style.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------------------------------------
# Device detection: prefer Apple MPS GPU, then CUDA, else CPU
# -----------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# -----------------------------------------------------------
# Load a BART-large-cnn model for higher quality summaries
# -----------------------------------------------------------
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

article = """
Remote work has undergone a dramatic transformation since 2020. What began as an
emergency response to a global pandemic has evolved into a permanent fixture of the
modern workplace. Major technology companies including Google, Microsoft, and Meta have
adopted hybrid policies that allow employees to work from home several days per week.
Studies from Stanford University indicate that remote workers are 13 percent more
productive than their in-office counterparts, though they report higher rates of
isolation and difficulty collaborating on creative tasks. The commercial real estate
market has responded accordingly, with office vacancy rates in major cities reaching
historic highs. Meanwhile, smaller cities and rural areas have experienced population
growth as workers relocate away from expensive urban centers. Employers are investing
heavily in collaboration software, virtual reality meeting rooms, and asynchronous
communication tools to bridge the gap. Labor economists predict that by 2030,
approximately 40 percent of knowledge workers will operate in a fully remote or
hybrid arrangement, fundamentally altering commuting patterns, urban planning, and
the nature of professional relationships.
"""

inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=1024).to(device)

# -----------------------------------------------------------
# Short, precise summary using greedy decoding
# -----------------------------------------------------------
with torch.no_grad():
    short_ids = model.generate(
        inputs["input_ids"],
        max_length=40,
        min_length=15,
        num_beams=1,          # greedy decoding
        do_sample=False,
        length_penalty=0.5,   # favor brevity
    )
short_summary = tokenizer.decode(short_ids[0], skip_special_tokens=True)
print("=== Short Summary (greedy, length_penalty=0.5) ===")
print(short_summary)

# -----------------------------------------------------------
# Longer, more detailed summary using beam search
# -----------------------------------------------------------
with torch.no_grad():
    long_ids = model.generate(
        inputs["input_ids"],
        max_length=120,
        min_length=60,
        num_beams=4,          # beam search for fluency
        do_sample=False,
        length_penalty=1.5,   # encourage longer output
    )
long_summary = tokenizer.decode(long_ids[0], skip_special_tokens=True)
print("\n=== Detailed Summary (beam=4, length_penalty=1.5) ===")
print(long_summary)

# -----------------------------------------------------------
# Creative summary using sampling with temperature
# -----------------------------------------------------------
with torch.no_grad():
    creative_ids = model.generate(
        inputs["input_ids"],
        max_length=80,
        min_length=30,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
    )
creative_summary = tokenizer.decode(creative_ids[0], skip_special_tokens=True)
print("\n=== Creative Summary (sampling, temp=0.8) ===")
print(creative_summary)