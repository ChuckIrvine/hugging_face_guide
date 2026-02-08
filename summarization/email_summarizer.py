"""
email_summarizer.py
Summarizes email threads using T5 with direct model/tokenizer usage.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# -----------------------------------------------------------
# Device detection: prefer Apple MPS GPU, then CUDA, else CPU
# -----------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# -----------------------------------------------------------
# Load T5-small model and tokenizer
# -----------------------------------------------------------
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# -----------------------------------------------------------
# Sample email thread to summarize
# -----------------------------------------------------------
email_thread = """
From: Sarah Chen <sarah@example.com>
To: Dev Team
Subject: Q3 Sprint Planning

Hi team,

I wanted to follow up on our sprint planning discussion from Monday. After reviewing
the backlog, I propose we prioritize the authentication overhaul (JIRA-1042) and the
database migration (JIRA-1078) for the first two weeks. The API rate limiting feature
(JIRA-1095) can move to the second half of the sprint.

Also, DevOps confirmed that the new staging environment will be ready by Thursday.
Please plan your integration tests accordingly.

Let me know if there are any blockers.

Best,
Sarah

---

From: Marcus Johnson <marcus@example.com>
To: Dev Team
Subject: Re: Q3 Sprint Planning

Sarah,

Sounds good. One concern: JIRA-1078 depends on the new ORM version, which has a
known issue with our Postgres setup. I've filed a patch upstream but we might need
a workaround. I'll have a status update by Wednesday.

For JIRA-1042, I can start immediately. Estimated effort is 5 story points.

- Marcus
"""

# -----------------------------------------------------------
# Prepend T5 task prefix and tokenize the input
# -----------------------------------------------------------
input_text = "summarize: " + email_thread
input_ids = tokenizer.encode(
    input_text,
    return_tensors="pt",
    max_length=512,
    truncation=True,
).to(device)

# -----------------------------------------------------------
# Generate summary with beam search
# -----------------------------------------------------------
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=100,
        min_length=20,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True,
    )

# -----------------------------------------------------------
# Decode and print the summary
# -----------------------------------------------------------
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n=== Email Thread Summary ===")
print(summary)