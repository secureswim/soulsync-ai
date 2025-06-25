# Mental Health Summarization using Fine-Tuned BART

## Overview
This project focuses on the abstractive summarization of mental health journal entries—specifically, posts expressing symptoms of depression. The goal was to build a domain-specific summarizer that generates concise, non-hallucinated, and emotionally aligned summaries for use in therapeutic interfaces, self-reflection tools, or AI-assisted journaling systems.

General-purpose summarizers like Pegasus, BART, and T5 often produce verbose, inaccurate, or emotionally tone-deaf results when applied to such sensitive content. This work addresses those limitations through custom dataset creation, semantic quality filtering, and task-specific fine-tuning.

## Motivation
Experiments with off-the-shelf models showed the following:
- Pegasus, trained on news data, frequently hallucinated or fabricated information.
- T5 produced verbose summaries lacking emotional fidelity.
- BART generalized better but still performed poorly out-of-the-box.

As existing datasets like MENTSUM are not publicly available, a new dataset was created. Pegasus was initially used to generate summaries, which were then filtered for quality using semantic similarity before being used to fine-tune a BART model.

## Methodology
### 1. Dataset Creation
- Source: Reddit (r/depression) posts.
- Filtering: Removed deleted/removed posts, short/long texts, duplicates.
- Summary Generation: Used `google/pegasus-xsum` on 500 posts.

### 2. Semantic Filtering
- Sentence embeddings via `all-MiniLM-L6-v2`.
- Cosine similarity filtering to retain grounded, faithful summaries (threshold ≥ 0.6).

### 3. Fine-Tuning BART
- Model: `facebook/bart-base` chosen due to MENTSUM paper benchmarks.
- Used filtered dataset (~76 text-summary pairs).
- Trained using Hugging Face Transformers for 8 epochs.
- Summaries truncated/padded to 128 tokens.
- Evaluation with ROUGE.

## Why MiniLM for Filtering?
BERT/BART embeddings are stronger but memory-intensive. MiniLM is fast and light (~384-dim), perfect for semantic filtering on limited compute resources.

## Project Pipeline
Reddit Posts → Data Cleaning → Pegasus Summarization → Cosine Similarity Filtering → BART Fine-Tuning

## Example Output
**Input:**  
"I've noticed that a lot of the time when I'm depressed, I get suicidal thoughts... I love my dog and my parents and wouldn't want to hurt them... I just wish it would all stop."

**Summary (BART):**  
"The user experiences suicidal thoughts during depression but refrains due to love for their family and dog."

## Limitations
- Dataset size (~76 samples) limits generalization.
- Fine-tuned model produces shallow summaries; room for deeper context understanding.
- ROUGE used in absence of human gold summaries.

## Future Work
- Complete the sentiment analysis module and integrate with summarization.
- Develop full-stack journaling app with login, journal view, and mood tracking.
- Visualize user mood over time with Chart.js or similar tools.
- Offer real-time coping suggestions based on negative sentiment trends.
- Add authentication and deploy to a public domain (Render/Vercel).
- Train on more diverse or manually curated mental health entries.
- Explore multi-task models that combine summarization + emotion classification.

## Installation
```bash
pip install pandas datasets transformers sentence-transformers scikit-learn evaluate
```

## Model Access
Hugging Face Model: https://huggingface.co/marku007/bart-mental-health-summarizer
