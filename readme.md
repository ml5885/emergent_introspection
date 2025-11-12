# Open Source Replication of Introspective Awareness Experiments

This repository contains code to replicate the introspective-awareness experiments [from Anthropic](https://transformer-circuits.pub/2025/introspection/index.html).

## Repository structure

- `experiment.py`: Runs the introspection prompt and control questions with concept-vector injection.
- `steering.py`: Utilities for concept-vector construction, activation injection, and scoring.
- `plots.py`: Code for generating plots from experiment results.

## Setup

To set up the environment, create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Experiments

By default, `experiment.py` uses GPT-2 small. It will run the main introspection experiment along with control questions. It saves the results in a timestamped directory under `runs/`.

```bash
# Simply run the command
python experiment.py
```
