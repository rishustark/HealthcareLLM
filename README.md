
## Setup

1.  **Create a project directory:**
    ```bash
    mkdir BioHealthLLM
    cd BioHealthLLM
    ```
2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r code/requirements.txt
    # You might need to install PyTorch separately based on your CUDA version
    # See: https://pytorch.org/get-started/locally/
    ```

## Methodology Overview

This project utilizes Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)**, applied via the `peft` library.

1.  **Supervised Fine-Tuning (SFT - `sft.py`):** Fine-tunes a base LLM (e.g., Llama-2, SmolLM) directly on examples of input-output pairs (e.g., medical question -> answer).
2.  **Chain-of-Thought Data Generation (`datagen.py` & `cot.py`):** Generates synthetic training data where the model produces not just the answer but also the reasoning steps (Chain-of-Thought). It filters for responses that lead to the *correct* final answer.
3.  **Rejection Sampling Fine-Tuning (RFT - `rft.py`):** Fine-tunes the base LLM using the correctly reasoned examples generated in the previous step. This aims to teach the model *how* to reason correctly for the domain.

## Usage (Example - RFT Training)

*(Adapt paths and parameters as needed)*

1.  **(Optional) Generate RFT data:**
    ```bash
    python code/datagen.py \
        --output_json code/sample_data/rft_sample.jsonl \
        --cot_checkpoint <base_model_name_or_path> \
        # Add other datagen args (oversample, temp, etc.)
    ```
2.  **Run RFT Training:**
    ```bash
    python code/rft.py train \
        --output_dir code/rft_model \
        --rft_data_path code/sample_data/rft_sample.jsonl \
        --checkpoint <base_model_name_or_path> \
        # Add other training args (epochs, batch_size, lr, lora_r, etc.)
    ```

**Note:** Successful fine-tuning requires appropriate computational resources (GPU with sufficient VRAM) and carefully curated, high-quality datasets relevant to the specific biology/healthcare task. The provided sample data is illustrative.

## Data Considerations

Working with real healthcare data requires strict adherence to privacy regulations (e.g., HIPAA, GDPR). Any real-world application **must** use properly anonymized, synthetic, or ethically sourced data. Robust anonymization and ethical review are critical.