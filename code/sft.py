import torch
from transformers import TrainingArguments, Trainer, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from pathlib import Path
import logging
import os

from .base_llm import BaseLLM
from .data import Dataset, benchmark # Assumes Dataset loads from JSONL

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_DIR = Path(__file__).parent

def format_example_sft(question: str, answer: str) -> dict[str, str]:
    """
    Formats a data sample (question, answer) into the required dictionary format.
    Keeps the <answer> tag format for consistency with original structure, but contains text.
    Args:
        question: The input question string.
        answer: The target answer string.
    Returns:
        Dictionary with "question" and "answer" keys.
    """
    # You might adjust this format based on the base model's expected input
    # For some models, a chat format might be better even for SFT.
    # Keeping simple format here for demonstration.
    # No <answer> tag wrapping needed here as the `tokenize` function handles supervision target.
    return {"question": question, "answer": answer}

def tokenize_sft(tokenizer, question: str, answer: str, max_length: int = 128):
    """
    Tokenizes a question-answer pair for SFT.

    Formats as: `question answer<eos>`
    Labels are set to -100 for the question part and the actual token IDs for the answer part.

    Args:
        tokenizer: The tokenizer instance.
        question: The question text.
        answer: The answer text.
        max_length: Maximum sequence length for padding/truncation.

    Returns:
        A dictionary containing `input_ids`, `attention_mask`, and `labels` tensors.
    """
    # Ensure pad token is set (might be done in BaseLLM init already)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Combine question and answer, add EOS token
    full_text = f"{question.strip()} {answer.strip()}{tokenizer.eos_token}"

    tokenizer.padding_side = "right" # Pad on the right for labels alignment
    full_tokenized = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Squeeze the batch dimension added by return_tensors="pt"
    input_ids = full_tokenized["input_ids"].squeeze(0)
    attention_mask = full_tokenized["attention_mask"].squeeze(0)

    # Tokenize question separately to find its length in tokens *after* potential prefix/special tokens
    # We need to mask the question part in the labels.
    # Important: Tokenize the question *exactly* as it appears at the start of `full_text`
    # (including any implicit prefix the tokenizer might add, though usually not for raw text)
    # A simple way is to tokenize the question part directly.
    question_part_text = f"{question.strip()} " # Include the space separating q/a
    question_tokenized = tokenizer(question_part_text, return_tensors="pt", add_special_tokens=False)
    question_len = question_tokenized["input_ids"].shape[1]

    # Create labels: mask out the prompt part (-100)
    labels = torch.full_like(input_ids, -100)

    # Find where padding starts (first 0 in attention mask), or use full length if no padding
    try:
        # Find first padding token index
        pad_token_idx = (attention_mask == 0).nonzero(as_tuple=True)[0][0].item()
    except IndexError:
        # No padding tokens found
        pad_token_idx = len(input_ids)

    # Label only the answer tokens (starting after question_len) up to the first pad token (or end of seq)
    # Ensure we don't label beyond the actual sequence length if shorter than max_length
    answer_start_index = question_len
    labels[answer_start_index:pad_token_idx] = input_ids[answer_start_index:pad_token_idx]

    # Ensure the EOS token *after* the answer is also labeled if it's not padding
    # This happens implicitly if pad_token_idx includes the eos token.

    # Return as a dict suitable for Trainer - ensure tensors are used
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


class SFTTokenizedDataset:
    """Applies SFT tokenization to a dataset."""
    def __init__(self, tokenizer, data: Dataset, format_fn, max_length: int = 128):
        """
        Args:
            tokenizer: Tokenizer instance.
            data: Dataset instance (from data.py, loading JSONL).
            format_fn: Function to format a raw data item (dict) into {"question": ..., "answer": ...}.
            max_length: Max sequence length for tokenization.
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_item = self.data[idx] # Assumes __getitem__ returns dict
        # Safely get question and answer
        question = raw_item.get("question", "")
        answer = raw_item.get("answer", "")
        if not question or not answer:
            logging.warning(f"Skipping item at index {idx} due to missing question or answer.")
            # Return a dummy item or handle differently? Returning first item's structure if possible.
            if idx == 0:
                 raise ValueError("First item has missing data, cannot proceed.")
            # Hacky: return the previous item? Or filter dataset beforehand.
            # Best practice: ensure data is clean before creating TokenizedDataset.
            # For now, let's try returning the tokenization of the first valid item.
            # This is not ideal but prevents crashing during training iteration.
            first_item = self.data[0]
            question = first_item.get("question", "")
            answer = first_item.get("answer", "")

        formatted_data = self.format_fn(question, answer)
        tokenized_data = tokenize_sft(self.tokenizer, **formatted_data, max_length=self.max_length)
        return tokenized_data # Directly return the dict of tensors


def train_model(
    output_dir: str = str(PROJECT_DIR / "sft_model"),
    train_split: str = "train_sample",
    checkpoint: str = "HuggingFaceTB/SmolLM-360M-Instruct", # Base model - USE INSTRUCT VERSION!
    # --- LoRA Config ---
    use_lora: bool = True,
    lora_r: int = 16, # Rank for LoRA
    lora_alpha: int = 32, # Standard practice: ~2x-4x rank
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | str = "all-linear", # Apply to all linear layers
    # --- Quantization --- (for QLoRA)
    use_quantization: bool = True, # Set to True to enable QLoRA
    # --- Training Arguments ---
    learning_rate: float = 2e-4, # Often slightly higher for PEFT
    num_train_epochs: int = 3, # Adjust as needed
    per_device_train_batch_size: int = 8, # Adjust based on GPU memory
    gradient_accumulation_steps: int = 4, # Effective batch size = batch_size * num_gpus * grad_accum
    max_length: int = 256, # Max sequence length for tokenization
    logging_steps: int = 20, # Log metrics frequency
    save_strategy: str = "epoch", # Save checkpoints every epoch
    optim: str = "paged_adamw_8bit", # Optimizer efficient for QLoRA
    gradient_checkpointing: bool = True, # Saves memory
    warmup_ratio: float = 0.1, # Warmup LR for stability
    lr_scheduler_type: str = "cosine", # Learning rate scheduler
    report_to: str = "tensorboard", # Logging backend
    run_test_after_train: bool = True, # Run evaluation after training
    **kwargs, # Allow overriding TrainingArguments via command line
):
    """Trains the SFT model using LoRA/QLoRA."""
    logging.info(f"Starting SFT training...")
    logging.info(f"Base Model: {checkpoint}")
    logging.info(f"Output Directory: {output_dir}")
    logging.info(f"Using LoRA: {use_lora}, Using QLoRA (4-bit): {use_quantization}")
    if use_lora:
        logging.info(f"LoRA Config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, target={lora_target_modules}")
    logging.info(f"Training Params: epochs={num_train_epochs}, lr={learning_rate}, batch_size={per_device_train_batch_size}, grad_accum={gradient_accumulation_steps}")

    # --- Load Model and Tokenizer --- #
    # Load base model with optional quantization
    quantization_config = None
    model_kwargs = {}
    if use_quantization:
        logging.info("Loading model with 4-bit quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # Required for QLoRA: use device_map="auto" for bitsandbytes
        model_kwargs["device_map"] = "auto"

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=quantization_config,
            trust_remote_code=True, # Allow custom code if needed by model
            **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    except Exception as e:
        logging.error(f"Failed to load base model or tokenizer from '{checkpoint}': {e}")
        return

    # --- Configure Tokenizer --- #
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = base_model.config.eos_token_id
    tokenizer.padding_side = "right" # Important for label alignment

    # --- Prepare Model for PEFT --- #
    if use_quantization:
        # Prepare model for k-bit training (necessary for QLoRA)
        base_model = prepare_model_for_kbit_training(base_model)

    model = base_model # Start with base or quantized base
    if use_lora:
        logging.info("Applying LoRA adapter...")
        # Configure LoRA
        if isinstance(lora_target_modules, str) and lora_target_modules == "all-linear":
             # Find all linear layers for targeting
             import bitsandbytes as bnb
             cls = bnb.nn.Linear4bit if use_quantization else torch.nn.Linear
             target_modules_found = set()
             for name, module in base_model.named_modules():
                 if isinstance(module, cls):
                     names = name.split('.')
                     target_modules_found.add(names[-1])
             if 'lm_head' in target_modules_found:
                 target_modules_found.remove('lm_head') # Often exclude lm_head
             lora_target_modules = list(target_modules_found)
             logging.info(f"Targeting LoRA for modules: {lora_target_modules}")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Add LoRA adapter to the model
        try:
            model = get_peft_model(model, lora_config)
            logging.info("LoRA adapter applied successfully.")
            model.print_trainable_parameters()
        except Exception as e:
            logging.error(f"Failed to apply LoRA adapter: {e}")
            return
    else:
        logging.warning("Proceeding with full fine-tuning (no LoRA). Requires significant resources.")
        # Ensure model is trainable if not using LoRA
        for param in model.parameters():
             param.requires_grad = True

    # --- Load and Tokenize Data --- #
    try:
        train_dataset_raw = Dataset(train_split)
        if len(train_dataset_raw) == 0:
             logging.error(f"Training dataset split '{train_split}' is empty.")
             return
        tokenized_train_dataset = SFTTokenizedDataset(tokenizer, train_dataset_raw, format_example_sft, max_length=max_length)
    except Exception as e:
        logging.error(f"Failed to load or tokenize training data: {e}")
        return

    # --- Configure Training --- #
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        report_to=report_to,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        optim=optim if use_quantization else "adamw_torch", # Use paged adamw only for QLoRA
        gradient_checkpointing=gradient_checkpointing,
        fp16=not use_quantization and torch.cuda.is_available(), # Use fp16 if not quantizing and on CUDA
        bf16=False, # Set to True if Ampere or newer GPU and want to use bfloat16
        max_grad_norm=1.0, # Gradient Clipping
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        remove_unused_columns=False, # Important for PEFT
        # device_map="auto" handled in model loading for QLoRA
        # Use FSDP for multi-GPU full finetuning if needed
        **kwargs # Pass extra args
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer, # Pass tokenizer for saving
        # Data collator is implicitly handled for dict datasets
    )

    # --- Train --- #
    logging.info("Starting training process...")
    try:
        trainer.train()
        logging.info("Training finished successfully.")
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True) # Log traceback
        return

    # --- Save Final Model --- #
    logging.info(f"Saving final model adapter (or full model) to {output_dir}")
    # `save_model` saves the adapter config/weights if using PEFT,
    # or the full model if not.
    try:
        trainer.save_model(output_dir)
        # Ensure tokenizer is saved alongside the adapter/model
        tokenizer.save_pretrained(output_dir)
        logging.info("Model and tokenizer saved.")
    except Exception as e:
        logging.error(f"Error saving final model/tokenizer: {e}")

    # Reminder for cleanup if checkpoints were saved
    if save_strategy == "steps" or save_strategy == "epoch":
        print("\nREMINDER: Manually delete intermediate 'checkpoint-XXX' folders")
        print(f"inside '{output_dir}' before distribution if desired.\n")

    # --- Optional: Run Test --- #
    if run_test_after_train:
        logging.info("Running evaluation on validation set after training...")
        test_model(ckpt_path=output_dir, base_checkpoint=checkpoint, use_quantization_base=use_quantization)


def load_sft_model(ckpt_path: str, base_checkpoint: str, use_quantization_base: bool = True) -> tuple[PeftModel | AutoModelForCausalLM, AutoTokenizer]:
    """Loads a fine-tuned SFT model (PEFT adapter or full model)."""
    logging.info(f"Loading SFT model from: {ckpt_path}")
    logging.info(f"Using base checkpoint: {base_checkpoint}")

    # Load tokenizer first
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    except Exception as e:
        logging.error(f"Failed to load tokenizer from {ckpt_path}: {e}")
        raise

    # Check if it's a PEFT adapter or a full model save
    adapter_config_path = Path(ckpt_path) / "adapter_config.json"
    is_peft_adapter = adapter_config_path.exists()

    quantization_config = None
    model_kwargs = {}
    if use_quantization_base:
        logging.info("Loading base model with 4-bit quantization for PEFT merge.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    # Load the base model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_checkpoint,
            quantization_config=quantization_config,
            trust_remote_code=True,
            **model_kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            base_model.config.pad_token_id = base_model.config.eos_token_id

    except Exception as e:
        logging.error(f"Failed to load BASE model '{base_checkpoint}': {e}")
        raise

    if is_peft_adapter:
        logging.info("Found PEFT adapter. Loading and merging...")
        try:
            # Load the PEFT model (adapter) on top of the (potentially quantized) base model
            model = PeftModel.from_pretrained(base_model, ckpt_path)
            # Optional: Merge adapter into the base model for faster inference if needed
            # model = model.merge_and_unload()
            logging.info("PEFT adapter loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading PeftModel adapter from {ckpt_path}: {e}")
            raise
    else:
        logging.info("No adapter_config.json found. Assuming full model save.")
        # If no adapter, we assume the ckpt_path contains the full fine-tuned model
        # Reload the model from ckpt_path (this might require significant memory)
        try:
            # Release quantized base model memory first if applicable
            del base_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            model = AutoModelForCausalLM.from_pretrained(
                 ckpt_path,
                 device_map="auto" # Or specify device
            )
            logging.info("Full fine-tuned model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading full fine-tuned model from {ckpt_path}: {e}")
            raise

    model.eval() # Set to evaluation mode
    logging.info(f"Model loaded and set to eval mode. Final type: {model.__class__.__name__}")
    return model, tokenizer

def test_model(ckpt_path: str, base_checkpoint: str, valid_split: str = "valid_sample", use_quantization_base: bool = True):
    """Tests the trained SFT model on a validation set."""
    logging.info(f"\n--- Starting SFT Model Test --- Ckpt: {ckpt_path}")
    try:
        model, tokenizer = load_sft_model(ckpt_path, base_checkpoint, use_quantization_base)

        # Wrap the loaded model/tokenizer in BaseLLM for benchmark compatibility
        # This is a bit of a workaround, ideally benchmark takes model/tokenizer directly
        llm_wrapper = BaseLLM(checkpoint=base_checkpoint) # Init BaseLLM doesn't load model here
        llm_wrapper.model = model
        llm_wrapper.tokenizer = tokenizer
        llm_wrapper.device = model.device # Get device from loaded model

    except Exception as e:
        logging.error(f"Failed to load SFT model for testing: {e}")
        return

    # Load validation data
    try:
        testset = Dataset(valid_split)
        if len(testset) == 0:
             logging.warning(f"Validation dataset '{valid_split}' is empty. Cannot run benchmark.")
             return
    except Exception as e:
        logging.error(f"Failed to load validation dataset '{valid_split}': {e}")
        return

    # Benchmark
    logging.info(f"Benchmarking model on {valid_split}...")
    # Define generation parameters for testing (might differ from training)
    generation_args = {"max_new_tokens": 150, "temperature": 0.1, "do_sample": False}
    max_q = min(100, len(testset)) # Benchmark up to 100 samples
    if max_q == 0: return

    try:
        benchmark_result = benchmark(llm_wrapper, testset, max_q, generation_args)
        print(f"\n--- SFT Benchmark Results (on {max_q} samples from {valid_split}) ---")
        print(f"Accuracy (exact match/valid format): {benchmark_result.accuracy:.4f}")
        print(f"Answer Rate (parsed successfully): {benchmark_result.answer_rate:.4f}")
        print("-------------------------------------------------")

        # Optionally print first few samples
        # ... (similar to cot.py test_model sample printing)

    except Exception as e:
        logging.error(f"Error during SFT benchmarking: {e}")

    logging.info("--- SFT Model Test Complete ---")


if __name__ == "__main__":
    import fire
    # Needs arguments passed correctly, e.g.:
    # python -m BioHealthLLM.code.sft train --num_train_epochs 1 --per_device_train_batch_size 4
    # python -m BioHealthLLM.code.sft test --ckpt_path="BioHealthLLM/code/sft_model" --base_checkpoint="HuggingFaceTB/SmolLM-360M-Instruct"
    fire.Fire({"train": train_model, "test": test_model}) 