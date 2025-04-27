import json
from tqdm import tqdm
import logging
import re
import math

from .cot import CoTModel
from .data import Dataset, is_answer_valid, DATA_DIR, PROJECT_DIR

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Note: format_float_answer removed as answers are expected to be more general text now.
# If specific numerical formatting is needed, it should be reimplemented based on requirements.

def generate_dataset(
    output_jsonl: str = str(DATA_DIR / "rft_sample.jsonl"),
    train_split: str = "train_sample", # Which split of the Dataset to use as input
    oversample: int = 5, # Generate fewer samples per question by default for text
    temperature: float = 0.5,
    cot_checkpoint: str = "HuggingFaceTB/SmolLM-360M-Instruct", # Use instruct model
    generation_batch_size: int = 8, # Smaller batch size might be needed for longer text
    max_new_tokens_gen: int = 300, # Allow longer reasoning/answers
    debug_limit: int = 2 # Print debug info for the first N questions
):
    """
    Generates a dataset for RFT by sampling multiple CoT answers for questions
    from the training data and keeping only those where the reasoning leads to
    a correct final answer (matching the ground truth from the training data).

    Args:
        output_jsonl: Path to save the generated RFT data (JSON Lines format).
        train_split: The name of the training data split to load (e.g., "train_sample").
        oversample: How many candidate responses to generate per question.
        temperature: Sampling temperature for generation.
        cot_checkpoint: Checkpoint for the CoT model used for generation.
        generation_batch_size: Micro-batch size for generation.
        max_new_tokens_gen: Max new tokens for the generated response.
        debug_limit: Print detailed logs for the first N questions.
    """
    logging.info(f"Generating RFT dataset using CoT model: {cot_checkpoint}")
    logging.info(f"Oversampling: {oversample}, Temperature: {temperature}")
    logging.info(f"Generation Batch Size: {generation_batch_size}, Max New Tokens: {max_new_tokens_gen}")
    logging.info(f"Output will be saved to: {output_jsonl}")

    # Load the base training data (e.g., questions and correct answers)
    try:
        train_dataset = Dataset(train_split)
        if len(train_dataset) == 0:
            logging.error(f"Training dataset split '{train_split}' is empty. Cannot generate RFT data.")
            return
    except Exception as e:
        logging.error(f"Failed to load training dataset split '{train_split}': {e}")
        return

    # Load the CoT model
    try:
        cot_model = CoTModel(checkpoint=cot_checkpoint)
    except Exception as e:
        logging.error(f"Failed to load CoT model from checkpoint '{cot_checkpoint}': {e}")
        return

    rft_data = []
    num_success = 0

    # Prepare questions and correct answers from the dataset
    # Assumes dataset items are dicts with "question" and "answer" keys
    questions = [item.get("question", "") for item in train_dataset]
    correct_answers = [item.get("answer", None) for item in train_dataset]

    # Filter out items where question or answer is missing
    valid_indices = [i for i, (q, a) in enumerate(zip(questions, correct_answers)) if q and a is not None]
    if not valid_indices:
        logging.error("No valid question-answer pairs found in the training data.")
        return

    questions = [questions[i] for i in valid_indices]
    correct_answers = [correct_answers[i] for i in valid_indices]
    original_dataset_len = len(train_dataset)
    logging.info(f"Processing {len(questions)} valid question-answer pairs from the original {original_dataset_len} samples.")

    # Format the prompts using the CoTModel's format_prompt method
    logging.info("Formatting prompts...")
    formatted_prompts = [cot_model.format_prompt(q) for q in tqdm(questions, desc="Formatting prompts")]

    # Generate multiple responses per question using batched generation
    logging.info("Generating candidate answers...")
    generated_responses = cot_model.batched_generate(
        formatted_prompts,
        num_return_sequences=oversample,
        temperature=temperature,
        micro_batch_size=generation_batch_size,
        max_new_tokens=max_new_tokens_gen,
        do_sample=True, # Ensure sampling is enabled if temperature < 1.0
        top_k=50, # Add top_k and top_p for better sampling
        top_p=0.95
    ) # Returns list[list[str]] - outer list matches prompts, inner has `oversample` responses

    logging.info("Filtering for correct answers...")
    # Iterate through original questions and their generated responses
    processed_count = 0
    for i, (question, correct_answer) in enumerate(tqdm(zip(questions, correct_answers), total=len(questions), desc="Filtering answers")):
        found_correct_reasoning = False
        response_batch_index = i * oversample
        responses_for_question = generated_responses[response_batch_index : response_batch_index + oversample]

        if not responses_for_question:
             logging.warning(f"No responses generated for question index {i}. Skipping.")
             continue # Skip this question if responses are missing

        for response_idx, response in enumerate(responses_for_question):
            # Parse the final answer from the generated CoT response string
            parsed_answer = cot_model.parse_answer(response)

            # --- Debugging --- #
            if i < debug_limit:
                print("-" * 20)
                print(f"DEBUG (Q: {i+1}/{len(questions)})")
                print(f"Q: {question}")
                print(f"Expected Answer: {correct_answer} (Type: {type(correct_answer)})")
                # print(f"Formatted Prompt (start): {formatted_prompts[i][:200]}...") # Can be long
                print(f"Candidate Response {response_idx+1}/{oversample}: {response[:500]}...") # Limit long responses
                print(f"Parsed Answer from Response: {parsed_answer} (Type: {type(parsed_answer)})")

                # Check if parsed answer is valid (not NaN) and matches correct answer
                is_valid = False
                if isinstance(parsed_answer, float) and math.isnan(parsed_answer):
                    print(f"Is Valid Check: Failed (Parsed is NaN)")
                else:
                    is_valid = is_answer_valid(parsed_answer, correct_answer)
                    print(f"Is Valid Check Result: {is_valid}")
                print("-" * 20)
            # --- End Debugging --- #

            # Check validity (not NaN and matches ground truth)
            is_valid_check = False
            if not (isinstance(parsed_answer, float) and math.isnan(parsed_answer)):
                 is_valid_check = is_answer_valid(parsed_answer, correct_answer)

            if is_valid_check:
                # Found a response chain that resulted in the correct answer.
                # Store the original question, the correct answer, and the full reasoning response.
                # We use the full 'response' string as the target for RFT training.
                rft_data.append({
                    "question": question,
                    "correct_answer": correct_answer, # Keep for reference if needed
                    "reasoning_answer": response # The full generated text is the target
                })
                num_success += 1
                found_correct_reasoning = True
                break # Move to the next question once one good reasoning chain is found

        processed_count += 1
        # Optional: Log if no correct reasoning was found for a question after trying all samples
        # if not found_correct_reasoning:
        #     logging.debug(f"No valid reasoning found for question: {question}")

    logging.info(f"Dataset generation complete. Found correct reasoning for {num_success}/{processed_count} questions.")
    logging.info(f"Total RFT examples generated: {len(rft_data)}")
    logging.info(f"Saving dataset to {output_jsonl}...")

    # Save the generated data as JSON Lines
    try:
        with open(output_jsonl, "w") as f:
            for entry in rft_data:
                json.dump(entry, f)
                f.write("\n")
        logging.info("RFT dataset saved successfully.")
    except IOError as e:
        logging.error(f"Failed to write RFT dataset to {output_jsonl}: {e}")


if __name__ == "__main__":
    import fire
    # Example: python -m BioHealthLLM.code.datagen --oversample 3 --debug_limit 1
    fire.Fire(generate_dataset) 