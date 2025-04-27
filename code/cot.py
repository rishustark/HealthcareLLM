from .base_llm import BaseLLM
import logging

class CoTModel(BaseLLM):
    """Extends BaseLLM to implement Chain-of-Thought prompting for Bio/Health tasks."""

    def format_prompt(self, question: str) -> str:
        """
        Formats a question using a chat template suitable for CoT in Bio/Health.
        Uses the model's tokenizer's chat template functionality.

        Args:
            question: The user's question.

        Returns:
            A formatted prompt string ready for the model.
        """
        # Example prompt structure for a medical QA task
        # Replace with a few-shot example more relevant to your specific task if needed.
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant specializing in biology and healthcare. Provide a step-by-step reasoning process to answer the question accurately. Conclude your reasoning with the final answer enclosed in <answer> tags.",
            },
            {
                "role": "user",
                "content": "What are the main functions of the liver?",
            },
            {
                "role": "assistant",
                "content": "The liver has many vital functions. Step 1: It produces bile, which helps digest fats. Step 2: It metabolizes nutrients absorbed from the small intestine. Step 3: It detoxifies chemicals and metabolizes drugs. Step 4: It synthesizes proteins important for blood clotting and other functions. Step 5: It stores glycogen, vitamins, and minerals. Therefore, the main functions include bile production, metabolism, detoxification, protein synthesis, and storage. <answer>The main functions of the liver include bile production, metabolism of nutrients, detoxification of harmful substances, synthesis of essential proteins, and storage of glycogen, vitamins, and minerals.</answer>",
            },
            {
                "role": "user",
                "content": question, # The actual question to be answered
            },
        ]

        try:
            # Use apply_chat_template to format the messages correctly for the model
            # tokenize=False returns a string
            # add_generation_prompt=True adds the prompt indicating it's the assistant's turn to speak
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            logging.error(f"Error applying chat template: {e}. Using basic formatting.")
            # Fallback to basic formatting if chat template fails
            # This might be less effective for chat-tuned models
            return f"System: You are a helpful AI assistant specializing in biology and healthcare. Provide a step-by-step reasoning process to answer the question accurately. Conclude your reasoning with the final answer enclosed in <answer> tags.\nUser: {question}\nAssistant:"

# Example usage (for testing the class)
def load() -> CoTModel:
    """Loads a CoTModel instance (potentially with specific checkpoint)."""
    # You might want to specify a checkpoint optimized for chat/instruction following
    # return CoTModel(checkpoint="HuggingFaceTB/SmolLM-360M-Instruct") # Example
    return CoTModel() # Uses default from BaseLLM

def test_model():
    """Tests the CoTModel with a sample question."""
    from .data import Dataset, benchmark # Use adapted data/benchmark
    import logging

    logging.info("--- Testing CoT Model ---")

    # Create a dummy validation set for testing structure
    # In reality, you'd load actual validation data
    # Example: Create a dummy valid_sample.jsonl with relevant questions
    # {"question": "What is photosynthesis?", "answer": "Photosynthesis is the process used by plants..."}
    try:
        # Assuming you have a 'valid_sample.jsonl' in sample_data
        testset = Dataset("valid_sample")
        if len(testset) == 0:
            logging.warning("Validation dataset is empty. Cannot run benchmark.")
            return
    except Exception as e:
        logging.error(f"Failed to load validation dataset for testing: {e}")
        print("Please create a `code/sample_data/valid_sample.jsonl` file for testing.")
        return

    # Load the CoT model (consider an instruction-tuned base)
    try:
        model = CoTModel(checkpoint="HuggingFaceTB/SmolLM-360M-Instruct") # Example instruct model
    except Exception as e:
        logging.error(f"Failed to load CoT model: {e}")
        print("Ensure the specified checkpoint model exists and dependencies are installed.")
        return

    # Define generation parameters if needed
    generation_args = {"max_new_tokens": 250, "temperature": 0.1}

    # Run benchmark (using adapted benchmark function)
    # Evaluate on a small number of questions, e.g., 10
    max_q = min(10, len(testset))
    if max_q == 0:
        logging.warning("No questions to benchmark.")
        return

    logging.info(f"Benchmarking on {max_q} questions...")
    try:
        benchmark_result = benchmark(model, testset, max_q, generation_args)
        print(f"Benchmark Results (on {max_q} samples):")
        print(f"Accuracy (exact match/valid format): {benchmark_result.accuracy:.4f}")
        print(f"Answer Rate (parsed successfully): {benchmark_result.answer_rate:.4f}")

        # Optionally print first few samples for qualitative check
        print("\nSample Generations:")
        for i, sample in enumerate(benchmark_result.samples[:3]): # Print first 3
            print("---")
            print(f"Q: {sample.question}")
            print(f"Expected: {sample.correct_answer}")
            print(f"Generated Response: {sample.generated_response}")
            print(f"Parsed Answer: {sample.parsed_answer}")
            print(f"Correct?: {sample.is_correct}")
            print("---")

    except Exception as e:
        logging.error(f"Error during benchmarking: {e}")

    logging.info("--- CoT Model Test Complete ---")


if __name__ == "__main__":
    import fire
    # Example: Run `python -m BioHealthLLM.code.cot test`
    fire.Fire({"test": test_model, "load": load}) 