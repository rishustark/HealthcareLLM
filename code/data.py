import json
from dataclasses import dataclass
from pathlib import Path
import math # For isnan check
from typing import Any, Union
import logging # Use logging for warnings

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming data might be in a 'sample_data' subdirectory relative to this script's location
# Or adjust DATA_DIR as needed for your project structure
PROJECT_DIR = Path(__file__).parent # This is BioHealthLLM/code/
DATA_DIR = PROJECT_DIR / "sample_data"

class Dataset:
    """Loads datasets (e.g., train, valid) from JSONL files."""
    def __init__(self, split: str, file_path: Path | None = None):
        """
        Initializes the Dataset.

        Args:
            split: The name of the split (e.g., "train_sample", "valid_sample").
            file_path: Optional direct path to the JSONL file. If None, constructs
                       path using DATA_DIR and split name (e.g., DATA_DIR / f"{split}.jsonl").
        """
        if file_path is None:
            file_path = DATA_DIR / f"{split}.jsonl"

        self.file_path = file_path
        self.data = []
        try:
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        # Assumes each line is a valid JSON object
                        # Expected format: {"question": "...", "answer": "..."}
                        # Modify this parsing if your JSONL structure is different
                        self.data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in {file_path}: {line.strip()}")
            logging.info(f"Loaded {len(self.data)} examples from {self.file_path}")
        except FileNotFoundError:
            logging.error(f"Error: Data file not found at {self.file_path}")
            # Optionally, raise the error or handle it as appropriate
            # raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Returns a dictionary representing the data sample at the given index.
        Assumes self.data contains list of dicts like {"question": ..., "answer": ...}
        """
        return self.data[idx]


def is_answer_valid(
    parsed_answer: Union[str, float],
    correct_answer: Union[str, float],
    relative_tolerance: float = 0.05 # Only used if both are floats
) -> bool:
    """
    Checks if the parsed answer is valid compared to the correct answer.
    Handles both float and string comparisons.

    Args:
        parsed_answer: The answer parsed from the model's response.
        correct_answer: The ground truth answer.
        relative_tolerance: Tolerance for float comparison.

    Returns:
        True if the answer is considered valid, False otherwise.
    """
    # Handle NaN cases first (often indicates parsing failure)
    if isinstance(parsed_answer, float) and math.isnan(parsed_answer):
        return False

    # Try float comparison if both seem numeric
    if isinstance(parsed_answer, (int, float)) and isinstance(correct_answer, (int, float)):
        try:
            p_ans_float = float(parsed_answer)
            c_ans_float = float(correct_answer)
            # Use relative tolerance for floats
            return abs(round(p_ans_float, 3) - round(c_ans_float, 3)) < relative_tolerance * abs(round(c_ans_float, 3))
        except (ValueError, TypeError):
            # Fallback to string comparison if conversion fails
            pass

    # Default to string comparison (case-insensitive, whitespace-stripped)
    # NOTE: This is a very basic check for text. Real-world text evaluation
    # often requires more sophisticated methods (e.g., ROUGE, BLEU, semantic similarity).
    try:
        return str(parsed_answer).strip().lower() == str(correct_answer).strip().lower()
    except Exception as e:
        logging.warning(f"Error during string comparison: {e}. Parsed: {parsed_answer}, Correct: {correct_answer}")
        return False


@dataclass
class BenchmarkResult:
    @dataclass
    class Sample:
        question: str
        generated_response: str # Keep the full response for inspection
        parsed_answer: Union[str, float] # What was actually parsed
        correct_answer: Union[str, float]
        is_correct: bool

    accuracy: float
    answer_rate: float  # How often was the parsed answer not NaN
    samples: list[Sample]

    @classmethod
    def from_answers(
        cls,
        parsed_answers: list[Union[str, float]],
        generated_responses: list[str], # Add original responses
        dataset: Dataset,
        questions: list[str], # Pass questions used for prompts
        max_question: int
    ) -> "BenchmarkResult":
        """
        Creates a BenchmarkResult from parsed answers and dataset.
        """
        n = min(len(dataset), max_question, len(parsed_answers))
        samples = []
        num_valid_answers = 0 # Count non-NaN answers

        for i in range(n):
            item = dataset[i] # Get the original dict {"question": ..., "answer": ...}
            correct_answer = item.get("answer", float('nan')) # Safely get correct answer
            parsed_answer = parsed_answers[i]
            is_correct = is_answer_valid(parsed_answer, correct_answer)

            samples.append(
                cls.Sample(
                    question=questions[i], # Use the actual question passed to model
                    generated_response=generated_responses[i],
                    parsed_answer=parsed_answer,
                    correct_answer=correct_answer,
                    is_correct=is_correct,
                )
            )
            if not (isinstance(parsed_answer, float) and math.isnan(parsed_answer)):
                num_valid_answers += 1

        if n == 0:
            return cls(accuracy=0.0, answer_rate=0.0, samples=[])

        return cls(
            accuracy=sum(sample.is_correct for sample in samples) / n,
            answer_rate=num_valid_answers / n,
            samples=samples,
        )


def benchmark(
    model: 'BaseLLM', # Use forward reference for type hint
    dataset: Dataset,
    max_question: int,
    generation_args: dict | None = None # Pass generation args like max_new_tokens
) -> BenchmarkResult:
    """
    Benchmarks a model on a given dataset.

    Args:
        model: The model instance (subclass of BaseLLM).
        dataset: The dataset instance to benchmark on.
        max_question: Maximum number of questions to evaluate.
        generation_args: Dictionary of arguments for model.generate/batched_generate.

    Returns:
        A BenchmarkResult object.
    """
    idx = range(min(len(dataset), max_question))
    # Extract questions directly from dataset items (assuming "question" key)
    questions = [dataset[i].get("question", "") for i in idx] # Use .get for safety
    valid_questions = [q for q in questions if q] # Filter empty questions

    if not valid_questions:
        logging.warning("No valid questions found in the dataset for benchmarking.")
        return BenchmarkResult(accuracy=0.0, answer_rate=0.0, samples=[])

    logging.info(f"Benchmarking on {len(valid_questions)} questions...")

    # Prepare generation arguments
    gen_args = generation_args or {}
    gen_args.setdefault("max_new_tokens", 150) # Default if not provided

    # Generate responses - model.answer formats prompts and calls batched_generate
    # Important: We need the raw responses AND the parsed answers
    # Let's modify the flow slightly: generate first, then parse.
    prompts = [model.format_prompt(q) for q in valid_questions]
    generated_responses = model.batched_generate(prompts, **gen_args)
    parsed_answers = [model.parse_answer(resp) for resp in generated_responses]

    # Create result using the collected info
    return BenchmarkResult.from_answers(
        parsed_answers=parsed_answers,
        generated_responses=generated_responses,
        dataset=dataset, # Pass the dataset slice used
        questions=valid_questions, # Pass the questions used
        max_question=len(valid_questions) # Use the actual number evaluated
    )


if __name__ == "__main__":
    # Example usage: Load a sample dataset and print the first item
    try:
        # Make sure sample_data/train_sample.jsonl exists and is formatted correctly
        # Example line for train_sample.jsonl:
        # {"question": "What is the function of the mitochondria?", "answer": "Mitochondria are known as the powerhouses of the cell."}
        sample_dataset = Dataset("train_sample")
        if len(sample_dataset) > 0:
            print("First item from sample dataset:")
            print(sample_dataset[0])
        else:
            print("Sample dataset loaded but is empty.")
    except Exception as e:
        print(f"Could not load or process sample dataset: {e}")
        print(f"Please ensure '{DATA_DIR / 'train_sample.jsonl'}' exists and contains valid JSONL.")
        print("Example line format: {\"question\": \"some question\", \"answer\": \"some answer\"}") 
