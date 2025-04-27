import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import math # Import math for isnan check

class BaseLLM:
    """Base class for handling LLM loading, tokenization, and generation."""
    def __init__(
        self,
        checkpoint: str = "HuggingFaceTB/SmolLM-360M", # Example, replace with desired model
        device: str | None = None,
        use_quantization: bool = False, # Set to True to use 4-bit quantization
    ):
        """
        Initializes the BaseLLM.

        Args:
            checkpoint: Path or name of the pre-trained model checkpoint.
            device: Device to load the model onto ('cuda', 'mps', 'cpu', or None for auto).
            use_quantization: Whether to load the model with 4-bit quantization (QLoRA).
        """
        self.checkpoint = checkpoint
        self.device = device or self._get_default_device()
        print(f"Using device: {self.device}")

        quantization_config = None
        if use_quantization:
            print("Using 4-bit quantization (QLoRA compatible).")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, # Or bfloat16 if supported
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=quantization_config,
            # torch_dtype=torch.float16, # Use float16 or bfloat16 if device supports and not quantizing fully
            # device_map="auto" # Use device_map='auto' for multi-GPU or large models without quantization
        )
        # Move model to device only if not using quantization's device_map or bitsandbytes handling
        if not quantization_config: # Bitsandbytes handles device placement
             self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # Set pad token if not present (common for some models like Llama)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.eval() # Set model to evaluation mode by default

    def _get_default_device(self) -> str:
        """Determines the default device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def format_prompt(self, question: str) -> str:
        """
        Basic prompt formatting. Override this in subclasses (like CoTModel).
        Example: Simple instruction format.
        """
        # This is a placeholder. SFT/RFT might format differently or use tokenizers directly.
        # CoTModel will override this with a specific chat template.
        return f"Question: {question}\\nAnswer:"


    def parse_answer(self, response: str) -> str | float:
        """
        Parses the final answer from the model's response string.
        Assumes the answer is within <answer>...</answer> tags.
        Tries to convert to float if possible, otherwise returns string.
        Returns float('nan') if parsing fails or tag is missing.
        """
        match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if match:
            answer_str = match.group(1).strip()
            try:
                # Try converting to float for potential numerical tasks
                return float(answer_str)
            except ValueError:
                # If not a float, return the string content
                return answer_str
        else:
            # Optional: Attempt to extract final number if no tag found (legacy/fallback)
            # This is less reliable for text-based answers
            # match_num = re.findall(r"[-+]?\d*\.\d+|\d+", response)
            # if match_num:
            #     try:
            #         return float(match_num[-1])
            #     except ValueError:
            #         pass # Cannot convert last number
            return float('nan') # Indicate failure to parse required tag format


    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """
        Generates a response for a single prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id, # Use EOS token for padding in generation
            **kwargs
        )
        # Decode only the newly generated tokens
        response = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    @torch.inference_mode()
    def batched_generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        micro_batch_size: int = 8, # Add micro-batching for large prompt lists
        **kwargs
    ) -> list[str]:
        """
        Generates responses for a list of prompts with micro-batching.
        """
        responses = []
        print(f"Generating responses for {len(prompts)} prompts with micro-batch size {micro_batch_size}...")
        for i in range(0, len(prompts), micro_batch_size):
            batch_prompts = prompts[i:i + micro_batch_size]
            # Ensure tokenizer pads to the left for batch generation consistency
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            self.tokenizer.padding_side = "right" # Reset padding side

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

            # Decode responses, skipping prompt tokens and special tokens
            for j, output in enumerate(outputs):
                 prompt_len = inputs.input_ids[j].shape[0]
                 decoded_response = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                 responses.append(decoded_response.strip())

            # Optional: Clear cache if memory is tight between micro-batches
            if self.device == 'cuda':
                 torch.cuda.empty_cache()
            elif self.device == 'mps':
                 torch.mps.empty_cache()

        return responses


    def answer(self, *questions: str, max_new_tokens: int = 100, **kwargs) -> list[str | float]:
        """
        Generates and parses answers for multiple questions.
        """
        prompts = [self.format_prompt(q) for q in questions]
        responses = self.batched_generate(prompts, max_new_tokens=max_new_tokens, **kwargs)
        # Parse using the potentially overridden parse_answer method
        parsed_answers = [self.parse_answer(resp) for resp in responses]
        # Filter out NaN answers if needed by caller, but return list including them here
        # cleaned_answers = [a for a in parsed_answers if not (isinstance(a, float) and math.isnan(a))]
        return parsed_answers 