import time
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template

import json
import torch
import logging
import gc
import numpy as np

from dataclasses import dataclass, field

from transformers.tokenization_utils import PreTrainedTokenizer

from .judge_types import JudgeConfig
from ..datatypes import EvaluationResult, ReasoningSample
from ..utilities import is_main_process


logger = logging.getLogger(name=__name__)


@dataclass
class LLMJudge:
    judge_config: JudgeConfig
    model: FastLanguageModel|None = field(init=False, default=None, repr=False)
    tokenizer: PreTrainedTokenizer|None = field(init=False, default=None, repr=False)
    use_deepspeed: bool=True

    SYSTEM_PROMPT: str = field(
        init=False,
        default=(
            "You are an expert code security analyst evaluating reasoning quality for vulnerability detection."
            "Task: Rate the quality of the reasoning provided for identifying vulnerabilities in the given C code."
        ),
        repr=False,
    )

    PROMPT_SKELETON: str = field(
        init=False,
        default=(
            "**Metadata information:**\n"
            "Source project: {project}\n"
            "Ground truth label: {target} (0=safe, 1=vulnerable)\n\n"

            "Relevant CWEs:\n{cwe_info}\n\n"

            "**C Function of reference:**\n"
            "--- CODE START ---\n"
            "```c\n{func}\n```\n\n"
            "--- CODE END ---\n\n"

            "**Reasoning to evaluate:**\n"
            "{reasoning}\n\n"

            "**Evaluation Criteria**:\n"
            "Evaluate the reasoning across the following dimensions (each scored 0-1):\n\n"

            "1. **Correctness** (0-1): Does the reasoning correctly identify vulnerabilities and match the ground truth label?\n"
            "   - 0.0-0.3: Incorrect conclusion or misses critical vulnerabilities\n"
            "   - 0.4-0.6: Partially correct but with significant errors (including vulnerbilities not explicitly present)\n"
            "   - 0.7-0.9: Mostly correct with minor issues\n"
            "   - 1.0: Fully correct identification and conclusion\n\n"

            "2. **Completeness** (0-1): Are all relevant security issues and CWEs covered?\n"
            "   - 0.0-0.3: Major vulnerabilities or CWEs missing\n"
            "   - 0.4-0.6: Some issues covered but incomplete or detected vulnerabilities that are not clearly present\n"
            "   - 0.7-0.9: Most issues covered with minor omissions\n"
            "   - 1.0: Comprehensive coverage of all relevant issues\n\n"

            "3. **Clarity** (0-1): Is the reasoning clear, well-structured, and easy to follow?\n"
            "   - 0.0-0.3: Confusing, poorly structured, hard to understand\n"
            "   - 0.4-0.6: Understandable but could be clearer\n"
            "   - 0.7-0.9: Clear and well-organized\n"
            "   - 1.0: Exceptionally clear and well-structured\n\n"

            "4. **Technical Accuracy** (0-1): Are technical details, vulnerability patterns, and references accurate?\n"
            "   - 0.0-0.3: Contains significant technical errors\n"
            "   - 0.4-0.6: Mostly accurate but with some mistakes\n"
            "   - 0.7-0.9: Accurate with minor issues\n"
            "   - 1.0: Technically flawless\n\n"

            "5. **Logical Flow** (0-1): Does the reasoning follow a logical progression from analysis to conclusion?\n"
            "   - 0.0-0.3: Disjointed or illogical progression\n"
            "   - 0.4-0.6: Somewhat logical but with gaps\n"
            "   - 0.7-0.9: Good logical flow with minor issues\n"
            "   - 1.0: Perfect logical progression\n\n"

            "**Output Format:**\n"
            "Provide your evaluation in the following JSON format:\n\n"
            "```json\n"
            "{{\n"
            '  "quality_score": <float 0-1>,  // OVERALL quality score (weighted combination of criteria above)\n'
            '  "correctness": <float 0-1>,  // Criterion 1 score\n'
            '  "completeness": <float 0-1>,  // Criterion 2 score\n'
            '  "clarity": <float 0-1>,  // Criterion 3 score\n'
            '  "technical_accuracy": <float 0-1>,  // Criterion 4 score\n'
            '  "logical_flow": <float 0-1>,  // Criterion 5 score\n'
            '  "confidence": <float 0-1>,  // How confident are you in this evaluation? (0=not confident, 1=very confident)\n'
            '  "justification": "<string>"  // Brief explanation (2-3 sentences) justifying the quality_score\n'
            "}}\n"
            "```\n\n"

            "**Important Notes:**\n"
            "- quality_score should be a weighted combination reflecting overall quality (not just an average)\n"
            "- confidence reflects your certainty in the evaluation (low if reasoning is ambiguous)\n"
            "- justification should be concise and factual, highlighting key strengths/weaknesses\n"
            "- Output ONLY valid JSON, no additional text before or after\n"
        ).strip(),
        repr=False,
    )

    def __post_init__(self):
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate constructor inputs."""

        self.model_name = self.judge_config.model_name
        self.max_seq_length = self.judge_config.max_seq_length
        self.max_new_tokens = self.judge_config.max_new_tokens

        # Validate numeric parameters
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")

        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")

        if self.max_new_tokens > self.max_seq_length:
            logger.warning(
                f"⚠️  max_new_tokens ({self.max_new_tokens}) > max_seq_length ({self.max_seq_length}). "
                f"This may cause issues during generation."
            )

    def _set_padding_strategy(self):
        """Set padding side based on model architecture.

        - Decoder-only (GPT, Llama): Left padding (for batch generation)
        - Encoder-decoder (T5, BART): Right padding
        """

        decoder_only_models: set[str] = { "llama", "mistral", "qwen", "opt", "phi", "gemma"}
        model_type = getattr(self.model.config, "model_type", "").lower()

        if any(arch in model_type for arch in decoder_only_models):
            self.tokenizer.padding_side = "left"
            logger.info(f"📍 Set padding_side='left' for decoder-only model ({model_type})")
        else:
            self.tokenizer.padding_side = "right"
            logger.info(f"📍 Set padding_side='right' for encoder-decoder model ({model_type})")

        # ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"🔧 Set pad_token to eos_token: {self.tokenizer.eos_token}")

    def _configure_tokenizer(self):
        """Configure tokenizer settings (chat template, padding, special tokens)."""

        chat_template = self.judge_config.chat_template
        if chat_template is not None:
            logger.info(f"🎨 Applying chat template: {chat_template}")
            try:
                self.tokenizer = get_chat_template(
                    self.tokenizer, chat_template=chat_template
                )
            except ValueError as e:
                logger.error(f"Invalid chat template: {chat_template}")
                logger.error(f"Available templates: {list(CHAT_TEMPLATES.keys())}")
                raise ValueError(f"Chat template '{chat_template}' not found") from e

        self._set_padding_strategy()

    def load(self):
        """Loads the quantized base model and applies fine-tuned LoRA adapter for inference.

        Process:
            1. Extract base model name from adapter config
            2. Load base model in 4-bit quantization
            3. Apply LoRA adapter weights
            4. Enable Unsloth fast inference mode
            5. Configure tokenizer (chat template, padding)
        """

        if self.model is not None:
            logger.warning("Model already loaded, skipping...")
            return

        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=(torch.bfloat16 if is_bfloat16_supported() else None),
                load_in_4bit=True,
                device_map="auto" if not self.use_deepspeed else None,
                attn_implementation="flash_attention_2",
            )

            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model or tokenizer failed to load")

            logger.info("✅ Base model and tokenizer loaded successfully")

            FastLanguageModel.for_inference(model=self.model)
            logger.info("⚡ Unsloth fast inference mode enabled (2x speedup)")

            self._configure_tokenizer()

        except torch.cuda.OutOfMemoryError as oom:
            logger.critical("💥 OUT OF MEMORY!")
            logger.critical(f"Model requires more VRAM than available.")
            if is_main_process():
                print(f"Solutions:")
                print(f"  1. Use a smaller model")
                print(f"  2. Reduce max_seq_length (current: {self.max_seq_length})")
                print(f"  3. Use 8-bit quantization instead of 4-bit")
            torch.cuda.empty_cache()
            raise oom

        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}")
            raise

    def unload(self):
        """Free GPU memory"""

        if self.model is None: return
        logger.info(f"Unloading {self.model_name}")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

        time.sleep(10) # give time for resources to be released

    def _create_judging_prompt(self, sample: ReasoningSample) -> list[dict[str,str]]:
        """Create prompt for judging reasoning quality"""

        cwe_info = (
            "\n".join([f"- {cwe}: {desc}" for cwe, desc in zip(sample.cwe, sample.cwe_desc)])
            if bool(sample.target)
            else "None"
        )

        prompt = self.PROMPT_SKELETON.format(
            project=sample.project,
            target=sample.target,
            cwe_info=cwe_info,
            func=sample.func,
            reasoning=sample.reasoning,
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT}, # is it necessary?
            {"role": "user", "content": prompt},
        ]

    def evaluate(self, sample: ReasoningSample) -> EvaluationResult:
        """Performs inference on a single C code snippet using the CoT format.

        Parameters
        ----------
        c_code_input : str
            The raw C function code to be analyzed.

        Returns
        -------
        str
            The model's generated reasoning and final answer (assistant response only).

        Raises
        ------
        RuntimeError
            If model/tokenizer not loaded or if generation fails.
        """

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before running inference.")


        messages = self._create_judging_prompt(sample=sample)
        input_texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True, # mandatory for generation
        )

        inputs = self.tokenizer(
            input_texts, return_tensors="pt", padding=True,
            max_length=self.max_seq_length, truncation=True,
        ).to(self.model.device)

        # Generate response
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=self.judge_config.temperature,
                    top_p=self.judge_config.top_p,
                    min_p=self.judge_config.min_p,
                    repetition_penalty=self.judge_config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Model generation failed: {e}") from e

        generated_tokens = outputs[:, inputs.input_ids.shape[1] :]
        response = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, cleanup_tokenization_spaces=True
        )[0]

        return self._parse_response(response)

    def _parse_response(self, response: str) -> EvaluationResult:
        """Parse judge response into structured EvaluationResult.

        Parameters
        ----------
        response : str
            Raw text response from the judge model

        Returns
        -------
        EvaluationResult
            Parsed and validated evaluation result
        """

        pretty_name = self.judge_config.model_name.split("/")[1]
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            # Validate and clip all scores to [0, 1]
            return EvaluationResult(
                judge_name=pretty_name,
                quality_score=float(np.clip(result.get("quality_score", 0.5), 0, 1)),
                correctness=float(np.clip(result.get("correctness", 0.5), 0, 1)),
                completeness=float(np.clip(result.get("completeness", 0.5), 0, 1)),
                clarity=float(np.clip(result.get("clarity", 0.5), 0, 1)),
                technical_accuracy=float(np.clip(result.get("technical_accuracy", 0.5), 0, 1)),
                logical_flow=float(np.clip(result.get("logical_flow", 0.5), 0, 1)),
                confidence=float(np.clip(result.get("confidence", 0.5), 0, 1)),
                justification=result.get("justification", "No justification provided"),
                parse_error=False,
            )

        except (json.JSONDecodeError, ValueError, KeyError):
            # Return fallback evaluation
            return EvaluationResult(
                judge_name=pretty_name,
                quality_score=0.5,
                correctness=0.5,
                completeness=0.5,
                clarity=0.5,
                technical_accuracy=0.5,
                logical_flow=0.5,
                confidence=0.3,
                justification="Failed to parse response",
                parse_error=True,
            )
