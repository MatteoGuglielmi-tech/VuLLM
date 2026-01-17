import logging
import asyncio
import tiktoken

from typing import Any
from dotenv import load_dotenv
from openai import AzureOpenAI, AsyncAzureOpenAI, RateLimitError
from openai.types.chat import ChatCompletion

from .base import ReasoningGenerator
from ..loader_config import Loader
from ..prompts import Messages, CWEPromptTemplate
from ..parsers import OutputParser, ParseResult

load_dotenv()

logger = logging.getLogger(name=__name__)


class AzureCoTGenerator(ReasoningGenerator[CWEPromptTemplate]):
    def __init__(
        self,
        deployment_name: str,
        prompt_template: CWEPromptTemplate | None = None,
        max_concurrent: int | None = None,  # None = unlimited
        min_reasoning_length: int = 30,
        async_mode: bool = True
    ):
        """Initializes the Azure OpenAI client."""
        super().__init__(prompt_template=prompt_template or CWEPromptTemplate())
        self.deployment_name = deployment_name
        self.max_concurrent = max_concurrent
        self.min_reasoning_length = min_reasoning_length
        self.async_mode = async_mode

        self.parser = OutputParser() 
        try:
            with Loader(
                desc_msg=f"Loading model: GPT-4.1",
                end_msg="✅ Azure OpenAI client initialized and credentials validated.",
                logger=logger,
            ):
                if self.async_mode:
                    self.async_client = AsyncAzureOpenAI(max_retries=3)
                else:
                    self.sync_client = AzureOpenAI(max_retries=3)
                    self.sync_client.with_options(max_retries=1).models.list()
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI client. Check your credentials and endpoint. Error: {e}")
            raise

        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.token_stats = {
            "min": float("inf"),
            "max": 0,
            "total": 0,
            "count": 0,
        }

    def _update_token_stats(self, text: str):
        """Track token usage statistics."""
        token_count = len(self.tokenizer.encode(text))
        self.token_stats["min"] = min(self.token_stats["min"], token_count)
        self.token_stats["max"] = max(self.token_stats["max"], token_count)
        self.token_stats["total"] += token_count
        self.token_stats["count"] += 1

    def get_token_statistics(self) -> dict:
        """Get token usage statistics."""
        if self.token_stats["count"] == 0:
            return {"message": "No completions generated yet"}

        avg = self.token_stats["total"] / self.token_stats["count"]
        return {
            "min_tokens": self.token_stats["min"],
            "max_tokens": self.token_stats["max"],
            "avg_tokens": round(avg, 1),
            "total_completions": self.token_stats["count"],
        }

    def build_cot_prompt(
        self,
        c_code: str,
        is_vulnerable: bool,
        cwe_ids: list[str] | None = None,
        cwe_descs: list[str] | None = None
    ) -> Messages:
        """
        Build the structured prompt for CWE analysis.

        Parameters
        ----------
        c_code : str
            C code to analyze
        is_vulnerable : bool
            Whether the code is known to be vulnerable
        cwe_ids : list[str] | None
            List of CWE identifiers

        Returns
        -------
        dict[str, str]
            Dictionary with 'system' and 'user' prompt content
        """
        return self.prompt_template.build_messages(
            func_code=c_code,
            is_vulnerable=is_vulnerable,
            cwe_ids=cwe_ids,
            cwe_descs=cwe_descs
        )

    def _call_api_sync(self, messages: Messages, max_tokens: int) -> ChatCompletion:
        """Synchronous API call."""
        return self.sync_client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,  # type: ignore
            max_completion_tokens=max_tokens,
            temperature=0.1,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    async def _call_api_async(self, messages: Messages, max_tokens: int) -> ChatCompletion:
        """Asynchronous API call."""
        return await self.async_client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,  # type: ignore
            max_completion_tokens=max_tokens,
            temperature=0.1,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    async def _process_single(
        self,
        sample: dict[str, Any],
        max_completion_tokens: int,
        semaphore: asyncio.Semaphore | None = None,
    ) -> dict[str, Any] | None:
        """
        Process a single sample asynchronously.

        Parameters
        ----------
        sample : dict
            Sample with keys: "func", "target", "cwe", "cwe_desc"
        max_completion_tokens : int
            Maximum tokens for completion
        semaphore : asyncio.Semaphore | None
            Semaphore for limiting concurrent requests

        Returns
        -------
        ParsedSample|None
            dict of { "reasoning": str, "verdict": dict} or None in case of failed generation
        """

        async def _call() -> str:
            """Make API call and return raw content."""
            response = await self._call_api_async(messages, max_completion_tokens)
            content = response.choices[0].message.content
            return content.strip() if content else ""

        messages = self.build_cot_prompt(
            c_code=sample["func"],
            is_vulnerable=bool(sample["target"]),
            cwe_ids=sample["cwe"],
            cwe_descs=sample["cwe_desc"],
        )

        try:
            if semaphore:
                async with semaphore:
                    content = await _call()
            else:
                content = await _call()

        except RateLimitError:
            logger.warning("⚠️  Rate limit hit, waiting 10s before retry...")
            await asyncio.sleep(10)
            return await self._process_single(
                sample=sample,
                max_completion_tokens=max_completion_tokens,
                semaphore=semaphore,
            )

        except Exception:
            logger.exception("❌ API call failed")
            return None

        result = self.parser.parse(content)

        if result.success:
            self._update_token_stats(
                result.sample.model_dump_json(indent=2, ensure_ascii=False)
            )
            return result.sample.model_dump()
        else:
            logger.warning(f"Skipping sample: {result.error}")
            return None

    async def _generate_reasoning_async_impl(
        self, mini_batch: list[dict[str, Any]], max_completion_tokens: int
    ) -> list[dict[str,Any]]:
        """
        Async implementation: generate reasoning concurrently.

        Parameters
        ----------
        mini_batch : list[dict]
            Batch of samples to process
        max_completion_tokens : int
            Maximum tokens per completion

        Returns
        -------
        list[ParsedSample]
            List of successfully parsed samples (failures filtered out)
        """
        if not mini_batch:
            logger.warning("⚠️  Empty batch received")
            return []

        semaphore = (
            asyncio.Semaphore(self.max_concurrent) if self.max_concurrent else None
        )

        tasks = [
            self._process_single(sample, max_completion_tokens, semaphore)
            for sample in mini_batch
        ]

        logger.info(f"🚀 Processing {len(tasks)} samples concurrently...")
        results = await asyncio.gather(*tasks)

        # filter out failed samples (None values)
        successful = [r for r in results if r is not None]

        failed_count = len(results) - len(successful)
        if failed_count > 0:
            logger.warning(f"⚠️  {failed_count}/{len(results)} samples failed parsing")

        return successful

    def _generate_reasoning_async(
        self,
        mini_batch: list[dict[str, Any]],
        max_completion_tokens: int,
    ) -> list[ParseResult]:
        """Run async implementation in sync context."""
        return asyncio.run(
            self._generate_reasoning_async_impl(mini_batch, max_completion_tokens)
        )

    def _generate_reasoning_sync(
        self,
        mini_batch: list[dict[str, Any]],
        max_completion_tokens: int,
    ) -> list[ParseResult]:
        """Generates CoT reasoning for a batch of entries by making API calls to Azure OpenAI."""

        results: list[ParseResult] = []
        if not mini_batch:
            logger.warning("Empty batch detected.")
            return results

        for sample in mini_batch:
            messages = self.build_cot_prompt(
                c_code=sample["func"],
                is_vulnerable=bool(sample["target"]),
                cwe_ids=sample["cwe"],
                cwe_descs=sample["cwe_desc"],
            )
            try:
                response = self._call_api_sync(
                    messages=messages, max_tokens=max_completion_tokens
                )
                reasoning = response.choices[0].message.content
                result = self.parser.parse(reasoning.strip() if reasoning else "")
                if result.success and result.sample is not None:
                    results.append(result.sample)
            except Exception as e:
                logger.error(f"An error occurred during an API call: {e}")
                # results.append("")

        return results

    def generate_reasoning(
        self, mini_batch: list[dict[str, Any]], max_completion_tokens: int
    ) -> list[ParseResult]:
        """
        Generate Chain-of-Thought reasoning for a batch of samples.

        Parameters
        ----------
        mini_batch : list[dict]
            Batch of samples with keys: "func", "target", "cwe", "cwe_desc"
        max_completion_tokens : int
            Maximum tokens per completion
        async_mode : bool, default=True
            If True, process samples concurrently (faster).
            If False, process sequentially (slower but simpler).

        Returns
        -------
        list[str]
            List of generated reasoning strings (one per sample)

        Examples
        --------
        >>> generator = AzureCoTGenerator(deployment_name="gpt-4")
        >>> batch = [{"func": "...", "target": 1, "cwe": [...], ...}]
        >>> results = generator.generate_reasoning(batch, max_completion_tokens=500)
        """
        if self.async_mode:
            return self._generate_reasoning_async(mini_batch, max_completion_tokens)
        else:
            return self._generate_reasoning_sync(mini_batch, max_completion_tokens)
