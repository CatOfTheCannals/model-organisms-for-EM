from typing import Dict, List, Optional
import math
from functools import lru_cache
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv  # type: ignore

# We dynamically choose between AzureOpenAI and regular OpenAI clients based on which
# environment variables are available. This keeps the public API (the OpenAiJudge
# class) unchanged while adding support for the standard OpenAI service.

try:
    # The unified `openai` >= 1.x SDK ships specialised client classes for Azure and
    # regular OpenAI. Import both if available.
    from openai import AzureOpenAI, OpenAI  # type: ignore
except ImportError as e:  # pragma: no cover – handled at runtime
    raise ImportError(
        "openai python package is required for judging. Please install with `pip install openai`"
    ) from e

from em_organism_dir.global_variables import AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION

# Load environment variables from .env file
load_dotenv()

# Global configuration constants for Azure. For plain OpenAI, these are ignored.

# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

# Detect which credentials are present. We prefer Azure if its dedicated key is
# available. Otherwise we fall back to a standard OpenAI client.

_azure_key = os.getenv("AZURE_OPENAI_API_KEY")
_env_azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
_openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or os.getenv("API_KEY")

# We only choose the Azure client if BOTH a dedicated Azure key *and* an Azure
# endpoint are set **in the environment**. This prevents accidentally treating a
# regular OpenAI key (mis-named as AZURE_OPENAI_API_KEY) as an Azure key.
if _azure_key and _env_azure_endpoint:
    _CLIENT_TYPE = "azure"
    client = AzureOpenAI(
        api_key=_azure_key,
        azure_endpoint=_env_azure_endpoint,
        api_version=AZURE_API_VERSION,
    )
    _DEFAULT_MODEL = AZURE_DEPLOYMENT
else:
    _CLIENT_TYPE = "openai"
    if not _openai_key:
        # Fail fast with a clear error instead of the generic authentication one.
        raise EnvironmentError(
            "No valid OpenAI credentials found. Please set `AZURE_OPENAI_API_KEY`+`AZURE_OPENAI_ENDPOINT` "
            "for Azure or `OPENAI_API_KEY` for the standard OpenAI service."
        )

    client = OpenAI(api_key=_openai_key)
    # Default chat model for the public OpenAI service.
    _DEFAULT_MODEL = "gpt-4o"

class OpenAiJudge:
    """Azure OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""

    def __init__(self, deployment: str, prompt_template: str):
        self.deployment = deployment
        self.prompt_template = prompt_template


    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        # Choose model name depending on the client type. We keep the ability for the
        # caller to override `self.deployment` when constructing the judge.
        model_name = self.deployment or _DEFAULT_MODEL

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )

        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))

        return result

    def _aggregate_0_100_score(self, score: dict) -> Optional[float]:
        #   NOTE: we don't check for refusals explicitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)