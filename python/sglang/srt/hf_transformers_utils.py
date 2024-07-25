"""Utilities for Huggingface Transformers."""

import json
import os
import warnings
import functools
from typing import Optional, Union, AbstractSet, Collection, Literal

from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from sglang.srt.utils import is_multimodal_model


def download_from_hf(model_path: str):
    if os.path.exists(model_path):
        return model_path

    return snapshot_download(model_path, allow_patterns=["*.json", "*.bin", "*.model"])


def get_config_json(model_path: str):
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)
    return config


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_overide_args: Optional[dict] = None,
):
    config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code, revision=revision
    )
    if model_overide_args:
        config.update(model_overide_args)
    return config


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = config.rope_scaling["factor"]
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if tokenizer_name.endswith(".json"):
        return TiktokenTokenizer(tokenizer_name)

    """Gets a tokenizer for the given model name via Huggingface."""
    if is_multimodal_model(tokenizer_name):
        processor = get_processor(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs,
        )
        tokenizer = processor.tokenizer
        return tokenizer

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if (
        "llama" in tokenizer_name.lower()
        and kwargs.get("use_fast", True)
        and tokenizer_name != _FAST_LLAMA_TOKENIZER
    ):
        pass
        # warnings.warn(
        #    "For some LLaMA V1 models, initializing the fast tokenizer may "
        #    "take a long time. To reduce the initialization time, consider "
        #    f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
        #    "tokenizer."
        # )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs,
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    return tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
):
    processor = AutoProcessor.from_pretrained(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        **kwargs,
    )
    return processor


class TiktokenTokenizer:
    def __init__(self, tokenizer_path):
        import tiktoken
        PAT_STR_B = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

        # Read JSON
        name = "tmp-json"
        with open(tokenizer_path, "rb") as fin:
            tok_dict = json.load(fin)

        mergeable_ranks = {
            bytes(item["bytes"]): item["token"] for item in tok_dict["regular_tokens"]
        }
        special_tokens = {
            bytes(item["bytes"]).decode(): item["token"] for item in tok_dict["special_tokens"]
        }
        assert tok_dict["word_split"] == "V1"

        kwargs = {
            "name": name,
            "pat_str": tok_dict.get("pat_str", PAT_STR_B),
            "mergeable_ranks": mergeable_ranks,
            "special_tokens": special_tokens,
        }
        if "default_allowed_special" in tok_dict:
            default_allowed_special = set(
                [bytes(bytes_list).decode() for bytes_list in tok_dict["default_allowed_special"]]
            )
        else:
            default_allowed_special = None
        if "vocab_size" in tok_dict:
            kwargs["explicit_n_vocab"] = tok_dict["vocab_size"]

        tokenizer = tiktoken.Encoding(**kwargs)
        tokenizer._default_allowed_special = default_allowed_special or set()

        def encode_patched(
            self,
            text: str,
            *,
            allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
            disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        ) -> list[int]:
            if isinstance(allowed_special, set):
                allowed_special |= self._default_allowed_special
            return tiktoken.Encoding.encode(
                self, text, allowed_special=allowed_special, disallowed_special=disallowed_special
            )
        tokenizer.encode = functools.partial(encode_patched, tokenizer)

        # Convert to HF interface
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer._special_tokens["<|eos|>"]
        self.vocab_size = tokenizer.n_vocab

    def encode(self, x, add_special_tokens=False):
        return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)

    def batch_decode(self, batch, skip_special_tokens=True,  spaces_between_special_tokens=False):
        if isinstance(batch[0], int):
            batch = [[x] for x in batch]
        return self.tokenizer.decode_batch(batch)

    def convert_ids_to_tokens(self, index):
        return self.tokenizer.decode_single_token_bytes(index).decode("utf-8", errors="ignore")