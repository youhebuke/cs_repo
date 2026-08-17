import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_registry: Dict[str, Callable] = {}


class TokenizerFactory:
    """Create tokenizers from HuggingFace or user-registered sources.

    Priority: custom_fn > registered > HuggingFace.

    Usage:
        # HuggingFace (default)
        tokenizer = TokenizerFactory.create("Qwen/Qwen3-30B-A3B")

        # Register a custom creator
        TokenizerFactory.register("my_tok", my_tokenizer_fn)
        tokenizer = TokenizerFactory.create("my_tok")

        # Pass custom_fn directly
        tokenizer = TokenizerFactory.create("my_tok", custom_fn=my_tokenizer_fn)
    """

    @staticmethod
    def register(name: str, custom_fn: Callable):
        """Register a custom tokenizer creator.

        Args:
            name: Identifier for the custom creator.
            custom_fn: Callable that returns a tokenizer, receives **kwargs.
        """
        _registry[name] = custom_fn
        logger.info(f"Registered custom tokenizer creator: {name}")

    @staticmethod
    def create(
        source: str,
        custom_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """Create a tokenizer.

        Priority: custom_fn > registered > HuggingFace.

        Args:
            source: Model name / path (HuggingFace) or registered name.
            custom_fn: Optional callable that returns a tokenizer, receives **kwargs.
            **kwargs: Forwarded to the underlying create call.
        """
        if custom_fn is not None:
            logger.info(f"Creating tokenizer from custom_fn: {source}")
            return custom_fn(**kwargs)

        if source in _registry:
            logger.info(f"Creating tokenizer from registered source: {source}")
            return _registry[source](**kwargs)

        from transformers import AutoTokenizer
        logger.info(f"Creating tokenizer from HuggingFace: {source}")
        return AutoTokenizer.from_pretrained(source, **kwargs)
