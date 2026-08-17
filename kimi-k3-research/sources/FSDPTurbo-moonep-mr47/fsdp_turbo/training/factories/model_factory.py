import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_registry: Dict[str, Callable] = {}


class ModelFactory:
    """Create models from HuggingFace or user-registered sources.

    Priority: custom_fn > registered > HuggingFace.

    Usage:
        # HuggingFace (default)
        model = ModelFactory.create("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)

        # Register a custom creator
        ModelFactory.register("my_model", my_model_fn)
        model = ModelFactory.create("my_model")

        # Pass custom_fn directly
        model = ModelFactory.create("my_model", custom_fn=my_model_fn)
    """

    @staticmethod
    def register(name: str, custom_fn: Callable):
        """Register a custom model creator.

        Args:
            name: Identifier for the custom creator.
            custom_fn: Callable that returns a model, receives **kwargs.
        """
        _registry[name] = custom_fn
        logger.info(f"Registered custom model creator: {name}")

    @staticmethod
    def create(
        source: str,
        custom_fn: Optional[Callable] = None,
        model_class: Optional[str] = None,
        **kwargs,
    ):
        """Create a model.

        Priority: custom_fn > registered > HuggingFace.

        Args:
            source: Model name / path (HuggingFace) or registered name.
            custom_fn: Optional callable that returns a model, receives **kwargs.
            model_class: HuggingFace auto class name (default: AutoModelForCausalLM).
            **kwargs: Forwarded to the underlying create call (e.g. torch_dtype).
        """
        if custom_fn is not None:
            logger.info(f"Creating model from custom_fn: {source}")
            return custom_fn(**kwargs)

        if source in _registry:
            logger.info(f"Creating model from registered source: {source}")
            return _registry[source](**kwargs)

        import transformers
        cls_name = model_class or "AutoModelForCausalLM"
        auto_cls = getattr(transformers, cls_name)
        logger.info(f"Creating model from HuggingFace: {source} ({cls_name})")
        return auto_cls.from_pretrained(source, **kwargs)
