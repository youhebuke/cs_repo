import logging
from typing import Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)

_registry: Dict[str, Callable] = {}


class OptimizerFactory:
    """Create optimizers from torch.optim or user-registered sources.

    Priority: custom_fn > registered > torch.optim.

    Usage:
        # torch.optim (default)
        optimizer = OptimizerFactory.create("AdamW", model.parameters(), lr=1e-4)

        # Register a custom creator
        OptimizerFactory.register("my_opt", my_optimizer_fn)
        optimizer = OptimizerFactory.create("my_opt", model.parameters())

        # Pass custom_fn directly
        optimizer = OptimizerFactory.create("my_opt", model.parameters(), custom_fn=my_optimizer_fn)
    """

    @staticmethod
    def register(name: str, custom_fn: Callable):
        """Register a custom optimizer creator.

        Args:
            name: Identifier for the custom creator.
            custom_fn: Callable that returns an optimizer, receives **kwargs.
        """
        _registry[name] = custom_fn
        logger.info(f"Registered custom optimizer creator: {name}")

    @staticmethod
    def create(
        source: str,
        *args,
        custom_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """Create an optimizer.

        Priority: custom_fn > registered > torch.optim.

        Args:
            source: Optimizer name in torch.optim (e.g. "AdamW", "SGD") or registered name.
            *args: Positional args forwarded to the optimizer constructor (e.g. model.parameters()).
            custom_fn: Optional callable that returns an optimizer, receives *args, **kwargs.
            **kwargs: Keyword args forwarded to the optimizer constructor (e.g. lr, weight_decay).
        """
        if custom_fn is not None:
            logger.info(f"Creating optimizer from custom_fn: {source}")
            return custom_fn(*args, **kwargs)

        if source in _registry:
            logger.info(f"Creating optimizer from registered source: {source}")
            return _registry[source](*args, **kwargs)

        opt_cls = getattr(torch.optim, source, None)
        if opt_cls is None:
            available = [k for k in dir(torch.optim) if k[0].isupper()]
            raise KeyError(f"Optimizer '{source}' not found in torch.optim. "
                           f"Available: {available}")
        logger.info(f"Creating optimizer from torch.optim: {source}")
        return opt_cls(*args, **kwargs)
