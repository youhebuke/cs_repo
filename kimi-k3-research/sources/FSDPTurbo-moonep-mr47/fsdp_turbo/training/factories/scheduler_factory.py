import logging
from typing import Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)

_registry: Dict[str, Callable] = {}


class SchedulerFactory:
    """Create LR schedulers from torch.optim.lr_scheduler or user-registered sources.

    Priority: custom_fn > registered > torch.optim.lr_scheduler.

    Usage:
        # torch.optim.lr_scheduler (default)
        scheduler = SchedulerFactory.create("CosineAnnealingLR", optimizer, T_max=1000)

        # Register a custom creator
        SchedulerFactory.register("my_sched", my_scheduler_fn)
        scheduler = SchedulerFactory.create("my_sched", optimizer)

        # Pass custom_fn directly
        scheduler = SchedulerFactory.create("my_sched", optimizer, custom_fn=my_scheduler_fn)
    """

    @staticmethod
    def register(name: str, custom_fn: Callable):
        """Register a custom scheduler creator.

        Args:
            name: Identifier for the custom creator.
            custom_fn: Callable that returns a scheduler, receives **kwargs.
        """
        _registry[name] = custom_fn
        logger.info(f"Registered custom scheduler creator: {name}")

    @staticmethod
    def create(
        source: str,
        *args,
        custom_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """Create an LR scheduler.

        Priority: custom_fn > registered > torch.optim.lr_scheduler.

        Args:
            source: Scheduler name in torch.optim.lr_scheduler (e.g. "CosineAnnealingLR")
                    or registered name.
            *args: Positional args forwarded to the scheduler constructor (e.g. optimizer).
            custom_fn: Optional callable that returns a scheduler, receives *args, **kwargs.
            **kwargs: Keyword args forwarded to the scheduler constructor.
        """
        if custom_fn is not None:
            logger.info(f"Creating scheduler from custom_fn: {source}")
            return custom_fn(*args, **kwargs)

        if source in _registry:
            logger.info(f"Creating scheduler from registered source: {source}")
            return _registry[source](*args, **kwargs)

        sched_cls = getattr(torch.optim.lr_scheduler, source, None)
        if sched_cls is None:
            available = [k for k in dir(torch.optim.lr_scheduler) if k[0].isupper()]
            raise KeyError(f"Scheduler '{source}' not found in torch.optim.lr_scheduler. "
                           f"Available: {available}")
        logger.info(f"Creating scheduler from torch.optim.lr_scheduler: {source}")
        return sched_cls(*args, **kwargs)
