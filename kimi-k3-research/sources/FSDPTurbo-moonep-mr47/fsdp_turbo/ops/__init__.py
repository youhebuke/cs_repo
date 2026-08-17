from fsdp_turbo.ops.registry import (
    _OPERATOR_REGISTRY,
    dispatch_op,
    get_op,
    register_op,
)
import fsdp_turbo.ops.cpu

try:
    import fsdp_turbo.ops.npu  # noqa: F401
except ImportError:
    pass
try:
    import fsdp_turbo.ops.cuda  # noqa: F401
except ImportError:
    pass

__all__ = ['register_op', 'get_op', 'dispatch_op', '_OPERATOR_REGISTRY']
