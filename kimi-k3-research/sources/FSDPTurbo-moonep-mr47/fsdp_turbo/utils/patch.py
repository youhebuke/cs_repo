# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import importlib
import inspect
import sys
import types

# Attribute used to record replacement identities already installed on a target.
PATCH_IDS_ATTR = "_fsdp_turbo_patch_ids"


def patch_namespace_members(targets, replacements, *, after_patch=None):
    """Replace callable members in importable module namespaces.

    Args:
        targets: A dotted module-member path or an iterable of such paths.
        replacements: A callable/path or an iterable of callables/paths. A
            single replacement is reused for every target.
        after_patch: Optional callback receiving ``(original, replacement)``
            after each new installation. It is not called for an idempotent
            installation.

    Returns:
        The number of target-replacement pairs processed. This counts matched
        pairs even when a replacement was already installed.

    Notes:
        Already-loaded aliases that still reference an original member are
        updated as well, so ``from module import member`` users see the patch.
    """
    pairs = expand_pairs(targets, replacements)
    for target, replacement in pairs:
        owner, name = resolve_owner_member(target)
        replacement = resolve_callable(replacement)
        original = _patch_namespace_member(owner, name, replacement)
        if original is not None and after_patch is not None:
            after_patch(original, replacement)
    return len(pairs)


def apply_module_patches(model, module_patches):
    """Apply a sequence of ``target``/``replacement`` patch specifications.

    Args:
        model: Model-like object exposing ``named_modules()``.
        module_patches: One patch mapping or an iterable of mappings. Each
            mapping must contain a dotted ``target`` and a callable
            ``replacement`` or its dotted path.

    Returns:
        A tuple ``(model, matched)``. ``matched`` is the number of matched
        model members plus one for each matched module-level member.
    """
    matched = 0
    for spec in as_list(module_patches):
        target = spec["target"]
        namespace_target = resolve_namespace_member(target)
        if namespace_target is not None:
            owner, member_name = namespace_target
            replacement = resolve_callable(spec["replacement"])
            _patch_namespace_member(owner, member_name, replacement)
            matched += 1
        else:
            matched += patch_model_members(model, target, spec["replacement"])
    return model, matched


def patch_model_members(model, targets, replacements):
    """Patch members on every matching module in a model.

    Args:
        model: Model-like object exposing ``named_modules()``.
        targets: A dotted ``module.Class.member`` path or an iterable of paths.
        replacements: A direct callable/path or an iterable of replacements.
            A single replacement is reused for every target.

    Returns:
        The number of matched model modules across all target-replacement pairs.

    Raises:
        TypeError: If a matched member is neither a property nor callable.
        ValueError: If a callable path is malformed.
    """
    matched = 0
    for target, replacement in expand_pairs(targets, replacements):
        replacement = resolve_callable(replacement)
        cls, member_name = resolve_class_member(target)
        for _, module in model.named_modules():
            if not isinstance(module, cls):
                continue
            matched += 1
            _patch_member(module, member_name, replacement, target)
    return matched


def _patch_member(module, member_name, replacement, target):
    """Install a replacement while preserving the member's binding protocol.

    ``getattr_static`` is used to inspect the raw descriptor. Normal
    ``getattr`` would execute a property getter and would hide whether the
    member is a property, staticmethod, or classmethod.
    """
    descriptor = inspect.getattr_static(module, member_name)
    if isinstance(descriptor, property):
        _patch_property(module, member_name, descriptor, replacement)
        return

    original = getattr(module, member_name)
    if not callable(original):
        raise TypeError(f"Target member {target!r} is neither a property nor a callable.")
    receiver = getattr(original, "__self__", None)
    if receiver is not module and not isinstance(receiver, type):
        receiver = None
    # Ordinary methods bind to the instance; classmethods bind to the class.
    # Staticmethods and callable attributes have no receiver and stay unbound.
    replacement = types.MethodType(replacement, receiver) if receiver is not None else replacement
    _install(module, member_name, original, replacement)


def _patch_property(module, member_name, descriptor, replacement):
    """Replace a property getter and retain its setter, deleter, and docstring."""
    if descriptor.fget is None:
        raise TypeError(f"Cannot patch write-only property {type(module).__name__}.{member_name}.")
    owner = next(base for base in type(module).__mro__ if member_name in base.__dict__)
    installed = property(replacement, descriptor.fset, descriptor.fdel, descriptor.__doc__)
    _install(owner, member_name, descriptor.fget, installed, replacement)


def _patch_namespace_member(owner, member_name, replacement):
    """Install a module-level replacement and refresh loaded aliases."""
    original = getattr(owner, member_name)
    if not _install(owner, member_name, original, replacement):
        return None
    _replace_loaded_aliases(original, replacement)
    return original


def _install(owner, member_name, original, replacement, source=None):
    """Install a replacement once and record its identity on the callable.

    Args:
        owner: Object receiving the replacement attribute.
        member_name: Attribute name to replace.
        original: Current target object, used for idempotence lookup.
        replacement: Object assigned to ``owner.member_name``.
        source: Callable whose identity should be recorded. This is needed for
            properties because the assigned object is a new ``property`` while
            the replacement identity belongs to its getter.

    Returns:
        ``True`` when installed, otherwise ``False`` when the same replacement
        was already recorded on the target.
    """
    source = replacement if source is None else source
    patch_id = callable_id(source)
    applied = set(getattr(original, PATCH_IDS_ATTR, ()))
    if patch_id in applied:
        return False
    source = getattr(source, "__func__", source)
    setattr(source, PATCH_IDS_ATTR, tuple(applied | {patch_id}))
    setattr(owner, member_name, replacement)
    return True


def resolve_namespace_member(path):
    """Resolve ``module.member`` when the owner module and member are callable.

    Returns ``None`` when the path is not a module-level callable path, allowing
    callers to try class-member resolution next.
    """
    owner_path, sep, member_name = str(path).rpartition(".")
    if not sep:
        return None
    try:
        owner = import_module_or_suffix(owner_path)
    except ModuleNotFoundError:
        return None
    return (owner, member_name) if callable(getattr(owner, member_name, None)) else None


def _replace_loaded_aliases(original, replacement):
    """Replace loaded module attributes that still reference ``original``."""
    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if namespace is None:
            continue
        for name, value in tuple(namespace.items()):
            if value is original:
                namespace[name] = replacement


def resolve_callable(path_or_callable):
    """Return a callable from a direct object or a dotted import path.

    Args:
        path_or_callable: A callable object or a path such as
            ``package.module.replacement``.

    Raises:
        TypeError: If the resolved object is not callable.
        ValueError: If the path is not a dotted path or uses ``module:name``
            syntax.
    """
    if callable(path_or_callable):
        return path_or_callable
    path = str(path_or_callable)
    if ":" in path:
        raise ValueError(f"Use dotted callable paths, not colon paths: {path!r}.")
    module_path, sep, name = path.rpartition(".")
    if not sep:
        raise ValueError(f"Invalid callable path {path!r}.")
    module = import_module_or_suffix(module_path)
    fn = getattr(module, name, None)
    if not callable(fn):
        raise TypeError(f"Resolved object {path!r} is not callable.")
    return fn


def import_module_or_suffix(module_path):
    """Import a module path, retrying progressively shorter suffixes.

    The suffix fallback supports dotted paths whose leading components are
    namespace/package prefixes while preserving import errors raised from an
    existing module's own dependencies.
    """
    parts = module_path.split(".")
    for index in range(len(parts)):
        candidate = ".".join(parts[index:])
        try:
            return importlib.import_module(candidate)
        except ModuleNotFoundError as error:
            missing = error.name or ""
            if missing != candidate and not candidate.startswith(f"{missing}."):
                raise
    raise ModuleNotFoundError(f"No module named {module_path!r}.")


def resolve_owner_member(path: str):
    """Resolve the module owner and member name from ``module.member``."""
    module_path, sep, member_name = path.rpartition(".")
    if not sep:
        raise ValueError(f"Invalid member path {path!r}.")
    return importlib.import_module(module_path), member_name


def resolve_class_member(path: str):
    """Resolve the class and member name from ``module.Class.member``."""
    class_path, sep, member_name = path.rpartition(".")
    if not sep:
        raise ValueError(f"Invalid class member path {path!r}.")
    module_path, sep, class_name = class_path.rpartition(".")
    if not sep:
        raise ValueError(f"Invalid class member path {path!r}; use 'module.Class.member'.")
    cls = getattr(importlib.import_module(module_path), class_name, None)
    if not isinstance(cls, type):
        raise TypeError(f"Resolved class {class_path!r} is not a type.")
    return cls, member_name


def expand_pairs(targets, replacements):
    """Normalize target/replacement inputs and pair them by position.

    A scalar string or callable is treated as one item. One replacement may be
    broadcast to multiple targets; otherwise both collections must have the
    same length.
    """
    targets = as_list(targets)
    replacements = as_list(replacements)
    if not targets:
        raise ValueError("targets must not be empty.")
    if len(replacements) == 1:
        replacements = replacements * len(targets)
    elif len(replacements) != len(targets):
        raise ValueError("replacements must contain either 1 item or the same number of items as targets.")
    return list(zip(targets, replacements))


def callable_id(fn):
    """Return an object-specific identity used for replacement idempotence."""
    fn = getattr(fn, "__func__", fn)
    return (
        getattr(fn, "__module__", type(fn).__module__),
        getattr(fn, "__qualname__", type(fn).__qualname__),
        id(fn),
    )


def as_list(value):
    """Normalize a scalar patch input or iterable to a list."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, dict)) or callable(value):
        return [value]
    return list(value)
