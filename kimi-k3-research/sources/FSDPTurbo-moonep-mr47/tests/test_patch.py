import sys
import tempfile
import types
import unittest

from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, load_config_from_yaml
from fsdp_turbo.utils.patch import apply_module_patches, patch_model_members, patch_namespace_members


def patched_function(value, bias=0):
    return value + bias + 10


def patched_method(self, value):
    return value * 3


def patched_static_method(value):
    return value + 10


def patched_class_method(cls, value):
    return f"{cls.__name__}:{value + 10}"


def patched_class_method_again(cls, value):
    return f"{cls.__name__}:{value + 20}"


def patched_property(self):
    return self._value + 10


def patched_callable_attribute(value):
    return value + 10


def make_dynamic_replacement(offset):
    def replacement(value, bias=0):
        return value + bias + offset

    return replacement


class PatchTargetTests(unittest.TestCase):
    owner_name = "_fsdp_turbo_patch_test_owner"
    alias_name = "_fsdp_turbo_patch_test_alias"
    model_name = "_fsdp_turbo_patch_test_model"

    def setUp(self):
        self.owner = types.ModuleType(self.owner_name)
        self.alias = types.ModuleType(self.alias_name)
        self.model_module = types.ModuleType(self.model_name)

        def target_function(value, bias=0):
            return value + bias

        self.owner.target_function = target_function
        self.alias.target_function = target_function

        class Block:
            def __init__(self):
                self._value = 1

            def forward(self, value):
                return value * 2

            @staticmethod
            def static_method(value):
                return value * 2

            @classmethod
            def class_method(cls, value):
                return f"{cls.__name__}:{value * 2}"

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, value):
                self._value = value

            class CallableAttribute:
                def __call__(self, value):
                    return value * 2

            callable_attribute = CallableAttribute()

        self.model_module.Block = Block
        sys.modules[self.owner_name] = self.owner
        sys.modules[self.alias_name] = self.alias
        sys.modules[self.model_name] = self.model_module

        class Model:
            def __init__(self):
                self.block = Block()

            def named_modules(self):
                return [("", self), ("block", self.block)]

        self.model = Model()

    def tearDown(self):
        for name in (self.owner_name, self.alias_name, self.model_name):
            sys.modules.pop(name, None)

    def test_module_function_patch_updates_loaded_aliases_and_is_idempotent(self):
        _, matched = apply_module_patches(
            self.model,
            [
                {
                    "target": f"{self.owner_name}.target_function",
                    "replacement": patched_function,
                }
            ],
        )
        self.assertEqual(matched, 1)
        self.assertEqual(getattr(self.owner, "target_function")(2, bias=3), 15)
        self.assertEqual(getattr(self.alias, "target_function")(2, bias=3), 15)

        _, matched_again = apply_module_patches(
            self.model,
            [
                {
                    "target": f"{self.owner_name}.target_function",
                    "replacement": patched_function,
                }
            ],
        )
        self.assertEqual(matched_again, 1)

    def test_instance_method_patch_keeps_self_binding(self):
        matched = patch_model_members(self.model, f"{self.model_name}.Block.forward", patched_method)
        self.assertEqual(matched, 1)
        self.assertEqual(self.model.block.forward(4), 12)

    def test_static_class_property_and_callable_attributes_keep_their_protocols(self):
        specs = [
            ("static_method", patched_static_method),
            ("class_method", patched_class_method),
            ("value", patched_property),
            ("callable_attribute", patched_callable_attribute),
        ]
        for member_name, replacement in specs:
            _, matched = apply_module_patches(
                self.model,
                [
                    {
                        "target": f"{self.model_name}.Block.{member_name}",
                        "replacement": replacement,
                    }
                ],
            )
            self.assertEqual(matched, 1)

        self.assertEqual(self.model.block.static_method(2), 12)
        self.assertEqual(self.model.block.class_method(2), "Block:12")
        apply_module_patches(
            self.model,
            [
                {
                    "target": f"{self.model_name}.Block.class_method",
                    "replacement": patched_class_method_again,
                }
            ],
        )
        self.assertEqual(self.model.block.class_method(2), "Block:22")
        self.assertEqual(self.model.block.value, 11)
        self.model.block.value = 7
        self.assertEqual(self.model.block.value, 17)
        self.assertEqual(self.model.block.callable_attribute(2), 12)

    def test_namespace_member_patch_accepts_direct_replacements(self):
        matched = patch_namespace_members(
            f"{self.owner_name}.target_function",
            patched_function,
        )
        self.assertEqual(matched, 1)
        self.assertEqual(getattr(self.owner, "target_function")(2, bias=3), 15)
        self.assertEqual(getattr(self.alias, "target_function")(2, bias=3), 15)

    def test_namespace_member_patch_is_idempotent_and_calls_callback_once(self):
        callbacks = []

        def callback(original, replacement):
            callbacks.append((original, replacement))

        path = f"{self.owner_name}.target_function"
        patch_namespace_members(path, patched_function, after_patch=callback)
        patch_namespace_members(path, patched_function, after_patch=callback)
        self.assertEqual(len(callbacks), 1)

    def test_namespace_member_patch_distinguishes_same_qualified_name_replacements(self):
        path = f"{self.owner_name}.target_function"
        patch_namespace_members(path, make_dynamic_replacement(10))
        patch_namespace_members(path, make_dynamic_replacement(20))
        self.assertEqual(getattr(self.owner, "target_function")(2, bias=3), 25)
        self.assertEqual(getattr(self.alias, "target_function")(2, bias=3), 25)

    def test_config_rejects_malformed_module_patch_values(self):
        invalid_values = (
            "invalid",
            [None],
            [{"target": "x.y", "replacement": "invalid"}],
            [{"target": "x.y", "replacement": object()}],
        )
        for module_patches in invalid_values:
            with self.subTest(module_patches=module_patches):
                with self.assertRaises(ValueError):
                    FSDPTurboConfig(module_patches=module_patches)

    def test_config_string_includes_module_patches(self):
        config = FSDPTurboConfig(module_patches=[{"target": "x.y", "replacement": patched_function}])
        self.assertIn("module_patches", str(config))

    def test_yaml_config_loads_and_applies_patch(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as config_file:
            config_file.write(
                "module_patches:\n"
                f"  - target: {self.owner_name}.target_function\n"
                f"    replacement: {__name__}.patched_function\n"
                f"  - target: {self.model_name}.Block.forward\n"
                f"    replacement: {__name__}.patched_method\n"
            )
            config_file.flush()
            config = load_config_from_yaml(config_file.name)

        self.assertEqual(len(config.module_patches), 2)
        _, matched = apply_module_patches(self.model, config.module_patches)
        self.assertEqual(matched, 2)
        self.assertEqual(getattr(self.owner, "target_function")(2, bias=3), 15)
        self.assertEqual(getattr(self.alias, "target_function")(2, bias=3), 15)
        self.assertEqual(self.model.block.forward(4), 12)


if __name__ == "__main__":
    unittest.main()
