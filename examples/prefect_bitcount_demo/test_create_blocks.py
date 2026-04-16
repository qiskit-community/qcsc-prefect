from __future__ import annotations

from examples.prefect_bitcount_demo import create_blocks as mod


def test_resolve_legacy_tutorial_asset_creation_defaults_to_false():
    assert (
        mod._resolve_legacy_tutorial_asset_creation(
            requested=None,
            tutorial_variable_name=None,
            bitcounter_block_name=None,
        )
        is False
    )


def test_resolve_legacy_tutorial_asset_creation_infers_from_explicit_names():
    assert (
        mod._resolve_legacy_tutorial_asset_creation(
            requested=None,
            tutorial_variable_name="miyabi-tutorial",
            bitcounter_block_name=None,
        )
        is True
    )


def test_resolve_legacy_tutorial_asset_creation_honors_explicit_false():
    assert (
        mod._resolve_legacy_tutorial_asset_creation(
            requested=False,
            tutorial_variable_name="miyabi-tutorial",
            bitcounter_block_name="miyabi-tutorial",
        )
        is False
    )


def test_normalize_bool_accepts_common_string_values():
    assert mod._normalize_bool("true") is True
    assert mod._normalize_bool("0") is False
    assert mod._normalize_bool("") is None
