from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any


FULL_TEXTURE_KEYS = {
    "full_map_block_textures",
    "full_map_item_textures",
    "full_map_player_textures",
    "full_map_player_textures_alpha",
    "melee_mob_textures",
    "melee_mob_texture_alphas",
    "passive_mob_textures",
    "passive_mob_texture_alphas",
    "ranged_mob_textures",
    "ranged_mob_texture_alphas",
    "projectile_textures",
    "projectile_texture_alphas",
    "night_texture",
    "night_noise_intensity_texture",
    "number_textures",
    "number_textures_alpha",
    "number_textures_with_zero",
    "number_textures_alpha_with_zero",
    "smaller_block_textures",
    "smaller_empty_texture",
    "armour_textures",
    "bow_textures",
    "player_projectile_textures",
    "potion_textures",
}


def _shared_cache_root() -> Path | None:
    raw = str(os.getenv("NANOHORIZON_CRAFTAX_CACHE_DIR") or "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sync_shared_cache(
    *,
    target_file: str | os.PathLike[str],
    shared_root: Path | None,
    namespace: str,
) -> dict[str, Any]:
    target_path = Path(target_file).expanduser().resolve()
    report: dict[str, Any] = {
        "texture_cache_file": str(target_path),
        "exists": target_path.exists(),
    }
    if shared_root is None:
        return report

    shared_path = shared_root / namespace / target_path.name
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    report["shared_texture_cache_file"] = str(shared_path)
    report["shared_exists"] = shared_path.exists()

    if shared_path.exists() and not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shared_path, target_path)
        report["restored_from_shared_cache"] = True
        report["exists"] = True

    if target_path.exists():
        if not shared_path.exists() or target_path.stat().st_mtime > shared_path.stat().st_mtime:
            shutil.copy2(target_path, shared_path)
            report["shared_cache_updated"] = True
            report["shared_exists"] = True
    return report


def _module_texture_cache_target(
    *,
    package_name: str,
    constants_module_name: str,
    fallback_filename: str,
) -> Path | None:
    constants_module = sys.modules.get(constants_module_name)
    texture_cache_file = getattr(constants_module, "TEXTURE_CACHE_FILE", None)
    if texture_cache_file:
        return Path(str(texture_cache_file)).expanduser().resolve()

    spec = importlib.util.find_spec(package_name)
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin:
        return None
    return Path(origin).expanduser().resolve().parent / "assets" / fallback_filename


def _merge_reports(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    merged.update(secondary)
    return merged


def _repair_incomplete_texture_cache(
    constants_module: Any,
    *,
    required_keys: set[str],
) -> dict[str, Any]:
    textures = getattr(constants_module, "TEXTURES", None)
    block_sizes = [
        int(getattr(constants_module, "BLOCK_PIXEL_SIZE_AGENT", 0) or 0),
        int(getattr(constants_module, "BLOCK_PIXEL_SIZE_IMG", 0) or 0),
        int(getattr(constants_module, "BLOCK_PIXEL_SIZE_HUMAN", 0) or 0),
    ]
    block_sizes = [item for item in dict.fromkeys(block_sizes) if item > 0]
    report: dict[str, Any] = {
        "texture_block_sizes": block_sizes,
        "repaired_incomplete_cache": False,
        "missing_required_keys": {},
    }
    if not isinstance(textures, dict):
        return report
    missing_by_size: dict[str, list[str]] = {}
    for block_size in block_sizes:
        texture_set = textures.get(block_size)
        if not isinstance(texture_set, dict):
            missing_by_size[str(block_size)] = sorted(required_keys)
            continue
        missing = sorted(required_keys - set(texture_set))
        if missing:
            missing_by_size[str(block_size)] = missing
    report["missing_required_keys"] = missing_by_size
    if not missing_by_size:
        return report
    load_all_textures = getattr(constants_module, "load_all_textures", None)
    if load_all_textures is None:
        report["repair_error"] = "load_all_textures unavailable"
        return report
    for block_size in block_sizes:
        if str(block_size) in missing_by_size:
            textures[block_size] = load_all_textures(block_size)
    report["repaired_incomplete_cache"] = True
    save_compressed_pickle = getattr(constants_module, "save_compressed_pickle", None)
    texture_cache_file = getattr(constants_module, "TEXTURE_CACHE_FILE", None)
    if save_compressed_pickle is None or texture_cache_file is None:
        return report
    try:
        save_compressed_pickle(texture_cache_file, textures)
        report["repair_saved_to_cache"] = True
    except Exception as exc:  # noqa: BLE001
        report["repair_save_error"] = f"{type(exc).__name__}: {exc}"
    return report


def ensure_texture_cache() -> dict[str, Any]:
    os.environ.pop("CRAFTAX_RELOAD_TEXTURES", None)
    shared_root = _shared_cache_root()
    report: dict[str, Any] = {"status": "ok", "full": {}, "classic": {}}

    full_target = _module_texture_cache_target(
        package_name="craftax.craftax",
        constants_module_name="craftax.craftax.constants",
        fallback_filename="texture_cache.pbz2",
    )
    if full_target is not None:
        report["full"] = _sync_shared_cache(
            target_file=full_target,
            shared_root=shared_root,
            namespace="full",
        )

    classic_target = _module_texture_cache_target(
        package_name="craftax.craftax_classic",
        constants_module_name="craftax.craftax_classic.constants",
        fallback_filename="texture_cache_classic.pbz2",
    )
    if classic_target is not None:
        report["classic"] = _sync_shared_cache(
            target_file=classic_target,
            shared_root=shared_root,
            namespace="classic",
        )

    try:
        import craftax.craftax.constants as craftax_constants

        repair_report = _repair_incomplete_texture_cache(
            craftax_constants,
            required_keys=FULL_TEXTURE_KEYS,
        )
        report["full"] = _merge_reports(
            report["full"],
            _sync_shared_cache(
                target_file=craftax_constants.TEXTURE_CACHE_FILE,
                shared_root=shared_root,
                namespace="full",
            ),
        )
        report["full"] = _merge_reports(report["full"], repair_report)
    except Exception as exc:  # pragma: no cover - optional dependency
        report["full"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import craftax.craftax_classic.constants as craftax_classic_constants

        report["classic"] = _merge_reports(
            report["classic"],
            _sync_shared_cache(
                target_file=craftax_classic_constants.TEXTURE_CACHE_FILE,
                shared_root=shared_root,
                namespace="classic",
            ),
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        report["classic"] = {"error": f"{type(exc).__name__}: {exc}"}
    return report
