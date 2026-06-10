"""Project import workflow for the P1AM backend."""

from __future__ import annotations

import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from fastapi import HTTPException, UploadFile
from models import PlantArea, PlantEquipment, PlantUnit, TagDefinitionDb
from parsers.indusoft_parser import parse_indusoft_tags
from parsers.plc_map_parser import parse_plc_map
from plant_model import TagDefinition
from sqlmodel import Session, select

MAX_IMPORT_UPLOAD_BYTES = int(
    os.environ.get("P1AM_MAX_IMPORT_BYTES", str(50 * 1024 * 1024))
)
MAX_IMPORT_MEMBERS = int(os.environ.get("P1AM_MAX_IMPORT_MEMBERS", "10000"))
MAX_IMPORT_MEMBER_BYTES = int(
    os.environ.get("P1AM_MAX_IMPORT_MEMBER_BYTES", str(100 * 1024 * 1024))
)
MAX_IMPORT_TOTAL_BYTES = int(
    os.environ.get("P1AM_MAX_IMPORT_TOTAL_BYTES", str(500 * 1024 * 1024))
)
MAX_IMPORT_COMPRESSION_RATIO = float(os.environ.get("P1AM_MAX_IMPORT_RATIO", "100.0"))
IMPORT_CHUNK_BYTES = 1024 * 1024


async def import_project_archive(
    file: UploadFile,
    db: Session,
    reload_tags: Callable[[Session], None],
) -> dict[str, Any]:
    """Import a zipped project into the plant model database atomically."""
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_file_path = temp_path / "uploaded.zip"
        await _write_bounded_upload(file, zip_file_path)
        _extract_project_zip(zip_file_path, temp_path)
        tags, mapped_count = _parse_project_files(temp_path)
        result = _replace_plant_configuration(db, tags, mapped_count)
        reload_tags(db)
        return result


async def _write_bounded_upload(file: UploadFile, zip_file_path: Path) -> None:
    total = 0
    with zip_file_path.open("wb") as output:
        while True:
            chunk = await file.read(IMPORT_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_IMPORT_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Uploaded file exceeds the maximum allowed size "
                        f"({MAX_IMPORT_UPLOAD_BYTES} bytes)."
                    ),
                )
            output.write(chunk)


def _extract_project_zip(zip_file_path: Path, temp_path: Path) -> None:
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            _safe_extract_zip(zip_ref, temp_path)
    except HTTPException:
        raise
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {exc}") from exc


def _safe_extract_zip(zip_ref: zipfile.ZipFile, dest: Path) -> None:
    """Validate and extract a zip with path, member, size, and ratio budgets."""
    infos = zip_ref.infolist()
    if len(infos) > MAX_IMPORT_MEMBERS:
        raise HTTPException(
            status_code=413,
            detail=f"Archive has too many members (> {MAX_IMPORT_MEMBERS}).",
        )

    total_uncompressed = 0
    dest_root = dest.resolve()
    for info in infos:
        _validate_member_path(info.filename, dest_root)
        total_uncompressed += _validate_member_budget(info)
        if total_uncompressed > MAX_IMPORT_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Archive uncompressed size exceeds total budget.",
            )

    zip_ref.extractall(dest_root)


def _validate_member_path(filename: str, dest_root: Path) -> None:
    member_path = Path(filename)
    target = (dest_root / member_path).resolve()
    try:
        target.relative_to(dest_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Archive member escapes import directory: {filename!r}.",
        ) from exc
    if member_path.is_absolute() or ".." in member_path.parts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsafe archive member path: {filename!r}.",
        )


def _validate_member_budget(info: zipfile.ZipInfo) -> int:
    if info.file_size > MAX_IMPORT_MEMBER_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Archive member '{info.filename}' exceeds size limit.",
        )
    if info.compress_size > 0:
        ratio = info.file_size / info.compress_size
        if ratio > MAX_IMPORT_COMPRESSION_RATIO:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Archive member '{info.filename}' has a suspicious "
                    "compression ratio (possible zip bomb)."
                ),
            )
    return info.file_size


def _parse_project_files(temp_path: Path) -> tuple[list[TagDefinition], int]:
    tag_json_path = _find_tag_json(temp_path)
    tags = _parse_tags(tag_json_path)
    plc_map = _parse_plc_maps(temp_path)
    return tags, _apply_plc_map(tags, plc_map)


def _find_tag_json(temp_path: Path) -> Path:
    for path in temp_path.rglob("*"):
        if path.name.lower() == "tagl.json":
            return path
    raise HTTPException(
        status_code=400,
        detail="Could not find 'tagl.json' in the uploaded ZIP file.",
    )


def _parse_tags(tag_json_path: Path) -> list[TagDefinition]:
    try:
        return cast("list[TagDefinition]", parse_indusoft_tags(tag_json_path))
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse tags: {exc}"
        ) from exc


def _parse_plc_maps(temp_path: Path) -> dict[str, dict[str, Any]]:
    plc_map: dict[str, dict[str, Any]] = {}
    for path in temp_path.rglob("*.SDV"):
        try:
            plc_map.update(parse_plc_map(path))
        except (OSError, ValueError, TypeError):
            continue
    return plc_map


def _apply_plc_map(
    tags: list[TagDefinition], plc_map: dict[str, dict[str, Any]]
) -> int:
    mapped_count = 0
    for tag in tags:
        if tag.name not in plc_map:
            continue
        mapping = plc_map[tag.name]
        tag.register_type = mapping.get("register_type")
        tag.register_num = mapping.get("register_num")
        tag.data_format = mapping.get("data_format")
        if mapping.get("rw_mode"):
            tag.rw_mode = mapping.get("rw_mode")
        if mapping.get("scale_factor") is not None:
            tag.scale_factor = mapping.get("scale_factor")
        mapped_count += 1
    return mapped_count


def _replace_plant_configuration(
    db: Session,
    tags: list[TagDefinition],
    mapped_count: int,
) -> dict[str, Any]:
    areas: dict[str, PlantArea] = {}
    units: dict[str, PlantUnit] = {}
    equipment: dict[str, PlantEquipment] = {}

    try:
        _delete_existing_config(db)
        for tag in tags:
            area_name, unit_name, equipment_name = _tag_hierarchy(tag.name)
            area = _area(db, areas, area_name)
            unit = _unit(db, units, area_name, unit_name, area)
            equip = _equipment(
                db, equipment, area_name, unit_name, equipment_name, unit
            )
            db.add(_tag_row(tag, equip))
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except (OSError, ValueError, TypeError) as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Database save failed: {exc}"
        ) from exc

    return {
        "status": "success",
        "tags_imported": len(tags),
        "mapped_registers": mapped_count,
        "areas_created": list(areas.keys()),
        "units_created": [unit.name for unit in units.values()],
        "equipment_created": [equip.name for equip in equipment.values()],
    }


def _delete_existing_config(db: Session) -> None:
    for model in (TagDefinitionDb, PlantEquipment, PlantUnit, PlantArea):
        for row in db.exec(select(model)).all():
            db.delete(row)
    db.flush()


def _tag_hierarchy(tag_name: str) -> tuple[str, str, str]:
    for separator in ("_", "."):
        parts = tag_name.split(separator)
        if len(parts) >= 4:
            return parts[0], parts[1], parts[2]
    return "Default Area", "Default Unit", "Default Equipment"


def _area(db: Session, areas: dict[str, PlantArea], name: str) -> PlantArea:
    if name not in areas:
        area = PlantArea(name=name)
        db.add(area)
        db.flush()
        db.refresh(area)
        areas[name] = area
    return areas[name]


def _unit(
    db: Session,
    units: dict[str, PlantUnit],
    area_name: str,
    unit_name: str,
    area: PlantArea,
) -> PlantUnit:
    key = f"{area_name}:{unit_name}"
    if key not in units:
        unit = PlantUnit(name=unit_name, area_id=area.id)
        db.add(unit)
        db.flush()
        db.refresh(unit)
        units[key] = unit
    return units[key]


def _equipment(
    db: Session,
    equipment: dict[str, PlantEquipment],
    area_name: str,
    unit_name: str,
    equipment_name: str,
    unit: PlantUnit,
) -> PlantEquipment:
    key = f"{area_name}:{unit_name}:{equipment_name}"
    if key not in equipment:
        equip = PlantEquipment(name=equipment_name, unit_id=unit.id)
        db.add(equip)
        db.flush()
        db.refresh(equip)
        equipment[key] = equip
    return equipment[key]


def _tag_row(tag: TagDefinition, equipment: PlantEquipment) -> TagDefinitionDb:
    return TagDefinitionDb(
        name=tag.name,
        tag_type=tag.tag_type,
        description=tag.description,
        rw_mode=tag.rw_mode,
        register_type=tag.register_type,
        register_num=tag.register_num,
        data_format=tag.data_format,
        scale_factor=tag.scale_factor,
        equipment_id=equipment.id,
    )
