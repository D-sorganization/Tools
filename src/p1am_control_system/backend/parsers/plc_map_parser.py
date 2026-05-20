from pathlib import Path

RW_MODE_LABELS = {
    "0": "Read-only",
    "2": "Read/Write",
}


def parse_register(register_str: str) -> tuple[str, int | None, str]:
    if not register_str:
        return ("", None, "")

    parts = register_str.split(":")
    reg_type = parts[0] if len(parts) > 0 else ""
    reg_number = None
    data_format = parts[2] if len(parts) > 2 else ""

    if len(parts) > 1:
        try:
            reg_number = int(parts[1])
        except ValueError:
            reg_number = None

    return (reg_type, reg_number, data_format)


def parse_plc_map(sdv_path: Path) -> dict[str, dict]:
    """Parses KOYO.SDV or MODBU.SDV and returns a dict mapping

    tag names to their register info.
    """
    if not sdv_path.exists():
        raise FileNotFoundError(f"PLC driver file not found at {sdv_path}")

    with open(sdv_path, encoding="utf-16") as f:
        lines = f.readlines()

    tag_map = {}
    for line in lines:
        stripped = line.strip()
        if "\t" not in stripped:
            continue

        parts = [p.strip() for p in stripped.split("\t")]
        if len(parts) < 4:
            continue

        tag_name = parts[0]
        register = parts[2] if len(parts) > 2 else ""
        rw_mode_raw = parts[3] if len(parts) > 3 else ""
        scale_raw = parts[5] if len(parts) > 5 else ""

        reg_type, reg_num, data_fmt = parse_register(register)

        scale_factor = None
        if scale_raw:
            try:
                scale_factor = float(scale_raw)
            except ValueError:
                pass

        tag_map[tag_name] = {
            "register_type": reg_type,
            "register_num": reg_num,
            "data_format": data_fmt,
            "rw_mode": RW_MODE_LABELS.get(rw_mode_raw, rw_mode_raw),
            "scale_factor": scale_factor,
        }

    return tag_map
