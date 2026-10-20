# -*- coding: utf-8 -*-
"""Read depth-dependent spring soil profiles from text files.

The text file contains one layer per row:

    soil_type top_layer_m bottom_layer_m c_kpa phi_deg
    SM        30            29             5     30

The returned dictionaries use the field names and internal units_v4 expected by
Define_str._JoinSoilPile() and Define_str._get_pile_soil_params().
"""

from pathlib import Path
import csv

from units_v4 import get_units


def read_soil_profile(profile_file):
    """Read and convert one soil-profile text file.

    Parameters
    ----------
    profile_file : str or pathlib.Path
        Text file containing soil type, top elevation, bottom elevation,
        cohesion in kPa, and friction angle in degrees.

    Returns
    -------
    list of dict
        One normalized dictionary for every soil layer.
    """
    m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()
    gravity = 9.81 * m / sec**2

    # These defaults supply quantities used by the existing spring equations
    # when the profile file only specifies soil type, c, and phi.
    default_by_type = {
        "sand": {
            "gamma": 1800.0 * kg / m**3 * gravity,
            "G": 30.0 * MPa,
            "eps50": 0.0,
        },
        "clay": {
            "gamma": 1800.0 * kg / m**3 * gravity,
            "G": 30.0 * MPa,
            "eps50": 0.01,
        },
    }

    profile_path = Path(profile_file)
    if not profile_path.is_file():
        raise FileNotFoundError(f"Soil profile file not found: {profile_path}")

    layers = []
    with profile_path.open("r", encoding="utf-8") as profile_stream:
        for line_number, raw_line in enumerate(profile_stream, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            # Soil descriptions may contain spaces, for example
            # ``M desne sand SM``.  The final four fields are always the
            # numeric profile fields, so parse those from the right.
            if "," in line:
                fields = [field.strip() for field in next(csv.reader([line]))]
            else:
                fields = line.split()

            if len(fields) < 5:
                # Allow a descriptive/header line such as:
                # soil_type top_layer_m bottom_layer_m c_kpa phi_deg
                continue

            try:
                soil_type = " ".join(fields[:-4]).strip()
                top_m = float(fields[-4])
                bottom_m = float(fields[-3])
                cohesion_kpa = float(fields[-2])
                phi_degrees = float(fields[-1])
            except ValueError:
                # Header text is ignored; malformed data rows are reported.
                if line_number == 1 or fields[0].lower() in {"soil", "soil_type"}:
                    continue
                raise ValueError(
                    f"Invalid soil-profile row {line_number} in {profile_path}: {raw_line.rstrip()}"
                )

            normalized_type = soil_type.strip().upper()
            soil_tokens = set(normalized_type.replace("-", " ").split())
            sand_markers = {
                "SAND", "GRAVEL", "S", "G", "SP", "SW", "SM", "SC",
                "GP", "GW", "GM", "GC",
            }
            uscs_codes = {
                "SP", "SW", "SM", "SC",
                "GP", "GW", "GM", "GC",
                "ML", "MH", "CL", "CH",
            }
            spring_type = (
                "sand"
                if soil_tokens.intersection(sand_markers)
                else "clay"
            )

            # Keep the original description for traceability, but pass only
            # a canonical USCS code to the existing structure model.  The
            # structure model determines sand/clay from the first character
            # of ``uscs`` and therefore must not receive descriptions such as
            # ``M desne sand SM`` or ``stiff ML``.
            uscs_code = next(
                (token for token in reversed(
                    normalized_type.replace("-", " ").split()
                ) if token in uscs_codes),
                "SM" if spring_type == "sand" else "CL",
            )
            defaults = default_by_type[spring_type]

            layers.append({
                "layer_id": len(layers) + 1,
                "uscs": uscs_code,
                "soil_description": normalized_type,
                "top": top_m * m,
                "bottom": bottom_m * m,
                "su": cohesion_kpa * kPa,
                "phi": phi_degrees,
                "gamma": defaults["gamma"],
                "G": defaults["G"],
                "eps50": defaults["eps50"],
            })

    if not layers:
        raise ValueError(f"No soil layers were found in {profile_path}")

    for upper, lower in zip(layers, layers[1:]):
        if upper["bottom"] != lower["top"]:
            raise ValueError(
                f"Elevation gap/overlap between layers {upper['layer_id']} and "
                f"{lower['layer_id']} in {profile_path}"
            )

    return layers
