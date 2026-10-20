# -*- coding: utf-8 -*-
"""Read one simulation case at a time from the SSI simulation matrix."""

from pathlib import Path

from soil_profile_reader_v4 import read_soil_profile
from units_v4 import get_units



MATRIX_SHEET = "Simulation_Matrix"
MAX_PILE_GROUP_WIDTH_FT = 50.0


def _required(row, name):
    value = row.get(name)
    if value is None or value == "":
        raise ValueError(f"Missing required matrix value: {name}")
    return value


def _required_any(row, names):
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    raise ValueError(f"Missing required matrix value; expected one of: {', '.join(names)}")


def _integer_count(row, name, minimum):
    value = int(round(float(_required(row, name))))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _case_soil_geometry(case, soil_layers, ft):
    """Return soil/profile elevations and the case's actual tip elevations.

    Profile elevations are in the profile's local datum. Align the profile
    surface with the model ground surface before comparing node elevations.
    """
    if not soil_layers:
        return {}

    profile_surface_z = float(soil_layers[0]["top"])
    profile_bottom_z = float(soil_layers[-1]["bottom"])
    ground_surface_z = 102.0 * ft
    profile_to_model_offset = ground_surface_z - profile_surface_z
    soil_total_depth = profile_surface_z - profile_bottom_z

    column_top_z = 104.0 * ft
    column_bottom_z = column_top_z - case["H"]
    shaft_top_z = column_bottom_z - 2.0 * ft
    remaining_length = case["L_P"] - case["H"]
    shaft_length = remaining_length * (38.0 / 70.0)
    socket_length = remaining_length * (32.0 / 70.0)
    shaft_bottom_z = shaft_top_z - shaft_length
    socket_top_z = shaft_bottom_z - 2.0 * ft
    socket_bottom_z = socket_top_z - socket_length
    pile_bottom_z = column_top_z - 0.60 * case["L_P"]
    model_soil_bottom_z = profile_bottom_z + profile_to_model_offset

    return {
        "ground_surface_z": ground_surface_z,
        "soil_profile_surface_z": profile_surface_z,
        "soil_profile_bottom_z": profile_bottom_z,
        "soil_profile_to_model_offset": profile_to_model_offset,
        "soil_total_depth": soil_total_depth,
        "model_soil_bottom_z": model_soil_bottom_z,
        "socket_bottom_z": socket_bottom_z,
        "pile_bottom_z": pile_bottom_z,
        "socket_tip_depth_below_ground": ground_surface_z - socket_bottom_z,
        "pile_tip_depth_below_ground": ground_surface_z - pile_bottom_z,
        "socket_tip_below_profile": socket_bottom_z < model_soil_bottom_z - 1.0e-8,
        "pile_tip_below_profile": pile_bottom_z < model_soil_bottom_z - 1.0e-8,
        "ShaftLength": shaft_length,
        "SocketLength": socket_length,
        "HPileLength": 0.60 * case["L_P"],
    }


def write_case_input_file(case, input_file, mpi_rank=None):
    """Write effective case inputs and normalized soil profile to input_params.txt."""
    rank_text = "serial" if mpi_rank is None else str(mpi_rank)
    excluded = {"soil_layers", "validation_error"}
    lines = [
        f"MPI rank: {rank_text}",
        f"Simulation_No: {case.get('simulation_no')}",
        "Excel/model inputs:",
    ]
    for key, value in case.items():
        if key not in excluded:
            lines.append(f"  {key}: {value}")

    if case.get("validation_error"):
        lines.append(f"Validation error: {case['validation_error']}")

    layers = case.get("soil_layers") or []
    lines.append(f"Soil layers: {len(layers)}")
    for index, layer in enumerate(layers, start=1):
        lines.append(
            f"  Layer {index}: uscs={layer.get('uscs')}, "
            f"top={layer.get('top')}, bottom={layer.get('bottom')}, "
            f"c/su={layer.get('su')}, phi={layer.get('phi')}, "
            f"gamma={layer.get('gamma')}, G={layer.get('G')}, "
            f"eps50={layer.get('eps50')}"
        )

    path = Path(input_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def iter_simulation_cases(
    matrix_file=None,
    project_dir=None,
    load_soil_profile=True,
    simulation_numbers=None,
    eqs_dir=None,
    soil_profiles_dir=None,
):
    """Yield simulation cases from the matrix without loading all rows.

    The workbook profile column selects ``<soil_profiles_dir>/<profile>.txt`` and
    the EQ column selects ``<eqs_dir>/<EQ>.out``. If directories are omitted,
    they default to ``soil_profiles`` and ``EQs`` inside ``project_dir``.
    The matrix's Number of bents column is intentionally ignored; total bents
    are derived as number of spans plus one.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as error:
        raise ImportError(
            "Reading the simulation matrix requires openpyxl."
        ) from error

    project_path = Path(project_dir).resolve() if project_dir else Path(__file__).resolve().parent
    soil_profile_root = Path(soil_profiles_dir).expanduser() if soil_profiles_dir else project_path / "soil_profiles"
    eq_root = Path(eqs_dir).expanduser() if eqs_dir else project_path / "EQs"
    if not soil_profile_root.is_absolute():
        soil_profile_root = (project_path / soil_profile_root).resolve()
    if not eq_root.is_absolute():
        eq_root = (project_path / eq_root).resolve()
    selected_simulations = None if simulation_numbers is None else {
        int(number) for number in simulation_numbers
    }
    workbook_path = Path(matrix_file) if matrix_file else project_path / "SSI_Simulation_Matrix1.xlsx"
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Simulation matrix not found: {workbook_path}")

    m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    if MATRIX_SHEET not in workbook.sheetnames:
        workbook.close()
        raise ValueError(f"Worksheet {MATRIX_SHEET!r} was not found in {workbook_path}")

    worksheet = workbook[MATRIX_SHEET]
    rows = worksheet.iter_rows(values_only=True)
    try:
        headers = [str(value).strip() for value in next(rows)]
    except StopIteration:
        workbook.close()
        return

    try:
        for values in rows:
            if not any(value is not None and value != "" for value in values):
                continue

            row = dict(zip(headers, values))
            simulation_no = int(_required(row, "Simulation_No"))
            if selected_simulations is not None and simulation_no not in selected_simulations:
                continue

            profile_name = str(_required(row, "Soil Profile (based on SPT-N)")).strip()
            eq_name = str(_required(row, "EQ")).strip()
            number_of_spans = _integer_count(row, "Number of spans (-)", 1)
            number_of_piles = _integer_count(row, "Number of piles (-)", 2)

            H = float(_required(row, "Column height H (m)")) * m
            D = float(_required(row, "Column/pile diameter D (m)")) * m
            pl = float(_required(row, "Longitudinal steel ratio pl (%)"))
            fc = float(_required(row, "Concrete strength fc (MPa)")) * MPa
            fy = float(_required(row, "Steel yield strength fy (MPa)")) * MPa
            span_length = float(_required(row, "Span length (m)")) * m
            pile_spacing = float(_required(row, "Pile spacing (m)")) * m
            L_P = float(_required_any(row, ("L_P (m)", "Pile embedment length Lp (m)"))) * m
            target_pga = float(_required(row, "Peak ground acceleration PGA (g)"))
            incident_angle_deg = float(_required(row, "Angle of incidence (deg)"))
            # EQ files contain acceleration samples only. Their time step is
            # supplied by the matrix when available, otherwise preserve the
            # existing OpenSees setup's 0.005 s assumption.
            dt_value = next((row.get(name) for name in (
                "Ground motion time step (s)", "EQ time step (s)", "dt (s)",
            ) if row.get(name) not in (None, "")), 0.005)
            ground_motion_dt_s = float(dt_value)

            validation_errors = []
            if H <= 0 or D <= 0 or span_length <= 0 or pile_spacing <= 0:
                validation_errors.append("H, D, span length, and pile spacing must be positive")
            if pl <= 0 or fc <= 0 or fy <= 0 or target_pga <= 0:
                validation_errors.append("pl, fc, fy, and PGA must be positive")
            if not 0.0 <= incident_angle_deg <= 360.0:
                validation_errors.append(
                    "Angle of incidence must be between 0 and 360 degrees; "
                    f"got {incident_angle_deg}"
                )
            if ground_motion_dt_s <= 0:
                validation_errors.append("Ground-motion time step must be positive")
            if L_P <= H:
                validation_errors.append(f"L_P must be greater than H; got L_P={L_P/m} m and H={H/m} m")

            total_pile_width = (number_of_piles - 1) * pile_spacing
            if total_pile_width > MAX_PILE_GROUP_WIDTH_FT * ft:
                validation_errors.append(
                    f"Pile group width is {total_pile_width/ft:.3f} ft; "
                    f"maximum is {MAX_PILE_GROUP_WIDTH_FT:.3f} ft"
                )

#            profile_path = project_path / "soil_profiles" / (
#                profile_name if profile_name.lower().endswith('.txt') else profile_name + '.txt'
#            )
            profile_name = str(
                    _required(row, "Soil Profile (based on SPT-N)")
                    ).strip()

            soil_profile_dir = soil_profile_root

            profile_candidates = [
                              soil_profile_dir / f"{profile_name}.csv",
                              soil_profile_dir / f"{profile_name}.txt",
                              ]

            profile_override = row.get("Soil Profile file path")
            if profile_override not in (None, ""):
                profile_path = Path(str(profile_override)).expanduser()
                if not profile_path.is_absolute():
                    profile_path = (project_path / profile_path).resolve()
            else:
                profile_path = next(
                              (path for path in profile_candidates if path.is_file()),
                                profile_candidates[0],
                                  )
            
            # EQ file
            eq_override = row.get("EQ file path")
            if eq_override not in (None, ""):
                eq_path = Path(str(eq_override)).expanduser()
                if not eq_path.is_absolute():
                    eq_path = (project_path / eq_path).resolve()
            else:
                eq_path = eq_root / (
                    eq_name if eq_name.lower().endswith('.out') else eq_name + '.out'
                )
            soil_layers = None
            try:
                soil_layers = read_soil_profile(profile_path) if load_soil_profile else None
            except Exception as error:
                validation_errors.append(str(error))

            case = {
                "simulation_no": simulation_no,
                "soil_profile_name": profile_name,
                "soil_profile_file": profile_path,
                "soil_layers": soil_layers,
                "H": H,
                "D": D,
                "pl": pl,
                "fc": fc,
                "fy": fy,
                "number_of_spans": number_of_spans,
                "span_length": span_length,
                "number_of_bents": number_of_spans + 1,
                "number_of_piles": number_of_piles,
                "pile_spacing": pile_spacing,
                "total_pile_width": total_pile_width,
                "L_P": L_P,
                "HPileLength": 0.60 * L_P,
                "target_pga": target_pga,
                "incident_angle_deg": incident_angle_deg,
                "ground_motion_dt_s": ground_motion_dt_s,
                "EQ": eq_name,
                "EQPath": eq_path,
                "validation_error": "; ".join(validation_errors),
            }
            if soil_layers and L_P > H:
                soil_geometry = _case_soil_geometry(case, soil_layers, ft)
                case.update(soil_geometry)
                if L_P > H and soil_geometry:
                    if soil_geometry["socket_tip_below_profile"] or soil_geometry["pile_tip_below_profile"]:
                        case["bottom_layer_fallback"] = True
                        case["bottom_layer_fallback_uscs"] = soil_layers[-1]["uscs"]
                    else:
                        case["bottom_layer_fallback"] = False
            yield case
    finally:
        workbook.close()
