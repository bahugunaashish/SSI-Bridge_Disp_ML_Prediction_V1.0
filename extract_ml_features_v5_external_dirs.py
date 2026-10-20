"""Build OpenSees input graphs for ML without running any analysis."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import openseespy.opensees as op

from simulation_matrix_reader_v5_external_dirs import iter_simulation_cases
from ssi_ml_surrogate_v1.hdf5_data import write_input_database


def discover_simulation_numbers(results_dir):
    """Return numeric IDs from exact simulation_N result-folder names."""
    root = Path(results_dir)
    if not root.is_dir():
        return []
    numbers = set()
    for path in root.iterdir():
        match = re.fullmatch(r"simulation_(\d+)", path.name)
        if match and path.is_dir():
            numbers.add(int(match.group(1)))
    return sorted(numbers)


def find_displacement_recorder(simulation_dir):
    """Find the real TXT displacement recorder; PVD files do not qualify."""
    root = Path(simulation_dir)
    matches = sorted(
        path for path in root.rglob("*.txt")
        if path.is_file() and (
            "structure_disp" in path.stem.lower()
            or path.stem.lower().endswith("_disp")
        )
    )
    return matches[0] if matches else None


def extract(matrix_file=None, project_dir=None, output=None, start=None, end=None, dt=0.005,
            simulation_numbers=None, available_results_dir=None, prediction_only=False,
            eqs_dir=None, soil_profiles_dir=None):
    project = Path(project_dir).resolve() if project_dir else Path(__file__).resolve().parent
    matrix = Path(matrix_file).resolve() if matrix_file else project / "SSI_Simulation_Matrix1.xlsx"
    destination = Path(output).resolve() if output else project / "results" / "ml_inputs" / "ml_input_features.h5"
    skipped = []
    results_root = Path(available_results_dir) if available_results_dir else project / "results"
    if not results_root.is_absolute():
        results_root = (project / results_root).resolve()
    discovered_numbers = discover_simulation_numbers(results_root)
    if prediction_only:
        # Inference inputs are defined by the matrix, not by completed result
        # folders. This lets users predict a new EQ case before running OpenSees.
        verified_numbers = sorted({int(n) for n in simulation_numbers}) if simulation_numbers is not None else None
        selected_numbers = verified_numbers
    else:
        # Training feature cases must have completed displacement TXT data.
        selected_numbers = sorted({int(n) for n in simulation_numbers}) if simulation_numbers is not None else discovered_numbers
        verified_numbers = []
        for number in selected_numbers:
            sim_dir = results_root / f"simulation_{number}"
            if not sim_dir.is_dir():
                skipped.append({"simulation_no": number, "reason": f"Simulation folder not found: {sim_dir}"})
                continue
            disp_path = find_displacement_recorder(sim_dir)
            if disp_path is None:
                skipped.append({
                    "simulation_no": number,
                    "reason": f"Displacement TXT recorder (*_disp.txt / *Structure_Disp*.txt) not found under {sim_dir}",
                })
                print(f"SKIP simulation_{number}: displacement TXT recorder not found")
                continue
            verified_numbers.append(number)
            print(f"Verified simulation_{number} displacement recorder: {disp_path}")
    if not prediction_only and not verified_numbers:
        report = destination.with_suffix(destination.suffix + ".skipped.json")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"selected_simulation_numbers": selected_numbers,
                                      "skipped_simulations": skipped}, indent=2), encoding="utf-8")
        raise ValueError(f"No selected simulation has a displacement TXT recorder; see {report}")
    from main_v4 import main

    extracted_numbers = []
    def cases_to_write():
        for case in iter_simulation_cases(
            matrix_file=matrix, project_dir=project,
            simulation_numbers=verified_numbers,
            eqs_dir=eqs_dir, soil_profiles_dir=soil_profiles_dir,
        ):
            number = int(case["simulation_no"])
            if start is not None and number < start:
                continue
            if end is not None and number > end:
                continue
            if case.get("validation_error"):
                raise ValueError(f"Simulation {number} is invalid: {case['validation_error']}")
            case["ground_motion_dt_s"] = float(case.get("ground_motion_dt_s", dt))
            if not Path(case["EQPath"]).is_file():
                raise FileNotFoundError(f"Ground-motion file for simulation {number} not found: {case['EQPath']}")

            # Keep temporary feature-only working files beside the selected
            # HDF5 destination instead of writing into project/results.
            run_dir = destination.parent / "feature_work" / f"simulation_{number}_features"
            run_dir.mkdir(parents=True, exist_ok=True)
            old_cwd = Path.cwd()
            try:
                os.chdir(run_dir)
                graph = main(case=case, plot_results=False, feature_only=True)
                extracted_numbers.append(number)
                print(f"Extracted simulation {number}: {len(graph.node_ids)} nodes, {graph.edge_index.shape[1]} directed edges")
                yield f"simulation_{number}", graph
            finally:
                os.chdir(old_cwd)
                # Reset the OpenSees domain between matrix cases. wipe() removes
                # model state; it does not run an analysis.
                op.wipe()

    destination.parent.mkdir(parents=True, exist_ok=True)
    write_input_database(destination, cases_to_write(), schema={
        "description": "OpenSees-built SSI model input graphs; no response targets",
        "mode": "prediction_only" if prediction_only else "training",
        "analysis_executed": False,
        "ground_motion_dt_s": "per case from the simulation matrix; default 0.005 s",
        "selected_simulation_numbers": verified_numbers,
        "discovered_result_simulation_numbers": discovered_numbers,
    })
    if not extracted_numbers:
        destination.unlink(missing_ok=True)
        raise ValueError("None of the selected result-folder simulation numbers matched usable rows in the matrix")
    unmatched = sorted(set(verified_numbers or []) - set(extracted_numbers))
    if unmatched:
        skipped.extend({"simulation_no": number,
                        "reason": "No usable matching simulation matrix row or excluded by start/end range"}
                       for number in unmatched)
        print(f"Skipped simulation numbers with no usable matrix row or outside the selected range: {unmatched}")
    report = destination.with_suffix(destination.suffix + ".skipped.json")
    report.write_text(json.dumps({"selected_simulation_numbers": selected_numbers,
                                  "extracted_simulation_numbers": extracted_numbers,
                                  "skipped_simulations": skipped}, indent=2), encoding="utf-8")
    return destination


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-file", default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--simulations", type=int, nargs="+", default=None,
                        help="Select exact, possibly non-consecutive simulation numbers")
    parser.add_argument("--available-results-dir", default=None,
                        help="Discover simulation_N folders here (default: <project-dir>/results); used when --simulations is omitted")
    parser.add_argument("--eqs-dir", default=None,
                        help="Earthquake files directory (default: <project-dir>/EQs)")
    parser.add_argument("--soil-profiles-dir", default=None,
                        help="Soil profile directory (default: <project-dir>/soil_profiles)")
    parser.add_argument("--prediction-only", action="store_true",
                        help="Build inputs from matrix rows without requiring result folders or response TXT files")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="Fallback time step in seconds if a matrix row has no time-step value")
    args = parser.parse_args()
    print(extract(args.matrix_file, args.project_dir, args.output, args.start, args.end,
                  args.dt, args.simulations, args.available_results_dir, args.prediction_only,
                  args.eqs_dir, args.soil_profiles_dir))


if __name__ == "__main__":
    main_cli()
