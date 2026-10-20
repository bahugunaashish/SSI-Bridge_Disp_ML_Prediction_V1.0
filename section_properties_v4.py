# -*- coding: utf-8 -*-
"""Helpers for converting simulation section variables into section inputs."""

import math


def calculate_longitudinal_reinforcement(D, steel_ratio_percent, num_bars=18):
    """Return the fixed bar count and equivalent area per bar."""
    if D <= 0:
        raise ValueError("Section diameter D must be positive")
    if steel_ratio_percent <= 0:
        raise ValueError("Longitudinal steel ratio pl must be positive")
    if num_bars <= 0:
        raise ValueError("Number of longitudinal bars must be positive")

    gross_area = math.pi * D**2 / 4.0
    total_steel_area = (steel_ratio_percent / 100.0) * gross_area
    bar_area = total_steel_area / num_bars
    equivalent_bar_diameter = math.sqrt(4.0 * bar_area / math.pi)

    return {
        "numBarsSec": int(num_bars),
        "barAreaSec": bar_area,
        "totalSteelArea": total_steel_area,
        "equivalentBarDiameter": equivalent_bar_diameter,
    }
