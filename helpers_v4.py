# -*- coding: utf-8 -*-
import math
import numpy as np

def rotate_node(x, y, theta_degrees):
    """
    Rotates a point (x, y) about the origin (0,0).
    
    Args:
        x (float): Original x-coordinate.
        y (float): Original y-coordinate.
        theta_degrees (float): Rotation angle in degrees.
        
    Returns:
        tuple: (x_rotated, y_rotated)
    """
    # Convert angle to radians for math functions
    theta_rad = math.radians(theta_degrees)
    
    # Standard 2D rotation matrix logic
    x_rot = x * math.cos(theta_rad) - y * math.sin(theta_rad)
    y_rot = x * math.sin(theta_rad) + y * math.cos(theta_rad)
    
    return x_rot, y_rot


def transverse_capbeam_locations(pile_locations, bearing_locations=()):
    """Return sorted transverse cap-beam coordinates.

    Every pile/column location is retained, and the bearing locations are
    inserted into the same cap-beam mesh.  Consequently the cap beam always
    has a node exactly at each pile/column location while preserving the
    requested bearing positions.
    """
    combined = [float(value) for value in pile_locations]
    combined.extend(float(value) for value in bearing_locations)
    if not combined:
        raise ValueError("At least one pile or bearing location is required")

    combined.sort()
    unique = []
    for value in combined:
        if not unique or not np.isclose(value, unique[-1], rtol=0.0, atol=1.0e-9):
            unique.append(value)
    return np.asarray(unique, dtype=float)
