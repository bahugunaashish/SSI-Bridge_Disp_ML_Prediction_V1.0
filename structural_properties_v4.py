# -*- coding: utf-8 -*-
"""Simulation-specific structural mass and load calculations."""

import math


def calculate_structural_loads(
    *,
    D,
    ColumnLength,
    ShaftLength,
    SocketLength,
    HPileLength,
    HArea,
    Width_CapBeam,
    Depth_CapBeamSec,
    CapBeamLength,
    WidthEcapBeam,
    DepthEcapBeamSec,
    EcapBeamLength,
    DiaPhrgmArea,
    DeckArea,
    ParapetMass,
    FWS,
    SIPF,
    AdditionalLoad,
    GamaConc,
    GamaSteel,
    g,
    kN,
    m,
    mm,
    ft,
):
    """Calculate the original model's mass, self-weight, and gravity loads.

    The equations are unchanged from the original input script. Only the
    geometry values are supplied per simulation.
    """
    columnMass = ((math.pi * D**2 / 4) * ColumnLength * GamaConc) / g
    columnWt = columnMass * g / ColumnLength

    ShaftMass = (
        ((math.pi * D**2 / 4) * ShaftLength * GamaConc) / g
        + ((math.pi * (D**2 - (D - 4 * mm)**2) / 4) * ShaftLength * GamaSteel) / g
    )
    ShaftWt = ShaftMass * g / ShaftLength

    SocketMass = ((math.pi * D**2 / 4) * SocketLength * GamaConc) / g
    SocketWt = SocketMass * g / SocketLength

    PileMass = (HArea * HPileLength * GamaSteel) / g
    PileWt = PileMass * g / HPileLength

    ECapBeamMass = (WidthEcapBeam * DepthEcapBeamSec * EcapBeamLength * GamaConc) / g
    ECapBeamWt = ECapBeamMass * g / EcapBeamLength

    CapBeamMass = (Width_CapBeam * Depth_CapBeamSec * CapBeamLength * GamaConc) / g
    CapBeamWt = CapBeamMass * g / CapBeamLength

    DiaPhragmLoad = DiaPhrgmArea * GamaConc * 100
    SSTWT = (
        DeckArea * GamaConc
        + 2 * 7.0051 * kN / m
        + FWS
        + ParapetMass * g / (80 * ft)
        + SIPF
    )

    return {
        "columnMass": columnMass,
        "columnWt": columnWt,
        "ShaftMass": ShaftMass,
        "ShaftWt": ShaftWt,
        "SocketMass": SocketMass,
        "SocketWt": SocketWt,
        "PileMass": PileMass,
        "PileWt": PileWt,
        "ECapBeamMass": ECapBeamMass,
        "ECapBeamWt": ECapBeamWt,
        "CapBeamMass": CapBeamMass,
        "CapBeamWt": CapBeamWt,
        "DiaPhragmLoad": DiaPhragmLoad,
        "AdditionalLoad": AdditionalLoad,
        "SSTWT": SSTWT,
    }
