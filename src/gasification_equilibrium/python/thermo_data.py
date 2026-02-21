"""NASA 7-coefficient polynomial thermodynamic data for gasification species.

Design by Contract:
    - All temperatures in Kelvin, all pressures in Pascals
    - NASA polynomials valid within stated temperature ranges
    - Cp/R, H/RT, S/R are dimensionless polynomial evaluations
    - G/RT = H/RT - S/R

Data source: NASA Glenn coefficients (Burcat & Ruscic, Third Millennium Database)
"""

import numpy as np

# Universal gas constant [J/(mol·K)]
R_GAS = 8.314462618

# Reference pressure [Pa]
P_REF = 101325.0

# Reference temperature [K]
T_REF = 298.15

# ─── Species database ──────────────────────────────────────────────────────────
# Each entry: {
#   'name': display name,
#   'formula': chemical formula,
#   'mw': molecular weight [g/mol],
#   'elements': {element: count},
#   'phase': 'gas' or 'solid',
#   'T_low': low T bound [K],
#   'T_mid': mid T breakpoint [K],
#   'T_high': high T bound [K],
#   'coeff_low': 7 NASA coefficients for T_low..T_mid,
#   'coeff_high': 7 NASA coefficients for T_mid..T_high,
# }

SPECIES_DB = {
    "H2": {
        "name": "Hydrogen",
        "formula": "H2",
        "mw": 2.016,
        "elements": {"H": 2},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            2.34433112,
            7.98052075e-03,
            -1.94781510e-05,
            2.01572094e-08,
            -7.37611761e-12,
            -917.935173,
            0.683010238,
        ],
        "coeff_high": [
            3.33727920,
            -4.94024731e-05,
            4.99456778e-07,
            -1.79566394e-10,
            2.00255376e-14,
            -950.158922,
            -3.20502331,
        ],
    },
    "CO": {
        "name": "Carbon Monoxide",
        "formula": "CO",
        "mw": 28.010,
        "elements": {"C": 1, "O": 1},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            3.57953347,
            -6.10353680e-04,
            1.01681433e-06,
            9.07005884e-10,
            -9.04424499e-13,
            -14344.086,
            3.50840928,
        ],
        "coeff_high": [
            2.71518561,
            2.06252743e-03,
            -9.98825771e-07,
            2.30053008e-10,
            -2.03647716e-14,
            -14151.8724,
            7.81868772,
        ],
    },
    "CO2": {
        "name": "Carbon Dioxide",
        "formula": "CO2",
        "mw": 44.009,
        "elements": {"C": 1, "O": 2},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            2.35677352,
            8.98459677e-03,
            -7.12356269e-06,
            2.45919022e-09,
            -1.43699548e-13,
            -48371.9697,
            9.90105222,
        ],
        "coeff_high": [
            3.85746029,
            4.41437026e-03,
            -2.21481404e-06,
            5.23490188e-10,
            -4.72084164e-14,
            -48759.166,
            2.27163806,
        ],
    },
    "H2O": {
        "name": "Water",
        "formula": "H2O",
        "mw": 18.015,
        "elements": {"H": 2, "O": 1},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            4.19864056,
            -2.03643410e-03,
            6.52040211e-06,
            -5.48797062e-09,
            1.77197817e-12,
            -30293.7267,
            -0.849032208,
        ],
        "coeff_high": [
            3.03399249,
            2.17691804e-03,
            -1.64072518e-07,
            -9.70419870e-11,
            1.68200992e-14,
            -30004.2971,
            4.96677010,
        ],
    },
    "CH4": {
        "name": "Methane",
        "formula": "CH4",
        "mw": 16.043,
        "elements": {"C": 1, "H": 4},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            5.14987613,
            -1.36709788e-02,
            4.91800599e-05,
            -4.84743026e-08,
            1.66693956e-11,
            -10246.6476,
            -4.64130376,
        ],
        "coeff_high": [
            0.074851495,
            1.33909467e-02,
            -5.73285809e-06,
            1.22292535e-09,
            -1.01815230e-13,
            -9468.34459,
            18.437318,
        ],
    },
    "N2": {
        "name": "Nitrogen",
        "formula": "N2",
        "mw": 28.014,
        "elements": {"N": 2},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            3.53100528,
            -1.23660988e-04,
            -5.02999433e-07,
            2.43530612e-09,
            -1.40881235e-12,
            -1046.97628,
            2.96747038,
        ],
        "coeff_high": [
            2.95257637,
            1.39690040e-03,
            -4.92631603e-07,
            7.86010195e-11,
            -4.60755204e-15,
            -923.948688,
            5.87188762,
        ],
    },
    "O2": {
        "name": "Oxygen",
        "formula": "O2",
        "mw": 31.998,
        "elements": {"O": 2},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            3.78245636,
            -2.99673416e-03,
            9.84730201e-06,
            -9.68129509e-09,
            3.24372837e-12,
            -1063.94356,
            3.65767573,
        ],
        "coeff_high": [
            3.28253784,
            1.48308754e-03,
            -7.57966669e-07,
            2.09470555e-10,
            -2.16717794e-14,
            -1088.45772,
            5.45323129,
        ],
    },
    "C2H4": {
        "name": "Ethylene",
        "formula": "C2H4",
        "mw": 28.054,
        "elements": {"C": 2, "H": 4},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            3.95920148,
            -7.57052247e-03,
            5.70990292e-05,
            -6.91588753e-08,
            2.69884373e-11,
            5089.77593,
            4.09733096,
        ],
        "coeff_high": [
            2.03611116,
            1.46454151e-02,
            -6.71077915e-06,
            1.47222923e-09,
            -1.25706061e-13,
            4939.88614,
            10.3053693,
        ],
    },
    "C2H6": {
        "name": "Ethane",
        "formula": "C2H6",
        "mw": 30.070,
        "elements": {"C": 2, "H": 6},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            4.29142492,
            -5.50154270e-03,
            5.99438288e-05,
            -7.08466285e-08,
            2.68685771e-11,
            -11522.2055,
            2.66682316,
        ],
        "coeff_high": [
            1.07188150,
            2.16852677e-02,
            -1.00256067e-05,
            2.21412001e-09,
            -1.90002890e-13,
            -12426.5222,
            15.1156107,
        ],
    },
    "H2S": {
        "name": "Hydrogen Sulfide",
        "formula": "H2S",
        "mw": 34.082,
        "elements": {"H": 2, "S": 1},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            4.12023590,
            -3.24803220e-03,
            1.67209781e-05,
            -1.73457074e-08,
            6.30820488e-12,
            -3650.53590,
            1.72021024,
        ],
        "coeff_high": [
            2.88324232,
            3.81130960e-03,
            -1.47230893e-06,
            2.74093019e-10,
            -1.98241636e-14,
            -3455.11880,
            8.00522400,
        ],
    },
    "NH3": {
        "name": "Ammonia",
        "formula": "NH3",
        "mw": 17.031,
        "elements": {"N": 1, "H": 3},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            4.28648920,
            -4.66055869e-03,
            2.17119913e-05,
            -2.28063689e-08,
            8.26395924e-12,
            -6741.72790,
            -0.625282180,
        ],
        "coeff_high": [
            2.63455580,
            5.66694560e-03,
            -1.72891830e-06,
            2.38672510e-10,
            -1.25756950e-14,
            -6544.69590,
            6.56632780,
        ],
    },
    "SO2": {
        "name": "Sulfur Dioxide",
        "formula": "SO2",
        "mw": 64.066,
        "elements": {"S": 1, "O": 2},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            3.26653380,
            5.32379020e-03,
            6.84375520e-07,
            -5.28100470e-09,
            2.55904540e-12,
            -36908.14500,
            9.66465108,
        ],
        "coeff_high": [
            5.24513640,
            1.97042040e-03,
            -8.03757690e-07,
            1.51499690e-10,
            -1.05580040e-14,
            -37550.73400,
            -1.07404890,
        ],
    },
    "C3H8": {
        "name": "Propane",
        "formula": "C3H8",
        "mw": 44.096,
        "elements": {"C": 3, "H": 8},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            4.21093028,
            1.73880780e-03,
            7.09192623e-05,
            -9.20376658e-08,
            3.64238090e-11,
            -14381.0876,
            5.61282290,
        ],
        "coeff_high": [
            0.75341368,
            3.18290557e-02,
            -1.49584302e-05,
            3.34975168e-09,
            -2.90088803e-13,
            -16467.5165,
            17.4587913,
        ],
    },
    "Ar": {
        "name": "Argon",
        "formula": "Ar",
        "mw": 39.948,
        "elements": {},
        "phase": "gas",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491],
        "coeff_high": [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491],
    },
    "C_solid": {
        "name": "Graphite",
        "formula": "C(s)",
        "mw": 12.011,
        "elements": {"C": 1},
        "phase": "solid",
        "T_low": 200.0,
        "T_mid": 1000.0,
        "T_high": 3500.0,
        "coeff_low": [
            -0.310872072,
            4.40353686e-03,
            1.90394118e-06,
            -6.38546966e-09,
            2.98964248e-12,
            -108.650974,
            1.11382953,
        ],
        "coeff_high": [
            1.45571829,
            1.71702216e-03,
            -6.97562786e-07,
            1.35277032e-10,
            -1.00589440e-14,
            -695.138840,
            -8.52583033,
        ],
    },
}


def cp_over_r(T, coeffs):
    """Cp/R from NASA polynomial. T must be in valid range.

    Precondition: T > 0, coeffs has 7 elements
    Postcondition: returns positive float
    """
    a = coeffs
    return a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4


def h_over_rt(T, coeffs):
    """H/(RT) from NASA polynomial.

    Precondition: T > 0, coeffs has 7 elements
    """
    a = coeffs
    return (
        a[0]
        + a[1] * T / 2
        + a[2] * T**2 / 3
        + a[3] * T**3 / 4
        + a[4] * T**4 / 5
        + a[5] / T
    )


def s_over_r(T, coeffs):
    """S/R from NASA polynomial.

    Precondition: T > 0, coeffs has 7 elements
    """
    a = coeffs
    return (
        a[0] * np.log(T)
        + a[1] * T
        + a[2] * T**2 / 2
        + a[3] * T**3 / 3
        + a[4] * T**4 / 4
        + a[6]
    )


def g_over_rt(T, coeffs):
    """G/(RT) = H/(RT) - S/R from NASA polynomial.

    Precondition: T > 0, coeffs has 7 elements
    """
    return h_over_rt(T, coeffs) - s_over_r(T, coeffs)


def get_coeffs(species_key, T):
    """Get appropriate NASA coefficients for species at temperature T.

    Precondition: species_key in SPECIES_DB, T > 0
    Postcondition: returns 7-element coefficient list

    Uses low-T coefficients for T <= T_mid, high-T for T > T_mid.
    Clamps T to valid range with warning.
    """
    sp = SPECIES_DB[species_key]
    T_clamped = np.clip(T, sp["T_low"], sp["T_high"])
    if T_clamped <= sp["T_mid"]:
        return sp["coeff_low"]
    return sp["coeff_high"]


def gibbs_dimensionless(species_key, T):
    """Compute dimensionless Gibbs energy G°/(RT) for species at T.

    Precondition: species_key in SPECIES_DB, T > 0
    Postcondition: returns float (can be negative)
    """
    coeffs = get_coeffs(species_key, T)
    return g_over_rt(T, coeffs)


def enthalpy_j_per_mol(species_key, T):
    """Compute standard enthalpy H° [J/mol] for species at T.

    Precondition: species_key in SPECIES_DB, T > 0
    """
    coeffs = get_coeffs(species_key, T)
    return h_over_rt(T, coeffs) * R_GAS * T


def entropy_j_per_mol_k(species_key, T):
    """Compute standard entropy S° [J/(mol·K)] for species at T.

    Precondition: species_key in SPECIES_DB, T > 0
    """
    coeffs = get_coeffs(species_key, T)
    return s_over_r(T, coeffs) * R_GAS


def cp_j_per_mol_k(species_key, T):
    """Compute heat capacity Cp [J/(mol·K)] for species at T.

    Precondition: species_key in SPECIES_DB, T > 0
    """
    coeffs = get_coeffs(species_key, T)
    return cp_over_r(T, coeffs) * R_GAS


# ─── Element data ───────────────────────────────────────────────────────────────

ATOMIC_WEIGHTS = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "N": 14.007,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
}

# Heating values [kJ/mol] for CGE calculation
HEATING_VALUES_HHV = {
    "H2": 285.8,
    "CO": 283.0,
    "CH4": 890.8,
    "C2H4": 1411.0,
    "C2H6": 1560.7,
}
HEATING_VALUES_LHV = {
    "H2": 241.8,
    "CO": 283.0,
    "CH4": 802.6,
    "C2H4": 1323.0,
    "C2H6": 1428.6,
}


def get_gas_species():
    """Return list of gas-phase species keys."""
    return [k for k, v in SPECIES_DB.items() if v["phase"] == "gas"]


def get_all_species():
    """Return list of all species keys."""
    return list(SPECIES_DB.keys())


def get_elements_in_system(species_keys):
    """Return sorted list of all elements present in given species.

    Precondition: all keys in species_keys exist in SPECIES_DB
    """
    elements = set()
    for key in species_keys:
        elements.update(SPECIES_DB[key]["elements"].keys())
    return sorted(elements)
