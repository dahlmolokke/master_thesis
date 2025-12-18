import numpy as np
from scipy import constants as con
import iri2016 as iri
import pymsis as msis
import ppigrf as igrf


def nu_en(nO, nO2, nN2, Te, epsilon=1e-8):
    """
    Function for calculating the total electron neutral
    collision frequency
    Based on Schunk & Nagy 2009
    Input:
        nO: O density in cm^-3
        nO2: O2 density in cm^-3
        nN2: N2 density in cm^-3
        Te: electron temperature in K
        epsilon: small value to avoid sqrt(0)
    Output:
        nu: total electron-neutral collision frequency in Hz
    """
    Te[Te < 0] = 0
    # Electron - O collision
    e_O = 8.9e-11 * nO * (1 + 5.7e-4 * Te) * np.sqrt(Te + epsilon)
    # Electron - O2 collision
    e_O2 = 1.82e-10 * nO2 * (1 + 3.6e-2 * np.sqrt(Te + epsilon)) * np.sqrt(Te + epsilon)
    # Electron - N2 collision
    e_N2 = 2.33e-11 * nN2 * (1 - 1.21e-4 * Te) * Te
    
    return e_O2 + e_O + e_N2

def nu_in(nO, nO2, nN2, Tr, epsilon=1e-8):
    """
    Function for calculating the ion neutral
    collision frequencies for each of the chosen ion species
    Based on Schunk & Nagy 2009
    Input:
        nO: O density in cm^-3
        nO2: O2 density in cm^-3
        nN2: N2 density in cm^-3
        Tr: mean temperature in K for ion and neutral species: (Ti + Tn)/2
        epsilon: small value to avoid log(0) or sqrt(0)
    Output:
        Op_terms: O+ - neutral collision frequency in Hz
        O2p_terms: O2+ - neutral collision frequency in Hz
        N2p_terms: N2+ - neutral collision frequency in Hz
        NOp_terms: NO+ - neutral collision frequency in Hz
    """
    Tr[Tr < 0] = 0

    # Op collisions
    Op_O = np.where(
        Tr > 235,
        3.67e-11 * nO * np.sqrt(Tr + epsilon) * (1 - 0.064 * np.log10(Tr + epsilon))**2,
        0.0)
    Op_O2 = 6.64e-10 * nO2
    Op_N2 = 6.82e-10 * nN2

    Op_terms = Op_N2 + Op_O + Op_O2

    # O2p collisions
    O2p_O = 2.31e-10 * nO
    O2p_O2 = np.where(
        Tr > 800, 
        2.59e-11 * nO2 * np.sqrt(Tr + epsilon) * (1 - 0.073 * np.log10(Tr + epsilon))**2,
        0.0)
    O2p_N2 = 4.13e-10 * nN2

    O2p_terms = O2p_N2 + O2p_O + O2p_O2
    
    # N2p collisions
    N2p_O = 2.58e-10 * nO
    N2p_O2 = 4.49e-10 * nO2
    N2p_N2 = np.where(
        Tr > 170,
        5.14e-11 * nN2 * np.sqrt(Tr + epsilon) * (1 - 0.069 * np.log10(Tr +  epsilon))**2,
        0.0)

    N2p_terms = N2p_N2 + N2p_O + N2p_O2

    # NOp collisions
    NOp_O = 2.44e-10 * nO
    NOp_O2 = 4.27e-10 * nO2
    NOp_N2 = 4.34e-10 * nN2

    NOp_terms = NOp_O + NOp_O2 + NOp_N2

    return Op_terms, O2p_terms, N2p_terms, NOp_terms


def gyrofreq(B, mass, hertz=True):
    """
    Function for calculating gyrofrequency
    Input:
        B: Magnetic field strength in Tesla
        mass: mass of the charged particle in kg
        hertz: If True, return frequency in Hz. If False, return angular frequency in rad/s
    Output:
        f: gyrofrequency in Hz or rad/s
    """
    omega = B * con.e / mass 
    if not hertz:
        return omega
    return omega / (2 * np.pi)

def ion_ratio(numerators):
    """
    Function for calculating ion density ratios
    Input:
        numerators: list of ion densities
    Output:
        ratios: list of ion density ratios
    """
    ratios = []
    denominator = sum(numerators)
    for numerator in numerators:
        ratios.append(numerator/denominator)
    return ratios

def conductivity_term(density, coll_freq, mag_field, mass):
    """
    Function for calculating individual terms of Pedersen conductivity
    Input:
        density: density of charged species in m^-3
        coll_freq: collision frequency of charged species in Hz
        mag_field: magnetic field strength in Tesla
        mass: mass of charged species in kg 
    """
    term = ((density * con.e**2) / mass) * (coll_freq / (coll_freq**2 + gyrofreq(mag_field, mass, hertz=False)**2))
    return term

def _calculate_environment_params(date, altitude, lat, lon, no_calc=False):
    """
    Performs the expensive, N_e-independent calculations for a given timestamp and location.
    This function should only be called once per unique date.
    Input:
        date: datetime object
        altitude: altitude array in km
        lat: latitude in degrees
        lon: longitude in degrees
        no_calc: If True, skip collision frequency calculations and return only densities and temperatures
    Output:
        params: dictionary containing calculated parameters
    """
    # Altitude vector for interpolation
    eq_alt = np.linspace(0, 600, 601)  # [km]

    # IGRF Magnetic field
    Be, Bn, Bu = igrf.igrf(lon, lat, altitude, date)  # [nT]
    B = np.sqrt(Be**2 + Bn**2 + Bu**2).squeeze() * 1e-9  # [T]

    # PyMSIS Atmosphere composition
    msis_data = msis.calculate(date, lon, lat, altitude)
    nN2 = msis_data[0, 0, 0, :, 1]
    nO2 = msis_data[0, 0, 0, :, 2]
    nO = msis_data[0, 0, 0, :, 3]
    nO_cm, nO2_cm, nN2_cm = nO * 1e-6, nO2 * 1e-6, nN2 * 1e-6

    # IRI Temperatures and Ion Densities
    iri_data = iri.IRI(date, (0, 600, 1), lon, lat)
    Te = np.interp(altitude, eq_alt, iri_data["Te"], -1)
    Ti = np.interp(altitude, eq_alt, iri_data["Ti"], -1)
    Tn = np.interp(altitude, eq_alt, iri_data["Tn"], -1)
    ne = np.interp(altitude, eq_alt, iri_data["ne"], -1)
    Tr = (Tn + Ti) / 2
    
    nOp = np.interp(altitude, eq_alt, iri_data["nO+"])   # [m^-3]
    nO2p = np.interp(altitude, eq_alt, iri_data["nO2+"]) # [m^-3]
    nNOp = np.interp(altitude, eq_alt, iri_data["nNO+"]) # [m^-3]
    nN2p = np.zeros_like(nN2)                            # [m^-3]]

    if no_calc:
        return { 
            "B": B,
            "ne": ne, 
            "nO": nO, 
            "nO2": nO2, 
            "nN2": nN2,
            "Te": Te, 
            "Ti": Ti, 
            "Tn": Tn,
            "nOp": nOp,
            "nO2p": nO2p,
            "nNOp": nNOp
            }
    
    # Calculate ion ratios once
    ion_ratios_val = ion_ratio([nOp, nO2p, nN2p, nNOp])

    # Collision frequencies
    nu_e = nu_en(nO_cm, nO2_cm, nN2_cm, Te)
    nu_Opn, nu_O2pn, nu_N2pn, nu_NOpn = nu_in(nO_cm, nO2_cm, nN2_cm, Tr)

    # Return all calculated parameters in a dictionary for easy access
    return {
        "B": B,
        "nu_e": nu_e,
        "nu_Opn": nu_Opn,
        "nu_O2pn": nu_O2pn,
        "nu_N2pn": nu_N2pn,
        "nu_NOpn": nu_NOpn,
        "ion_ratios": ion_ratios_val
    }

def conductivity_P_batch(ne_profiles, profile_datetimes, altitude, lat, lon, simplify=False):
    """
    Efficiently calculates Pedersen conductivity for multiple electron density profiles.

    Args:
        ne_profiles (list or np.ndarray): A list of 1D arrays, where each array is an ne profile.
                                          Or a 2D array of shape (num_profiles, num_altitudes).
        profile_datetimes (list): A list of datetime objectscorresponding to each ne_profile.
        altitude (np.ndarray): The 1D array of altitude values.
        lat (float): Latitude.
        lon (float): Longitude.
        simplify (bool): If True, ignores electron contribution.

    Returns:
        np.ndarray: A 2D array where each row is the calculated conductivity profile.
    """
    # Ion masses
    mO = 16.00 * con.u
    mO2 = 2 * mO
    mN2 = 2 * 14.00 * con.u
    mNO = (14.00 + 16.00) * con.u

    env_cache = {}  # Cache to store results of expensive calculations
    all_conductivities = []

    for ne, dt_object in zip(ne_profiles, profile_datetimes):
        if dt_object not in env_cache:
            # Calculate and store environment parameters
            env_cache[dt_object] = _calculate_environment_params(dt_object, altitude, lat, lon)
        # Retrieve cached parameters
        params = env_cache[dt_object]
        B = params["B"]
        ion_ratios = params["ion_ratios"]

        # Ion conductivity terms
        cond_i_terms = conductivity_term(ne * ion_ratios[0], params["nu_Opn"], B, mO) \
                     + conductivity_term(ne * ion_ratios[1], params["nu_O2pn"], B, mO2) \
                     + conductivity_term(ne * ion_ratios[2], params["nu_N2pn"], B, mN2) \
                     + conductivity_term(ne * ion_ratios[3], params["nu_NOpn"], B, mNO)
        
        terms = cond_i_terms

        if not simplify:
            # Include electron conductivity term
            cond_e_terms = conductivity_term(ne, params["nu_e"], B, con.m_e)
            terms += cond_e_terms

        # Append the total conductivity profile
        all_conductivities.append(terms)

    return np.array(all_conductivities)