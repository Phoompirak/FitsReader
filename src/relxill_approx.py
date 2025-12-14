"""
Simplified RELXILL-like Relativistic Reflection Models
สำหรับการจำลอง relativistic reflection จาก accretion disk รอบหลุมดำ

โมดูลนี้ประกอบด้วย:
1. Relativistic line profile (Laor-like)
2. Ionized reflection continuum  
3. Combined RELXILL-like model

หมายเหตุ: นี่เป็น approximation สำหรับการศึกษาเบื้องต้น
สำหรับการวิเคราะห์ที่แม่นยำควรใช้ XSPEC กับ relxill model จริง
"""

import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


def get_backend(array):
    """Return numpy or cupy module based on array type"""
    if HAS_GPU and cp is not None:
        return cp.get_array_module(array)
    return np


# ============================================================
# Physical Constants
# ============================================================
G = 6.674e-11       # Gravitational constant (m³/kg/s²)
c = 2.998e8         # Speed of light (m/s)
M_sun = 1.989e30    # Solar mass (kg)
keV_to_erg = 1.602e-9  # keV to erg conversion


# ============================================================
# Relativistic Line Profile (Laor-like approximation)
# ============================================================

def kerr_isco(spin):
    """
    Calculate Innermost Stable Circular Orbit (ISCO) for Kerr black hole
    
    Parameters:
    -----------
    spin : float
        Dimensionless spin parameter a/M (-1 to 1)
        a > 0: prograde rotation
        a < 0: retrograde rotation
        
    Returns:
    --------
    r_isco : float
        ISCO radius in units of GM/c²
    """
    xp = get_backend(spin) if hasattr(spin, 'shape') else np
    
    # Ensure spin is in valid range
    a = xp.clip(spin, -0.998, 0.998)
    
    # Calculate ISCO using Bardeen formula
    Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    Z2 = xp.sqrt(3 * a**2 + Z1**2)
    
    # ISCO for prograde/retrograde orbits
    r_isco = 3 + Z2 - xp.sign(a) * xp.sqrt((3 - Z1) * (3 + Z1 + 2*Z2))
    
    return r_isco


def gravitational_redshift(r, spin, incl):
    """
    Calculate gravitational redshift factor g = E_obs / E_emit
    
    Simplified version combining gravitational + Doppler effects
    
    Parameters:
    -----------
    r : array
        Radius in units of GM/c²
    spin : float
        Black hole spin (-1 to 1)
    incl : float
        Inclination angle (radians)
        
    Returns:
    --------
    g : array
        Redshift factor (0 to 1)
    """
    xp = get_backend(r)
    
    # Approximate orbital velocity at radius r (Keplerian)
    v_phi = xp.sqrt(1 / r)  # in units of c
    
    # Gravitational redshift
    g_grav = xp.sqrt(1 - 2/r)
    
    # Doppler factor (simplified for average over azimuth)
    # For disk, average Doppler shift cancels, but broadening remains
    beta = v_phi * xp.sin(incl)
    
    # Combined redshift (simplified)
    g = g_grav * (1 - beta**2)**0.5
    
    return xp.clip(g, 0.01, 1.0)


def relativistic_line_profile(energy, line_energy=6.4, spin=0.9, incl=30.0,
                              r_in=None, r_out=400.0, emissivity_index=3.0,
                              norm=1.0):
    """
    Relativistic emission line profile (Laor-like approximation)
    
    จำลอง line profile ที่ได้รับผลกระทบจาก:
    - Gravitational redshift
    - Doppler boosting/beaming
    - Light bending (approximated)
    
    Parameters:
    -----------
    energy : array
        Energy array (keV)
    line_energy : float
        Rest-frame line energy (keV), default 6.4 keV for Fe Kα
    spin : float
        Dimensionless spin parameter (-1 to 1)
    incl : float
        Inclination angle (degrees)
    r_in : float or None
        Inner disk radius (rg). If None, use ISCO
    r_out : float
        Outer disk radius (rg)
    emissivity_index : float
        Emissivity power-law index (ε ∝ r^-q)
    norm : float
        Normalization
        
    Returns:
    --------
    flux : array
        Line flux (photons/cm²/s/keV)
    """
    xp = get_backend(energy)
    
    # Convert inclination to radians
    incl_rad = xp.deg2rad(incl) if hasattr(xp, 'deg2rad') else np.deg2rad(incl)
    
    # Determine inner radius
    if r_in is None:
        r_in = float(kerr_isco(spin))
    
    # Create radial grid for integration
    n_r = 100
    r_grid = xp.linspace(r_in, r_out, n_r)
    
    # Initialize flux array
    if hasattr(energy, 'shape') and len(energy.shape) == 1:
        flux = xp.zeros_like(energy, dtype=float)
    else:
        flux = xp.zeros_like(energy, dtype=xp.float32)
    
    # Integrate over disk radii
    for i in range(len(r_grid) - 1):
        r = (r_grid[i] + r_grid[i+1]) / 2
        dr = r_grid[i+1] - r_grid[i]
        
        # Emissivity (power-law)
        emissivity = r ** (-emissivity_index)
        
        # Redshift factor
        g = gravitational_redshift(r, spin, incl_rad)
        
        # Observed energy
        E_obs = line_energy * g
        
        # Line width from velocity dispersion at this radius
        sigma = 0.05 * line_energy * (1 - g)  # Approximate broadening
        sigma = max(0.02, float(sigma))  # Minimum width
        
        # Add Gaussian contribution at this redshifted energy
        gaussian = xp.exp(-0.5 * ((energy - E_obs) / sigma)**2)
        gaussian /= (sigma * xp.sqrt(2 * np.pi))
        
        # Weight by emissivity and area element
        weight = emissivity * r * dr * g**3  # g³ for relativistic flux
        
        flux += weight * gaussian
    
    # Normalize
    flux = norm * flux / xp.max(flux) if xp.max(flux) > 0 else flux
    
    return flux


# ============================================================
# Ionized Reflection Continuum
# ============================================================

def ionized_reflection_continuum(energy, photon_index=2.0, xi=100.0, 
                                  refl_frac=1.0, Fe_abund=1.0, norm=1.0):
    """
    Simplified ionized reflection continuum
    
    Based on ionization parameter ξ = L / (n r²)
    
    Parameters:
    -----------
    energy : array
        Energy array (keV)
    photon_index : float
        Incident power-law photon index
    xi : float
        Ionization parameter (erg cm/s)
        ξ ~ 1: neutral reflection
        ξ ~ 100: moderately ionized
        ξ ~ 1000+: highly ionized
    refl_frac : float
        Reflection fraction (R)
    Fe_abund : float
        Iron abundance relative to solar
    norm : float
        Normalization
        
    Returns:
    --------
    flux : array
        Reflected flux (photons/cm²/s/keV)
    """
    xp = get_backend(energy)
    
    # Base power-law (incident continuum)
    incident = energy ** (-photon_index)
    
    # Compton hump parameters depend on ionization
    # Higher ionization → weaker hump
    hump_strength = 0.3 * xp.exp(-xp.log10(xi + 1) / 2)
    hump_energy = 25.0  # keV
    hump_width = 10.0   # keV
    
    compton_hump = hump_strength * xp.exp(-0.5 * ((energy - hump_energy) / hump_width)**2)
    
    # Soft excess from ionized emission
    # Higher ionization → more emission features
    log_xi = xp.log10(xi + 1)
    soft_strength = 0.1 * log_xi
    soft_emission = soft_strength * xp.exp(-energy / 1.0)  # Soft component < 2 keV
    
    # Absorption edge modifications
    # Lower ionization → stronger neutral edges
    edge_7keV = xp.where(energy > 7.1, 
                         xp.exp(-(1 - xp.tanh(log_xi - 2)) * 0.5 * (energy - 7.1)),
                         1.0)
    
    # Combine components
    reflection = refl_frac * (incident * edge_7keV + compton_hump + soft_emission)
    
    # Iron abundance affects Fe K edge and line region
    fe_region = (energy > 6.0) & (energy < 8.0)
    reflection = xp.where(fe_region, reflection * Fe_abund, reflection)
    
    return norm * reflection


# ============================================================
# Combined RELXILL-like Model
# ============================================================

def relxill_approx(energy, params):
    """
    Simplified RELXILL-like model combining:
    - Relativistic blurred reflection
    - Ionized disk emission
    - Power-law continuum
    
    Parameters:
    -----------
    energy : array
        Energy array (keV)
    params : dict
        Model parameters:
        - pl_norm: Power-law normalization
        - photon_index / Gamma: Photon index
        - spin: Black hole spin (a)
        - incl: Inclination (degrees)
        - r_in: Inner radius (rg, optional)
        - r_out: Outer radius (rg)
        - xi: Ionization parameter
        - refl_frac: Reflection fraction
        - Fe_abund: Iron abundance
        - line_norm: Fe Kα line normalization
        
    Returns:
    --------
    total_flux : array
        Total model flux
    """
    xp = get_backend(energy)
    
    # Extract parameters with defaults
    pl_norm = params.get('pl_norm', 1.0)
    Gamma = params.get('photon_index', params.get('Gamma', 2.0))
    spin = params.get('spin', 0.9)
    incl = params.get('incl', 30.0)
    r_in = params.get('r_in', None)
    r_out = params.get('r_out', 400.0)
    xi = params.get('xi', 100.0)
    refl_frac = params.get('refl_frac', 1.0)
    Fe_abund = params.get('Fe_abund', 1.0)
    line_norm = params.get('line_norm', 1.0)
    
    # 1. Power-law continuum
    powerlaw = pl_norm * energy ** (-Gamma)
    
    # 2. Ionized reflection continuum
    reflection = ionized_reflection_continuum(
        energy, 
        photon_index=Gamma,
        xi=xi,
        refl_frac=refl_frac,
        Fe_abund=Fe_abund,
        norm=pl_norm * 0.1  # Scale with power-law
    )
    
    # 3. Relativistic Fe Kα line
    # Line energy depends on ionization
    if xi < 10:
        line_E = 6.4   # Neutral Fe Kα
    elif xi < 500:
        line_E = 6.7   # He-like Fe XXV
    else:
        line_E = 6.97  # H-like Fe XXVI
    
    fe_line = relativistic_line_profile(
        energy,
        line_energy=line_E,
        spin=spin,
        incl=incl,
        r_in=r_in,
        r_out=r_out,
        emissivity_index=3.0,
        norm=line_norm * Fe_abund
    )
    
    # 4. Relativistic blurring of reflection (simplified)
    # Apply redshift smearing to reflection
    g_avg = 0.8 + 0.1 * spin  # Average redshift factor
    reflection_blurred = reflection * g_avg
    
    # Total model
    total_flux = powerlaw + reflection_blurred + fe_line
    
    return total_flux


def relxillCp_approx(energy, params):
    """
    Simplified RELXILLCP-like model
    Uses nthcomp-like continuum instead of cutoff power-law
    
    Additional parameters:
    - kT_e: Electron temperature (keV)
    - kT_bb: Seed photon temperature (keV)
    """
    xp = get_backend(energy)
    
    # Extract comptonization parameters
    kT_e = params.get('kT_e', 100.0)  # Electron temperature
    kT_bb = params.get('kT_bb', 0.1)  # Seed photon temperature
    Gamma = params.get('photon_index', params.get('Gamma', 2.0))
    pl_norm = params.get('pl_norm', 1.0)
    
    # Comptonized continuum (simplified nthcomp-like)
    # Low-energy rollover from seed photons
    low_E_factor = 1 - xp.exp(-energy / (3 * kT_bb))
    
    # High-energy cutoff from electron temperature
    high_E_factor = xp.exp(-energy / kT_e)
    
    # Power-law shape
    powerlaw_shape = energy ** (-Gamma)
    
    # Combined comptonized spectrum
    comptonized = pl_norm * powerlaw_shape * low_E_factor * high_E_factor
    
    # Replace power-law with comptonized in relxill
    params_modified = params.copy()
    params_modified['pl_norm'] = 0  # Remove simple power-law
    
    # Get reflection + line components
    reflection_components = relxill_approx(energy, params_modified)
    
    # Add comptonized continuum
    total_flux = comptonized + reflection_components
    
    return total_flux


# ============================================================
# Utility Functions
# ============================================================

def get_relxill_parameter_info():
    """
    Return parameter descriptions for RELXILL-like models
    """
    return {
        'spin': {
            'name': 'Black Hole Spin (a)',
            'description': 'Dimensionless spin parameter',
            'range': (-0.998, 0.998),
            'typical': 0.9,
            'physics': 'a > 0.9: rapidly spinning, ISCO very close to horizon'
        },
        'incl': {
            'name': 'Inclination',
            'description': 'Viewing angle (degrees)',
            'range': (3, 85),
            'typical': 30,
            'physics': '< 30°: face-on, > 60°: edge-on'
        },
        'xi': {
            'name': 'Ionization Parameter (ξ)',
            'description': 'Ionization state of the disk (erg cm/s)',
            'range': (1, 10000),
            'typical': 100,
            'physics': 'log ξ ~ 1-2: neutral, log ξ > 3: highly ionized'
        },
        'refl_frac': {
            'name': 'Reflection Fraction (R)',
            'description': 'Ratio of reflected to direct emission',
            'range': (0, 10),
            'typical': 1.0,
            'physics': 'R > 1: strong light bending, R ~ 0: corona above disk'
        },
        'Fe_abund': {
            'name': 'Iron Abundance',
            'description': 'Fe abundance relative to solar',
            'range': (0.5, 10),
            'typical': 1.0,
            'physics': 'AGN often show Fe > 1-3 solar'
        },
        'r_in': {
            'name': 'Inner Radius',
            'description': 'Inner disk radius (rg = GM/c²)',
            'range': (1, 100),
            'typical': 'ISCO',
            'physics': 'r_in = ISCO for standard disk'
        },
        'emissivity_index': {
            'name': 'Emissivity Index (q)',
            'description': 'Radial emissivity profile (ε ∝ r^-q)',
            'range': (0, 10),
            'typical': 3.0,
            'physics': 'q = 3: standard disk, q > 6: lamp-post geometry'
        }
    }


def interpret_relxill_results(params, chi2_dof):
    """
    Provide physical interpretation of RELXILL fit results
    
    Parameters:
    -----------
    params : dict
        Best-fit parameters
    chi2_dof : float
        Reduced chi-squared
        
    Returns:
    --------
    interpretation : str
        Physical interpretation text
    """
    lines = []
    
    # Spin interpretation
    spin = params.get('spin', 0)
    if abs(spin) > 0.9:
        lines.append(f"🌀 **High Spin (a = {spin:.2f})**: SMBH กำลังหมุนเร็วมาก ISCO อยู่ใกล้ขอบฟ้าเหตุการณ์")
    elif abs(spin) > 0.5:
        lines.append(f"🌀 **Moderate Spin (a = {spin:.2f})**: SMBH กำลังหมุนปานกลาง")
    else:
        lines.append(f"🌀 **Low Spin (a = {spin:.2f})**: SMBH หมุนช้า หรือ retrograde")
    
    # Ionization interpretation
    xi = params.get('xi', 1)
    log_xi = np.log10(xi + 1)
    if log_xi < 1.5:
        lines.append(f"⚡ **Low Ionization (ξ = {xi:.0f})**: Disk เย็น, Fe Kα ที่ 6.4 keV")
    elif log_xi < 3:
        lines.append(f"⚡ **Moderate Ionization (ξ = {xi:.0f})**: Disk อุ่น, Fe XXV ที่ 6.7 keV")
    else:
        lines.append(f"⚡ **High Ionization (ξ = {xi:.0f})**: Disk ร้อน, Fe XXVI ที่ 7.0 keV")
    
    # Reflection fraction interpretation
    R = params.get('refl_frac', 0)
    if R < 0.3:
        lines.append(f"🪞 **Weak Reflection (R = {R:.2f})**: Corona อยู่สูงเหนือ disk")
    elif R < 1.5:
        lines.append(f"🪞 **Standard Reflection (R = {R:.2f})**: Geometry ปกติ")
    else:
        lines.append(f"🪞 **Strong Reflection (R = {R:.2f})**: Light bending รุนแรง หรือ corona compact")
    
    # Inclination interpretation
    incl = params.get('incl', 30)
    if incl < 30:
        lines.append(f"👁️ **Face-on View (i = {incl:.0f}°)**: มองตรงลง disk")
    elif incl > 60:
        lines.append(f"👁️ **Edge-on View (i = {incl:.0f}°)**: มอง disk จากด้านข้าง")
    else:
        lines.append(f"👁️ **Intermediate View (i = {incl:.0f}°)**: มุมมองปานกลาง")
    
    # Fit quality
    if chi2_dof < 1.2:
        lines.append(f"\n✅ **Good Fit** (χ²/dof = {chi2_dof:.2f})")
    elif chi2_dof < 2.0:
        lines.append(f"\n⚠️ **Acceptable Fit** (χ²/dof = {chi2_dof:.2f})")
    else:
        lines.append(f"\n❌ **Poor Fit** (χ²/dof = {chi2_dof:.2f}) - ควรพิจารณา model อื่น")
    
    return "\n".join(lines)
