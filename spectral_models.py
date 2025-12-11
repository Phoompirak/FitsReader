"""
Spectral Models for X-ray Spectrum Analysis
สำหรับการฟิตสเปกตรัมรังสีเอกซ์ของ AGN

โมดูลนี้ประกอบด้วย spectral models ทางฟิสิกส์ต่างๆ ที่ใช้ในการวิเคราะห์สเปกตรัมรังสีเอกซ์
"""

import numpy as np

# ค่าคงที่ทางฟิสิกส์
h_planck = 6.62607015e-34  # J·s (Planck constant)
c_light = 2.99792458e8     # m/s (Speed of light)
k_boltzmann = 1.380649e-23 # J/K (Boltzmann constant)
keV_to_J = 1.60218e-16     # Conversion factor from keV to Joules


def powerlaw(energy, norm, photon_index):
    """
    Power-law model สำหรับ X-ray continuum
    
    องค์ประกอบทางกายภาพ:
    - เป็นโมเดลพื้นฐานที่ใช้อธิบาย continuum emission จาก AGN
    - เกิดจาก Comptonization ของโฟตอนใน region ใกล้หลุมดำ (corona)
    - Photon index (Γ) บอกความชัน: Γ ~ 1.7-2.0 สำหรับ AGN ทั่วไป
    
    Parameters:
    -----------
    energy : array-like
        พลังงานของโฟตอน (keV)
    norm : float
        Normalization factor (photons/cm²/s/keV ที่ 1 keV)
    photon_index : float
        Photon index (Γ), ความชันของ power-law
        
    Returns:
    --------
    flux : array-like
        Photon flux (photons/cm²/s/keV)
        
    Model: F(E) = norm × E^(-Γ)
    """
    return norm * np.power(energy, -photon_index)


def blackbody(energy, norm, kT):
    """
    Blackbody (Planck) spectrum
    
    องค์ประกอบทางกายภาพ:
    - อธิบาย thermal emission จาก accretion disk
    - kT คืออุณหภูมิในหน่วย keV (1 keV ≈ 1.16 × 10^7 K)
    - สำหรับ AGN มักพบใน soft X-ray band (< 2 keV)
    
    Parameters:
    -----------
    energy : array-like
        พลังงานของโฟตอน (keV)
    norm : float
        Normalization factor
    kT : float
        อุณหภูมิในหน่วย keV
        
    Returns:
    --------
    flux : array-like
        Photon flux (photons/cm²/s/keV)
        
    Model: Blackbody spectrum (simplified for X-ray)
    """
    # Blackbody photon spectrum
    x = energy / kT
    # Avoid overflow for large x
    x = np.clip(x, 0, 100)
    spectrum = norm * (energy**2) / (np.exp(x) - 1.0 + 1e-10)
    return np.where(x < 100, spectrum, 0.0)


def tbabs(energy, nH):
    """
    Photoelectric absorption (simplified Tübingen-Boulder model)
    
    องค์ประกอบทางกายภาพ:
    - การดูดกลืนรังสีเอกซ์โดย neutral hydrogen ในทางเดินของแสง
    - nH คือ column density ของ hydrogen (10^22 atoms/cm²)
    - ส่งผลมากในช่วง soft X-ray (< 2 keV)
    
    Parameters:
    -----------
    energy : array-like
        พลังงานของโฟตอน (keV)
    nH : float
        Hydrogen column density ในหน่วย 10^22 atoms/cm²
        
    Returns:
    --------
    transmission : array-like
        Transmission factor (0-1)
        
    Model: T(E) = exp(-σ(E) × nH)
    โดยที่ σ(E) คือ photoelectric cross-section
    
    Note: ใช้ fit approximation จาก Morrison & McCammon (1983)
    แก้ไขให้ cross-section เป็นบวกเสมอและมีค่าเหมาะสมในช่วง X-ray
    """
    E = np.asarray(energy, dtype=float)
    
    # Improved photoelectric cross-section approximation
    # Based on Morrison & McCammon (1983) ApJ 270, 119
    # Valid for 0.03-10 keV
    
    # Ensure minimum energy to avoid division issues
    E = np.maximum(E, 0.03)
    
    # Cross-section in units of 10^-24 cm^2 per H atom
    # Using piecewise approximation for better accuracy
    
    # For E < 0.1 keV (soft X-ray)
    # σ ~ constant × E^-3 with positive coefficients
    sigma_soft = 180.0 * np.power(E, -3.0)
    
    # For 0.1 < E < 2 keV (medium X-ray)
    # More accurate fit with reduced cross-section
    sigma_mid = (17.3 + 608.1/E) * np.power(E, -3.0)
    
    # For E > 2 keV (hard X-ray)
    # Simple power-law decline
    sigma_hard = 600.0 * np.power(E, -3.0)
    
    # Combine piecewise
    sigma = np.where(E < 0.1, sigma_soft,
                    np.where(E < 2.0, sigma_mid, sigma_hard))
    
    # Ensure sigma is always positive
    sigma = np.maximum(sigma, 0.0)
    
    # Optical depth τ = σ × nH
    # nH in units of 10^22 cm^-2, sigma in 10^-24 cm^2
    # τ = σ[10^-24 cm^2] × nH[10^22 cm^-2] × (10^-24 × 10^22) = σ × nH × 10^-2
    tau = sigma * nH * 1e-2
    
    # Transmission = exp(-τ)
    # Clip tau to avoid numerical issues
    tau = np.minimum(tau, 100.0)  # Avoid extreme underflow
    transmission = np.exp(-tau)
    
    # Ensure transmission is between 0 and 1
    transmission = np.clip(transmission, 0.0, 1.0)
    
    return transmission


def gaussian_line(energy, line_energy, sigma, norm):
    """
    Gaussian emission line
    
    องค์ประกอบทางกายภาพ:
    - เส้นสเปกตรัมจากการ fluorescence (เช่น Fe Kα ที่ 6.4 keV)
    - เกิดจากการกระตุ้นของอะตอมในสสารรอบหลุมดำ
    - Width (σ) บอกความเร็วของสสารที่ปล่อยเส้นสเปกตรัม
    
    Parameters:
    -----------
    energy : array-like
        พลังงานของโฟตอน (keV)
    line_energy : float
        พลังงานศูนย์กลางของเส้น (keV)
    sigma : float
        Width of the line (keV)
    norm : float
        Normalization (photons/cm²/s)
        
    Returns:
    --------
    flux : array-like
        Photon flux (photons/cm²/s/keV)
    """
    gaussian = norm * np.exp(-0.5 * ((energy - line_energy) / sigma)**2)
    gaussian /= (sigma * np.sqrt(2 * np.pi))
    return gaussian


def reflection_component(energy, refl_norm, photon_index):
    """
    Simplified X-ray reflection component
    
    องค์ประกอบทางกายภาพ:
    - รังสีเอกซ์ที่สะท้อนจาก accretion disk รอบหลุมดำ
    - ประกอบด้วย Compton hump (~20-40 keV) และ iron line (~6.4 keV)
    - เป็นลายเซ็นสำคัญของ AGN ที่มี strong reflection
    
    Parameters:
    -----------
    energy : array-like
        พลังงานของโฟตอน (keV)
    refl_norm : float
        Reflection normalization
    photon_index : float
        Photon index of incident continuum
        
    Returns:
    --------
    flux : array-like
        Reflected photon flux (photons/cm²/s/keV)
        
    Note: นี่เป็น simplified model
    สำหรับการวิเคราะห์จริงควรใช้ relxill model ที่สมบูรณ์กว่า
    """
    # Simplified reflection: includes Compton hump
    # Approximate with modified power-law + Compton shoulder
    
    # Base reflection (modified power-law)
    base_refl = refl_norm * np.power(energy, -photon_index + 0.5)
    
    # Compton hump (peaks around 20-30 keV)
    compton_peak = 25.0  # keV
    compton_width = 10.0  # keV
    compton_hump = refl_norm * 0.3 * np.exp(-0.5 * ((energy - compton_peak) / compton_width)**2)
    
    # Combine components
    reflection = base_refl + compton_hump
    
    # Reflection is stronger at lower energies
    # Add energy-dependent factor
    energy_factor = np.where(energy < 10.0, 
                            1.5 - 0.05 * energy,
                            1.0)
    
    return reflection * energy_factor


def combined_model(energy, params, model_components):
    """
    Combined spectral model
    
    รวม spectral models หลายๆ ตัวเข้าด้วยกัน
    
    Parameters:
    -----------
    energy : array-like
        พลังงานของโฟตอน (keV)
    params : dict
        พารามิเตอร์ของโมเดลทั้งหมด
    model_components : list of str
        รายชื่อ components ที่ต้องการใช้
        เช่น ['powerlaw', 'gaussian', 'reflection']
        
    Returns:
    --------
    total_flux : array-like
        Total photon flux (photons/cm²/s/keV)
    """
    total_flux = np.zeros_like(energy, dtype=float)
    
    # Absorption (multiplicative component)
    absorption = 1.0
    if 'tbabs' in model_components and 'nH' in params:
        absorption = tbabs(energy, params['nH'])
    
    # Power-law continuum
    if 'powerlaw' in model_components:
        pl_flux = powerlaw(energy, 
                          params.get('pl_norm', 1.0),
                          params.get('photon_index', 2.0))
        total_flux += pl_flux
    
    # Blackbody component
    if 'blackbody' in model_components:
        bb_flux = blackbody(energy,
                           params.get('bb_norm', 0.1),
                           params.get('kT', 0.5))
        total_flux += bb_flux
    
    # Reflection component
    if 'reflection' in model_components:
        refl_flux = reflection_component(energy,
                                        params.get('refl_norm', 0.5),
                                        params.get('photon_index', 2.0))
        total_flux += refl_flux
    
    # Gaussian emission line (e.g., Fe Kα)
    if 'gaussian' in model_components:
        line_flux = gaussian_line(energy,
                                 params.get('line_energy', 6.4),
                                 params.get('line_sigma', 0.1),
                                 params.get('line_norm', 1.0))
        total_flux += line_flux
    
    # Apply absorption
    total_flux *= absorption
    
    return total_flux


def get_model_description(model_name):
    """
    คืนค่าคำอธิบายโมเดลแต่ละตัว
    
    Parameters:
    -----------
    model_name : str
        ชื่อโมเดล
        
    Returns:
    --------
    description : dict
        คำอธิบายโมเดลและพารามิเตอร์
    """
    descriptions = {
        'powerlaw': {
            'name': 'Power-law Continuum',
            'physics': 'อธิบาย X-ray continuum จาก Comptonization ใน corona รอบหลุมดำ',
            'parameters': {
                'norm': 'Normalization (photons/cm²/s/keV at 1 keV)',
                'photon_index': 'Photon index Γ (ความชัน, typical 1.7-2.0 for AGN)'
            },
            'typical_values': {
                'photon_index': (1.5, 2.5)
            }
        },
        'tbabs': {
            'name': 'Photoelectric Absorption',
            'physics': 'การดูดกลืนรังสีเอกซ์โดย neutral hydrogen ในทาง Milky Way และ host galaxy',
            'parameters': {
                'nH': 'Hydrogen column density (10²² atoms/cm²)'
            },
            'typical_values': {
                'nH': (0.01, 10.0)
            }
        },
        'blackbody': {
            'name': 'Blackbody (Thermal) Emission',
            'physics': 'Thermal emission จาก accretion disk (soft X-ray)',
            'parameters': {
                'norm': 'Normalization',
                'kT': 'Temperature (keV, 1 keV ≈ 1.16×10⁷ K)'
            },
            'typical_values': {
                'kT': (0.1, 2.0)
            }
        },
        'reflection': {
            'name': 'X-ray Reflection',
            'physics': 'รังสีเอกซ์ที่สะท้อนจาก accretion disk รวมถึง Compton hump และ iron line',
            'parameters': {
                'refl_norm': 'Reflection normalization',
                'photon_index': 'Incident power-law index'
            },
            'typical_values': {
                'refl_norm': (0.0, 2.0)
            }
        },
        'gaussian': {
            'name': 'Gaussian Emission Line',
            'physics': 'เส้นสเปกตรัมจาก fluorescence (เช่น Fe Kα ที่ 6.4 keV)',
            'parameters': {
                'line_energy': 'Line center energy (keV)',
                'line_sigma': 'Line width σ (keV)',
                'line_norm': 'Line normalization (photons/cm²/s)'
            },
            'typical_values': {
                'line_energy': (6.2, 6.7),
                'line_sigma': (0.01, 0.5)
            }
        }
    }
    
    return descriptions.get(model_name, {})
