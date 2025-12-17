# -*- coding: utf-8 -*-
"""
Spectral Models Package

โฟลเดอร์นี้เก็บ Spectral Models สำหรับการ fit สเปกตรัมรังสีเอกซ์

Models Available:
    - powerlaw: Power-law continuum
    - blackbody: Blackbody (thermal) emission
    - tbabs: Photoelectric absorption
    - gaussian: Gaussian emission line
    - reflection: X-ray reflection component
"""

from .powerlaw import powerlaw
from .blackbody import blackbody
from .tbabs import tbabs
from .gaussian import gaussian_line
from .reflection import reflection_component

# รายชื่อ models ทั้งหมด
AVAILABLE_MODELS = [
    'powerlaw',
    'blackbody',
    'tbabs',
    'gaussian',
    'reflection'
]

def combined_model(energy, params, model_components):
    """
    รวม spectral models หลายๆ ตัวเข้าด้วยกัน
    
    Parameters:
        energy (array): Energy array (keV)
        params (dict): Model parameters
        model_components (list): List of model names to include
        
    Returns:
        array: Total photon flux (photons/cm²/s/keV)
        
    Example:
        >>> flux = combined_model(energy, params, ['powerlaw', 'gaussian'])
    """
    import numpy as np
    
    total_flux = np.zeros_like(energy, dtype=float)
    
    # Absorption (multiplicative)
    absorption = 1.0
    if 'tbabs' in model_components and 'nH' in params:
        absorption = tbabs(energy, params['nH'])
    
    # Power-law
    if 'powerlaw' in model_components:
        total_flux += powerlaw(
            energy, 
            params.get('pl_norm', 1.0),
            params.get('photon_index', 2.0)
        )
    
    # Blackbody
    if 'blackbody' in model_components:
        total_flux += blackbody(
            energy,
            params.get('bb_norm', 0.1),
            params.get('kT', 0.5)
        )
    
    # Reflection
    if 'reflection' in model_components:
        total_flux += reflection_component(
            energy,
            params.get('refl_norm', 0.5),
            params.get('photon_index', 2.0)
        )
    
    # Gaussian line
    if 'gaussian' in model_components:
        total_flux += gaussian_line(
            energy,
            params.get('line_energy', 6.4),
            params.get('line_sigma', 0.1),
            params.get('line_norm', 1.0)
        )
    
    # Apply absorption
    total_flux *= absorption
    
    return total_flux
