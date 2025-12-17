# -*- coding: utf-8 -*-
"""
Gaussian Emission Line Model

Gaussian line model สำหรับ emission lines เช่น Fe Kα

Physics:
    - เกิดจาก fluorescence ของอะตอมเหล็กใน accretion disk
    - Fe Kα line อยู่ที่ ~6.4 keV (neutral Fe)
    - Line width (σ) บอกความเร็วของสสารที่ปล่อยเส้น

Formula:
    F(E) = (norm / (σ√(2π))) × exp(-0.5 × ((E - E₀)/σ)²)
    
    where:
        F(E) = Photon flux density
        E₀ = Line center energy (keV)
        σ = Line width (keV)
        norm = Total line flux (photons/cm²/s)
"""

import numpy as np


def gaussian_line(energy, line_energy, sigma, norm):
    """
    Gaussian emission line
    
    Parameters:
        energy (array): Energy array (keV)
        line_energy (float): Center energy of line (keV)
        sigma (float): Width of line (keV)
        norm (float): Total line flux (photons/cm²/s)
        
    Returns:
        array: Photon flux (photons/cm²/s/keV)
        
    Common Lines:
        Fe Kα (neutral): 6.40 keV
        Fe Kα (ionized): 6.7-6.9 keV
        Fe Kβ: 7.06 keV
        
    Typical Values:
        sigma: 0.01 - 0.5 keV
        norm: 0.001 - 10 photons/cm²/s
        
    Example:
        >>> energy = np.linspace(5, 8, 100)
        >>> flux = gaussian_line(energy, line_energy=6.4, sigma=0.1, norm=1.0)
    """
    # Gaussian profile
    gaussian = norm * np.exp(-0.5 * ((energy - line_energy) / sigma)**2)
    
    # Normalize by width to get flux density
    gaussian /= (sigma * np.sqrt(2 * np.pi))
    
    return gaussian


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    energy = np.linspace(5, 8, 100)
    
    # ทดสอบ Gaussian lines ที่ต่าง width
    for sig in [0.05, 0.1, 0.3]:
        flux = gaussian_line(energy, line_energy=6.4, sigma=sig, norm=1.0)
        plt.plot(energy, flux, label=f'σ = {sig} keV')
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Flux (photons/cm²/s/keV)')
    plt.title('Gaussian Emission Line (Fe Kα at 6.4 keV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=6.4, color='gray', linestyle='--', alpha=0.5)
    plt.show()
