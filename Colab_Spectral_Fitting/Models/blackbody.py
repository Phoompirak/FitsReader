# -*- coding: utf-8 -*-
"""
Blackbody (Thermal) Spectral Model

Blackbody spectrum อธิบาย thermal emission จาก accretion disk

Physics:
    - Thermal radiation จากสสารร้อนใน accretion disk รอบหลุมดำ
    - พบเด่นในช่วง soft X-ray (< 2 keV)
    - kT คืออุณหภูมิในหน่วย keV (1 keV ≈ 1.16 × 10^7 K)

Formula (simplified):
    F(E) = norm × E² / (exp(E/kT) - 1)
    
    where:
        F(E) = Photon flux density
        E = Energy (keV)
        kT = Temperature (keV)
"""

import numpy as np


def blackbody(energy, norm, kT):
    """
    Blackbody (Planck) spectrum
    
    Parameters:
        energy (array): Energy array (keV)
        norm (float): Normalization factor
        kT (float): Temperature (keV), 1 keV ≈ 1.16×10⁷ K
        
    Returns:
        array: Photon flux (photons/cm²/s/keV)
        
    Typical Values:
        norm: 0.01 - 1.0
        kT: 0.1 - 2.0 keV (soft X-ray sources)
        
    Example:
        >>> energy = np.linspace(0.1, 5, 100)
        >>> flux = blackbody(energy, norm=0.1, kT=0.5)
    """
    # x = E/kT (dimensionless energy)
    x = energy / kT
    
    # Avoid overflow for large x
    x = np.clip(x, 0, 100)
    
    # Blackbody photon spectrum
    spectrum = norm * (energy**2) / (np.exp(x) - 1.0 + 1e-10)
    
    # Set to zero where overflow would occur
    spectrum = np.where(x < 100, spectrum, 0.0)
    
    return spectrum


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    energy = np.linspace(0.1, 5, 100)
    
    # ทดสอบ blackbody ที่ต่างอุณหภูมิ
    for kT in [0.2, 0.5, 1.0]:
        flux = blackbody(energy, norm=0.1, kT=kT)
        plt.loglog(energy, flux, label=f'kT = {kT} keV')
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Flux (photons/cm²/s/keV)')
    plt.title('Blackbody Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
