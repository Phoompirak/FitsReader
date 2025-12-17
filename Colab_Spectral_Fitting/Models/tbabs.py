# -*- coding: utf-8 -*-
"""
Photoelectric Absorption Model (TBabs)

TBabs (Tübingen-Boulder) model อธิบายการดูดกลืน X-rays โดย neutral hydrogen

Physics:
    - X-rays ถูกดูดกลืนโดย interstellar medium ในทางช้างเผือก
    - และโดย gas ใน host galaxy ของ AGN
    - ส่งผลมากในช่วง soft X-ray (< 2 keV)

Formula:
    T(E) = exp(-σ(E) × nH)
    
    where:
        T(E) = Transmission factor (0-1)
        σ(E) = Photoelectric cross-section (cm²)
        nH = Hydrogen column density (10²² atoms/cm²)

Reference:
    Morrison & McCammon (1983) ApJ 270, 119
"""

import numpy as np


def tbabs(energy, nH):
    """
    Photoelectric absorption (simplified TBabs model)
    
    Parameters:
        energy (array): Energy array (keV)
        nH (float): Hydrogen column density (10²² atoms/cm²)
        
    Returns:
        array: Transmission factor (0-1)
        
    Typical Values:
        nH: 0.01 - 10.0 (Galactic ~0.01-0.1, obscured AGN: 1-100)
        
    Example:
        >>> energy = np.linspace(0.3, 10, 100)
        >>> transmission = tbabs(energy, nH=0.5)
    """
    E = np.asarray(energy, dtype=float)
    
    # Ensure minimum energy to avoid division issues
    E = np.maximum(E, 0.03)
    
    # Cross-section approximation (10^-24 cm² per H atom)
    # Based on Morrison & McCammon (1983)
    
    # For E < 0.1 keV
    sigma_soft = 180.0 * np.power(E, -3.0)
    
    # For 0.1 < E < 2 keV
    sigma_mid = (17.3 + 608.1/E) * np.power(E, -3.0)
    
    # For E > 2 keV  
    sigma_hard = 600.0 * np.power(E, -3.0)
    
    # Combine piecewise
    sigma = np.where(E < 0.1, sigma_soft,
                    np.where(E < 2.0, sigma_mid, sigma_hard))
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 0.0)
    
    # Optical depth τ = σ × nH
    # σ in 10^-24 cm², nH in 10^22 cm^-2
    # τ = σ × nH × 10^-2
    tau = sigma * nH * 1e-2
    
    # Clip tau to avoid numerical issues
    tau = np.minimum(tau, 100.0)
    
    # Transmission = exp(-τ)
    transmission = np.exp(-tau)
    
    # Clip to valid range
    transmission = np.clip(transmission, 0.0, 1.0)
    
    return transmission


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    energy = np.linspace(0.3, 10, 100)
    
    # ทดสอบ absorption ที่ต่าง nH
    for nH in [0.01, 0.1, 1.0]:
        trans = tbabs(energy, nH=nH)
        plt.semilogx(energy, trans, label=f'nH = {nH}×10²² cm⁻²')
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Transmission')
    plt.title('Photoelectric Absorption (TBabs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.show()
