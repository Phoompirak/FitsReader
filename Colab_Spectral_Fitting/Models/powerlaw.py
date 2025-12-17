# -*- coding: utf-8 -*-
"""
Power-law Spectral Model

Power-law เป็นโมเดลพื้นฐานที่ใช้อธิบาย continuum emission จาก AGN

Physics:
    - เกิดจาก inverse Compton scattering ของโฟตอนใน corona รอบหลุมดำ
    - Photon index (Γ) บอกความชัน, ค่าปกติ: 1.7-2.0 สำหรับ AGN

Formula:
    F(E) = norm × E^(-Γ)
    
    where:
        F(E) = Photon flux density (photons/cm²/s/keV)
        E = Energy (keV)
        norm = Normalization at 1 keV (photons/cm²/s/keV)
        Γ = Photon index (slope)
"""

import numpy as np


def powerlaw(energy, norm, photon_index):
    """
    Power-law model สำหรับ X-ray continuum
    
    Parameters:
        energy (array): Energy array (keV)
        norm (float): Normalization factor (photons/cm²/s/keV at 1 keV)
        photon_index (float): Photon index Γ (ความชัน)
        
    Returns:
        array: Photon flux (photons/cm²/s/keV)
        
    Typical Values:
        norm: 0.001 - 10.0
        photon_index: 1.5 - 3.0 (typical AGN: 1.7-2.0)
        
    Example:
        >>> energy = np.linspace(0.5, 10, 100)
        >>> flux = powerlaw(energy, norm=0.01, photon_index=2.0)
    """
    return norm * np.power(energy, -photon_index)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # สร้าง energy grid
    energy = np.linspace(0.5, 10, 100)
    
    # ทดสอบ power-law ที่ต่าง Γ
    for gamma in [1.5, 2.0, 2.5]:
        flux = powerlaw(energy, norm=0.01, photon_index=gamma)
        plt.loglog(energy, flux, label=f'Γ = {gamma}')
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Flux (photons/cm²/s/keV)')
    plt.title('Power-law Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
