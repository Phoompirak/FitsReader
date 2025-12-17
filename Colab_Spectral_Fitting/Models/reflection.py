# -*- coding: utf-8 -*-
"""
X-ray Reflection Component Model

Reflection component อธิบายรังสีเอกซ์ที่สะท้อนจาก accretion disk

Physics:
    - X-rays จาก corona สะท้อนจาก accretion disk
    - สร้าง Compton hump ที่ ~20-40 keV
    - สร้าง fluorescent lines เช่น Fe Kα
    - Reflection fraction บอกสัดส่วนของแสงที่สะท้อน

Note:
    นี่เป็น simplified model สำหรับการวิเคราะห์เบื้องต้น
    สำหรับการวิเคราะห์จริงควรใช้ relxill model ที่สมบูรณ์กว่า
"""

import numpy as np


def reflection_component(energy, refl_norm, photon_index):
    """
    Simplified X-ray reflection component
    
    Parameters:
        energy (array): Energy array (keV)
        refl_norm (float): Reflection normalization
        photon_index (float): Photon index of incident continuum
        
    Returns:
        array: Reflected photon flux (photons/cm²/s/keV)
        
    Typical Values:
        refl_norm: 0.0 - 2.0 (fraction of incident flux)
        photon_index: 1.5 - 2.5
        
    Example:
        >>> energy = np.linspace(0.5, 50, 200)
        >>> flux = reflection_component(energy, refl_norm=0.5, photon_index=2.0)
    """
    # Base reflection (modified power-law)
    # Softer than incident by ~0.5
    base_refl = refl_norm * np.power(energy, -photon_index + 0.5)
    
    # Compton hump (peaks around 20-30 keV)
    compton_peak = 25.0   # keV
    compton_width = 10.0  # keV
    compton_hump = refl_norm * 0.3 * np.exp(
        -0.5 * ((energy - compton_peak) / compton_width)**2
    )
    
    # Combine components
    reflection = base_refl + compton_hump
    
    # Energy-dependent factor (stronger at lower energies)
    energy_factor = np.where(
        energy < 10.0, 
        1.5 - 0.05 * energy,
        1.0
    )
    
    return reflection * energy_factor


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    energy = np.linspace(0.5, 50, 200)
    
    # ทดสอบ reflection ที่ต่าง reflection fraction
    for refl in [0.3, 0.5, 1.0]:
        flux = reflection_component(energy, refl_norm=refl, photon_index=2.0)
        plt.loglog(energy, flux, label=f'R = {refl}')
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Flux (photons/cm²/s/keV)')
    plt.title('X-ray Reflection Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=25, color='gray', linestyle='--', alpha=0.5, label='Compton peak')
    plt.show()
