# -*- coding: utf-8 -*-
"""
Spectral Fitting Analysis - Main Script
สำหรับ Google Colab

โค้ดนี้ใช้สำหรับวิเคราะห์และ fit สเปกตรัมรังสีเอกซ์ของ AGN
รองรับการกำหนดช่วง Energy Range ก่อนรัน

วิธีใช้:
1. Upload โฟลเดอร์นี้ไปที่ Google Drive
2. เปิดไฟล์นี้ใน Google Colab
3. แก้ไข parameters ใน Section 1 (Energy Range, File Paths)
4. เลือก models ที่จะใช้ใน Section 2
5. Run ทุก cells

Dependencies:
    pip install astropy numpy scipy matplotlib
"""

#%% ============================================================
# SECTION 1: CONFIGURATION - กำหนดค่าก่อนรัน
# ============================================================

# ==============================
# 🎯 ENERGY RANGE (keV)
# กำหนดช่วงพลังงานที่จะวิเคราะห์
# ==============================
ENERGY_MIN = 0.3    # keV (ขอบล่าง)
ENERGY_MAX = 10.0   # keV (ขอบบน)

# ==============================
# 📁 FILE PATHS
# กำหนด path ไปยังไฟล์ข้อมูล
# ==============================
# ถ้าใช้ Google Drive ให้ mount ก่อน:
# from google.colab import drive
# drive.mount('/content/drive')

SOURCE_SPECTRUM_PATH = '/content/drive/MyDrive/data/source.pha'    # 📊 Source Spectrum
BACKGROUND_PATH = '/content/drive/MyDrive/data/background.pha'     # 🌌 Background Spectrum
ARF_PATH = '/content/drive/MyDrive/data/source.arf'                # 📈 ARF File
RMF_PATH = '/content/drive/MyDrive/data/source.rmf'                # 🔲 RMF File

# ==============================
# 🔧 FITTING PARAMETERS
# ==============================
N_STEPS = 10           # จำนวน steps สำหรับ grid search (5-20)
USE_BRUTE_FORCE = True # True = grid search, False = optimization

#%% ============================================================
# SECTION 2: SELECT MODELS - เลือก models ที่จะใช้
# ============================================================

# เลือก models โดย uncomment ที่ต้องการ
SELECTED_MODELS = [
    'powerlaw',      # ✅ Power-law continuum (default)
    'tbabs',         # ✅ Absorption (default)
    # 'blackbody',   # Thermal emission
    # 'gaussian',    # Fe Kα emission line
    # 'reflection',  # X-ray reflection
]

# ==============================
# 🎛️ INITIAL PARAMETERS
# กำหนดค่าเริ่มต้นสำหรับ fit
# ==============================
INITIAL_PARAMS = {
    # Power-law
    'pl_norm': 0.01,        # Normalization (photons/cm²/s/keV at 1 keV)
    'photon_index': 2.0,    # Photon index Γ
    
    # Absorption
    'nH': 0.1,              # Hydrogen column density (10²² atoms/cm²)
    
    # Blackbody (ถ้าใช้)
    'bb_norm': 0.1,
    'kT': 0.5,              # Temperature (keV)
    
    # Gaussian line (ถ้าใช้)
    'line_energy': 6.4,     # Fe Kα at 6.4 keV
    'line_sigma': 0.1,      # Line width (keV)
    'line_norm': 1.0,       # Line normalization
    
    # Reflection (ถ้าใช้)
    'refl_norm': 0.5,
}

# ==============================
# 📊 PARAMETER RANGES FOR GRID SEARCH
# ช่วงค่าสำหรับ brute-force search
# ==============================
PARAM_RANGES = {
    'pl_norm': (0.001, 1.0),
    'photon_index': (1.5, 2.5),
    'nH': (0.01, 1.0),
}

#%% ============================================================
# SECTION 3: LOAD DATA - โหลดข้อมูล
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Import local modules
import sys
sys.path.append('.')  # Add current directory

from data_loader import (
    read_spectrum_file, 
    read_arf_file, 
    read_rmf_file,
    subtract_background,
    get_energy_from_response,
    filter_energy_range,
    fold_model_through_response
)
from Models import combined_model

print("=" * 60)
print("📊 X-ray Spectral Fitting Analysis")
print("=" * 60)
print(f"Energy Range: {ENERGY_MIN} - {ENERGY_MAX} keV")
print(f"Selected Models: {SELECTED_MODELS}")
print("=" * 60)

# โหลดไฟล์
print("\n🔄 Loading files...")

source_spec = read_spectrum_file(SOURCE_SPECTRUM_PATH)
if source_spec:
    print(f"  ✅ Source spectrum loaded: {len(source_spec.counts)} channels")

bkg_spec = read_spectrum_file(BACKGROUND_PATH)
if bkg_spec:
    print(f"  ✅ Background spectrum loaded")

arf_data = read_arf_file(ARF_PATH)
if arf_data:
    print(f"  ✅ ARF loaded: {arf_data.energy_lo.min():.2f}-{arf_data.energy_hi.max():.2f} keV")

rmf_data = read_rmf_file(RMF_PATH)
if rmf_data:
    print(f"  ✅ RMF loaded")

#%% ============================================================
# SECTION 4: PREPARE DATA - เตรียมข้อมูล
# ============================================================

print("\n🔬 Preparing data...")

# Get energy array
energy = get_energy_from_response(arf_data, rmf_data)
print(f"  Energy bins: {len(energy)}")

# Background subtraction
if source_spec and bkg_spec:
    net_counts, net_error = subtract_background(source_spec, bkg_spec)
    print(f"  ✅ Background subtracted")
else:
    net_counts = source_spec.counts if source_spec else None
    net_error = np.sqrt(np.maximum(net_counts, 1.0)) if net_counts is not None else None
    print(f"  ⚠️ No background subtraction")

# Convert to count rate
if source_spec and source_spec.exposure:
    observed_rate = net_counts / source_spec.exposure
    observed_error = net_error / source_spec.exposure
else:
    observed_rate = net_counts
    observed_error = net_error

# Filter by energy range
energy_filtered, rate_filtered, error_filtered = filter_energy_range(
    energy, observed_rate, observed_error, 
    e_min=ENERGY_MIN, e_max=ENERGY_MAX
)

print(f"  Filtered to {ENERGY_MIN}-{ENERGY_MAX} keV: {len(energy_filtered)} bins")

#%% ============================================================
# SECTION 5: SPECTRAL FITTING - ฟิตสเปกตรัม
# ============================================================

print("\n🎯 Starting spectral fitting...")

def calculate_chi_squared(params, energy, obs_rate, obs_error, models, response):
    """คำนวณ chi-squared"""
    # Calculate model
    model_flux = combined_model(energy, params, models)
    
    # Fold through response
    if response and response.arf is not None:
        model_rate = fold_model_through_response(model_flux, response)
    else:
        model_rate = model_flux
    
    # Chi-squared
    mask = obs_error > 0
    chi2 = np.sum(((obs_rate[mask] - model_rate[mask]) / obs_error[mask])**2)
    
    return chi2


def fit_spectrum_scipy(energy, obs_rate, obs_error, models, initial_params, response):
    """Fit using scipy.optimize.minimize"""
    
    # Get parameter names that are relevant
    param_names = list(initial_params.keys())
    initial_values = [initial_params[p] for p in param_names]
    
    def objective(param_array):
        params = dict(zip(param_names, param_array))
        return calculate_chi_squared(params, energy, obs_rate, obs_error, models, response)
    
    # Bounds
    bounds = [
        (0.0001, 100),   # pl_norm
        (1.0, 3.0),      # photon_index
        (0.001, 10),     # nH
        (0.001, 10),     # bb_norm
        (0.05, 3.0),     # kT
        (5.5, 7.5),      # line_energy
        (0.01, 0.5),     # line_sigma
        (0.001, 100),    # line_norm
        (0.0, 5.0),      # refl_norm
    ]
    bounds = bounds[:len(param_names)]
    
    # Minimize
    result = minimize(objective, initial_values, method='L-BFGS-B', bounds=bounds)
    
    # Best params
    best_params = dict(zip(param_names, result.x))
    
    # Calculate chi-squared
    chi2 = result.fun
    dof = np.sum(obs_error > 0) - len(param_names)
    
    return {
        'best_params': best_params,
        'chi_squared': chi2,
        'dof': dof,
        'reduced_chi2': chi2/dof if dof > 0 else 0,
        'success': result.success
    }


def brute_force_fit(energy, obs_rate, obs_error, models, param_ranges, n_steps, response):
    """Grid search fitting"""
    import itertools
    
    param_names = list(param_ranges.keys())
    grids = [np.linspace(r[0], r[1], n_steps) for r in param_ranges.values()]
    
    best_chi2 = float('inf')
    best_params = None
    total = np.prod([len(g) for g in grids])
    
    print(f"  Testing {total} combinations...")
    
    for i, combo in enumerate(itertools.product(*grids)):
        params = dict(zip(param_names, combo))
        chi2 = calculate_chi_squared(params, energy, obs_rate, obs_error, models, response)
        
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_params = params.copy()
        
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{total} | Best χ² = {best_chi2:.2f}")
    
    dof = np.sum(obs_error > 0) - len(param_names)
    
    return {
        'best_params': best_params,
        'chi_squared': best_chi2,
        'dof': dof,
        'reduced_chi2': best_chi2/dof if dof > 0 else 0
    }


# Run fitting
if USE_BRUTE_FORCE:
    print("  Method: Brute-force grid search")
    fit_result = brute_force_fit(
        energy_filtered, rate_filtered, error_filtered,
        SELECTED_MODELS, PARAM_RANGES, N_STEPS, arf_data
    )
else:
    print("  Method: Scipy optimization")
    fit_result = fit_spectrum_scipy(
        energy_filtered, rate_filtered, error_filtered,
        SELECTED_MODELS, INITIAL_PARAMS, arf_data
    )

# Print results
print("\n" + "=" * 60)
print("📈 FITTING RESULTS")
print("=" * 60)
print(f"χ² = {fit_result['chi_squared']:.2f}")
print(f"dof = {fit_result['dof']}")
print(f"χ²/dof = {fit_result['reduced_chi2']:.3f}")
print("\nBest-fit parameters:")
for key, val in fit_result['best_params'].items():
    print(f"  {key}: {val:.4f}")

#%% ============================================================
# SECTION 6: PLOT RESULTS - แสดงผลกราฟ
# ============================================================

print("\n📊 Plotting results...")

# Calculate best-fit model
best_model_flux = combined_model(energy_filtered, fit_result['best_params'], SELECTED_MODELS)
if arf_data and arf_data.arf is not None:
    # Filter ARF to match energy range
    arf_mask = (arf_data.energy_mid >= ENERGY_MIN) & (arf_data.energy_mid <= ENERGY_MAX)
    arf_filtered = type(arf_data)()
    arf_filtered.arf = arf_data.arf[arf_mask]
    arf_filtered.energy_lo = arf_data.energy_lo[arf_mask]
    arf_filtered.energy_hi = arf_data.energy_hi[arf_mask]
    best_model_rate = fold_model_through_response(best_model_flux, arf_filtered)
else:
    best_model_rate = best_model_flux

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                          gridspec_kw={'height_ratios': [3, 1]})

# Top panel: Spectrum
ax1 = axes[0]
ax1.errorbar(energy_filtered, rate_filtered, yerr=error_filtered, 
             fmt='o', markersize=3, alpha=0.7, label='Data')
ax1.plot(energy_filtered, best_model_rate, 'r-', linewidth=2, label='Best-fit model')
ax1.set_ylabel('Count Rate (counts/s/keV)')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend()
ax1.set_title(f'X-ray Spectrum (χ²/dof = {fit_result["reduced_chi2"]:.2f})')
ax1.grid(True, alpha=0.3)

# Bottom panel: Residuals
ax2 = axes[1]
residuals = (rate_filtered - best_model_rate) / error_filtered
ax2.axhline(y=0, color='gray', linestyle='--')
ax2.plot(energy_filtered, residuals, 'ko', markersize=3)
ax2.set_xlabel('Energy (keV)')
ax2.set_ylabel('Residuals (σ)')
ax2.set_ylim(-5, 5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectral_fit_result.png', dpi=150)
plt.show()

print("\n✅ Analysis complete!")
print("  Results saved to: spectral_fit_result.png")
