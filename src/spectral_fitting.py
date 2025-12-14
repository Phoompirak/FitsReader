"""
Spectral Fitting Module
สำหรับการฟิตสเปกตรัมรังสีเอกซ์ของ AGN

โมดูลนี้ทำหน้าที่:
1. อ่านข้อมูลสเปกตรัม (source, background, ARF, RMF)
2. ทำการ fitting ด้วย chi-squared minimization
3. คำนวณ uncertainties และ goodness-of-fit
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
from astropy.io import fits
import spectral_models as sm
import time

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


class SpectrumData:
    """
    Class สำหรับเก็บข้อมูลสเปกตรัม
    """
    def __init__(self):
        self.channel = None
        self.counts = None
        self.exposure = None
        self.backscal = 1.0
        self.grouping = None
        self.quality = None
        
    def count_rate(self):
        """คืนค่า count rate (counts/s)"""
        if self.exposure is not None and self.exposure > 0:
            return self.counts / self.exposure
        return self.counts
    
    def count_rate_error(self):
        """คืนค่า error ของ count rate (Poisson statistics)"""
        if self.exposure is not None and self.exposure > 0:
            return np.sqrt(np.maximum(self.counts, 1.0)) / self.exposure
        return np.sqrt(np.maximum(self.counts, 1.0))


class ResponseData:
    """
    Class สำหรับเก็บข้อมูล response (ARF และ RMF)
    """
    def __init__(self):
        self.energy_lo = None  # keV
        self.energy_hi = None  # keV
        self.energy_mid = None  # keV
        self.arf = None  # Effective area (cm²)
        self.rmf_matrix = None  # Response matrix
        self.channel_lo = None
        self.channel_hi = None
        

def read_spectrum_file(filepath):
    """
    อ่านไฟล์สเปกตรัม FITS
    
    Parameters:
    -----------
    filepath : str
        Path to spectrum FITS file
        
    Returns:
    --------
    spectrum : SpectrumData
        ข้อมูลสเปกตรัม
    """
    spectrum = SpectrumData()
    
    try:
        with fits.open(filepath) as hdul:
            # อ่านข้อมูลจาก SPECTRUM extension (usually HDU 1)
            if len(hdul) > 1:
                data = hdul[1].data
                header = hdul[1].header
                
                # Read columns
                if 'CHANNEL' in data.columns.names:
                    spectrum.channel = data['CHANNEL']
                if 'COUNTS' in data.columns.names:
                    spectrum.counts = data['COUNTS'].astype(float)
                if 'GROUPING' in data.columns.names:
                    spectrum.grouping = data['GROUPING']
                if 'QUALITY' in data.columns.names:
                    spectrum.quality = data['QUALITY']
                    
                # Read header keywords
                if 'EXPOSURE' in header:
                    spectrum.exposure = header['EXPOSURE']
                if 'BACKSCAL' in header:
                    spectrum.backscal = header['BACKSCAL']
                    
    except Exception as e:
        print(f"Error reading spectrum file: {e}")
        return None
        
    return spectrum


def read_arf_file(filepath):
    """
    อ่านไฟล์ ARF (Ancillary Response File)
    
    Parameters:
    -----------
    filepath : str
        Path to ARF file
        
    Returns:
    --------
    response : ResponseData (partial)
        ข้อมูล ARF
    """
    response = ResponseData()
    
    try:
        with fits.open(filepath) as hdul:
            if len(hdul) > 1:
                data = hdul[1].data
                
                if 'ENERG_LO' in data.columns.names:
                    response.energy_lo = data['ENERG_LO']
                if 'ENERG_HI' in data.columns.names:
                    response.energy_hi = data['ENERG_HI']
                if 'SPECRESP' in data.columns.names:
                    response.arf = data['SPECRESP']
                    
                # Calculate energy midpoints
                if response.energy_lo is not None and response.energy_hi is not None:
                    response.energy_mid = (response.energy_lo + response.energy_hi) / 2.0
                    
    except Exception as e:
        print(f"Error reading ARF file: {e}")
        return None
        
    return response


def read_rmf_file(filepath):
    """
    อ่านไฟล์ RMF (Redistribution Matrix File)
    
    Parameters:
    -----------
    filepath : str
        Path to RMF file
        
    Returns:
    --------
    response : ResponseData (partial)
        ข้อมูล RMF
    """
    response = ResponseData()
    
    try:
        with fits.open(filepath) as hdul:
            # Read EBOUNDS extension for channel information
            ebounds_hdu = None
            matrix_hdu = None
            
            for i, hdu in enumerate(hdul):
                if hdu.name == 'EBOUNDS':
                    ebounds_hdu = hdu
                elif hdu.name in ['MATRIX', 'SPECRESP MATRIX']:
                    matrix_hdu = hdu
                    
            # Read channel boundaries from EBOUNDS
            if ebounds_hdu is not None:
                data = ebounds_hdu.data
                if 'CHANNEL' in data.columns.names:
                    response.channel_lo = data['CHANNEL']
                    response.channel_hi = data['CHANNEL']
                    
            # Read matrix data
            if matrix_hdu is not None:
                data = matrix_hdu.data
                
                if 'ENERG_LO' in data.columns.names:
                    response.energy_lo = data['ENERG_LO']
                if 'ENERG_HI' in data.columns.names:
                    response.energy_hi = data['ENERG_HI']
                    
                # Calculate energy midpoints
                if response.energy_lo is not None and response.energy_hi is not None:
                    response.energy_mid = (response.energy_lo + response.energy_hi) / 2.0
                    
                # Note: Full RMF matrix reading is complex
                # For simplified analysis, we'll use diagonal approximation
                # or energy-to-channel mapping
                
    except Exception as e:
        print(f"Error reading RMF file: {e}")
        return None
        
    return response


def fold_model_through_response(model_flux, response):
    """
    Fold theoretical model through instrument response
    
    การแปลง model spectrum (photon flux density) ให้เป็น predicted count rate density
    ตาม response ของเครื่องมือ
    
    Model: Predicted count rate density = Model(E) × ARF(E)
    
    Parameters:
    -----------
    model_flux : array
        Model photon flux density (photons/cm²/s/keV)
    response : ResponseData
        Response data (ARF + RMF)
        
    Returns:
    --------
    predicted_rate : array
        Predicted count rate density (counts/s/keV)
        
    Note: Simplified version - assumes diagonal response (no RMF redistribution)
    
    Units:
    - Input: photons/cm²/s/keV
    - ARF: cm²
    - Output: counts/s/keV = (photons/cm²/s/keV) × (cm²)
    """
    # Simplified: assume energy bins map to channels (diagonal RMF)
    # Full implementation would multiply by RMF matrix for energy redistribution
    
    if response.arf is not None:
        # Predicted count rate density = Model flux density × Effective area
        # Units: [photons/cm²/s/keV] × [cm²] = [counts/s/keV]
        # To get counts/s (which is what we observe), we need to multiply by bin width (keV)
        # Predicted Rate = Flux Density * ARF * dE
        
        predicted_rate = model_flux * response.arf
        
        # Apply energy bin width (dE) if available
        if response.energy_hi is not None and response.energy_lo is not None:
             # Ensure shapes match
             if len(response.energy_hi) == len(predicted_rate):
                 dE = response.energy_hi - response.energy_lo
                 predicted_rate *= dE
             elif len(response.energy_hi) > len(predicted_rate):
                 # Handle case where response is full but model is filtered/partial
                 # (simplistic assumption: linear mapping or pre-filtered)
                 pass 
        
        return predicted_rate
    
    return model_flux


def calculate_chi_squared(observed_counts, predicted_counts, errors, exposure=1.0):
    """
    คำนวณ chi-squared statistic
    
    χ² = Σ [(observed - predicted)² / error²]
    
    Parameters:
    -----------
    observed_counts : array
        Observed counts
    predicted_counts : array
        Predicted counts from model
    errors : array
        Uncertainties on observed counts
    exposure : float
        Exposure time (s)
        
    Returns:
    --------
    chi_squared : float
        Chi-squared value
    dof : int
        Degrees of freedom
    """
    # Filter out bad channels (zero errors or negative values)
    mask = (errors > 0) & (observed_counts >= 0) & (predicted_counts >= 0)
    
    obs = observed_counts[mask]
    pred = predicted_counts[mask] * exposure  # Convert rate to counts
    err = errors[mask]
    
    # Chi-squared
    chi_sq = np.sum(((obs - pred) / err) ** 2)
    
    # Degrees of freedom = number of data points - number of free parameters
    # (will subtract n_params later)
    dof = len(obs)
    
    return chi_sq, dof


def fit_spectrum(energy, observed_rate, observed_error, 
                 model_components, initial_params, 
                 param_bounds=None, exposure=1.0, response=None):
    """
    ฟิตสเปกตรัมด้วย chi-squared minimization
    
    IMPORTANT: ฟังก์ชันนี้ fold model spectrum ผ่าน ARF response ก่อนเปรียบเทียบกับข้อมูล
    เพื่อให้การฟิตถูกต้องตามหลักฟิสิกส์
    
    Parameters:
    -----------
    energy : array
        Energy bins (keV)
    observed_rate : array
        Observed count rate (counts/s)
    observed_error : array
        Error on count rate (counts/s)
    model_components : list
        List of model components to use
    initial_params : dict
        Initial parameter values
    param_bounds : dict, optional
        Parameter bounds (min, max)
    exposure : float
        Exposure time (s)
    response : ResponseData, optional
        ARF response data for folding model through instrument response
        
    Returns:
    --------
    result : dict
        Fitting results including best-fit parameters, errors, chi-squared
    """
    # Create parameter list and bounds
    param_names = []
    param_initial = []
    param_lower = []
    param_upper = []
    
    # Define parameter order and bounds
    param_config = {
        'pl_norm': (0.001, 0.01, 100.0),
        'photon_index': (1.0, 2.0, 3.0),
        'nH': (0.0, 0.1, 10.0),
        'refl_norm': (0.0, 0.5, 5.0),
        'bb_norm': (0.0, 0.1, 10.0),
        'kT': (0.05, 0.5, 3.0),
        'line_energy': (6.0, 6.4, 7.0),
        'line_sigma': (0.01, 0.1, 0.5),
        'line_norm': (0.0, 1.0, 100.0)
    }
    
    # Build parameter arrays
    for key, (lower, initial, upper) in param_config.items():
        if key in initial_params:
            param_names.append(key)
            param_initial.append(initial_params.get(key, initial))
            param_lower.append(lower if param_bounds is None else param_bounds.get(key, (lower, upper))[0])
            param_upper.append(upper if param_bounds is None else param_bounds.get(key, (lower, upper))[1])
    
    # Define objective function (chi-squared)
    def objective(params_array):
        # Convert array to dict
        params = dict(zip(param_names, params_array))
        
        # Calculate model photon flux (photons/cm²/s/keV)
        model_photon_flux = sm.combined_model(energy, params, model_components)
        
        # Fold model through ARF response to get predicted count rate (counts/s)
        if response is not None and response.arf is not None:
            # Predicted count rate = Model flux × Effective area × Energy width
            # Units: [photons/cm²/s/keV] × [cm²] × [keV] = [counts/s]
            model_rate = fold_model_through_response(model_photon_flux, response)
        else:
            # Fallback: if no response, use photon flux directly (less accurate)
            model_rate = model_photon_flux
        
        # Chi-squared calculation in count rate space
        # Both observed_rate and model_rate are in counts/s
        # observed_error is also in counts/s
        mask = observed_error > 0
        chi_sq = np.sum(((observed_rate[mask] - model_rate[mask]) / observed_error[mask]) ** 2)
        
        return chi_sq
    
    # Perform minimization
    bounds = list(zip(param_lower, param_upper))
    
    result = minimize(objective, 
                     param_initial,
                     method='L-BFGS-B',
                     bounds=bounds)
    
    # Get best-fit parameters
    best_params = dict(zip(param_names, result.x))
    
    # Calculate final chi-squared and reduced chi-squared
    chi_squared = result.fun
    n_params = len(param_names)
    n_data = np.sum(observed_error > 0)
    dof = n_data - n_params
    reduced_chi_squared = chi_squared / dof if dof > 0 else 0
    
    # Estimate parameter uncertainties (from Hessian approximation)
    # For more accurate errors, use MCMC or profile likelihood
    try:
        # Approximate covariance from Hessian
        from scipy.optimize import approx_fprime
        
        eps = np.sqrt(np.finfo(float).eps)
        hessian_diag = []
        
        for i, param in enumerate(result.x):
            def func_1d(p):
                params_test = result.x.copy()
                params_test[i] = p
                return objective(params_test)
            
            # Second derivative approximation
            grad = approx_fprime(np.array([param]), func_1d, eps)
            hessian_diag.append(max(abs(grad[0]), 1e-10))
        
        # Uncertainty = sqrt(2 / d²χ²/dp²) for Δχ² = 1
        uncertainties = np.sqrt(2.0 / np.array(hessian_diag))
        param_errors = dict(zip(param_names, uncertainties))
        
    except:
        # If error calculation fails, set to None
        param_errors = {key: None for key in param_names}
    
    # Create result dictionary
    fit_result = {
        'success': result.success,
        'message': result.message,
        'best_params': best_params,
        'param_errors': param_errors,
        'chi_squared': chi_squared,
        'dof': dof,
        'reduced_chi_squared': reduced_chi_squared,
        'n_data_points': n_data,
        'n_parameters': n_params
    }
    
    return fit_result


def calculate_model_spectrum(energy, params, model_components, response=None):
    """
    คำนวณ model spectrum ด้วยพารามิเตอร์ที่กำหนด
    
    IMPORTANT: ถ้ามี response data จะ fold model ผ่าน ARF response
    เพื่อให้ได้ predicted count rate ที่เทียบกับข้อมูลได้
    
    Parameters:
    -----------
    energy : array
        Energy array (keV)
    params : dict
        Model parameters
    model_components : list
        List of model components
    response : ResponseData, optional
        ARF response data for folding model
        
    Returns:
    --------
    model_rate : array
        Model count rate (counts/s/keV) if response provided,
        or photon flux (photons/cm²/s/keV) otherwise
    """
    # Calculate photon flux
    model_photon_flux = sm.combined_model(energy, params, model_components)
    
    # Fold through response
    # Fold through response
    if response is not None and response.arf is not None:
         # Use the standard folding function which now handles dE
         model_rate = fold_model_through_response(model_photon_flux, response)
         
         # Extra check: if fold_model didn't apply dE (because response object lacked headers), try approximation
         # This is a fallback for cases where dE wasn't applied inside fold_model
         # We check if values seem to be in density units (very high) vs rate units
         
         # Logic: if response.energy_hi was None inside fold_model, it wouldn't have multiplied.
         if response.energy_hi is None and len(energy) > 1:
              dE_approx = np.gradient(energy)
              model_rate *= dE_approx
              
         return model_rate
    
    return model_photon_flux


def calculate_residuals(observed_rate, model_rate, observed_error):
    """
    คำนวณ residuals
    
    Parameters:
    -----------
    observed_rate : array
        Observed count rate
    model_rate : array
        Model count rate
    observed_error : array
        Errors on observed rate
        
    Returns:
    --------
    residuals : array
        (Observed - Model) / Error
    """
    mask = observed_error > 0
    residuals = np.zeros_like(observed_rate)
    residuals[mask] = (observed_rate[mask] - model_rate[mask]) / observed_error[mask]
    return residuals


def goodness_of_fit_interpretation(reduced_chi_squared):
    """
    ให้คำแนะนำในการตีความ goodness of fit
    
    Parameters:
    -----------
    reduced_chi_squared : float
        Reduced chi-squared (χ²/dof)
        
    Returns:
    --------
    interpretation : str
        คำอธิบายผลการฟิต
    """
    if reduced_chi_squared < 0.5:
        return "❌ **Poor fit** - χ²/dof < 0.5: โมเดลอาจ overfit หรือ errors overestimated"
    elif 0.5 <= reduced_chi_squared < 0.9:
        return "⚠️ **Acceptable** - χ²/dof = 0.5-0.9: การฟิตค่อนข้างดี แต่อาจมี systematic uncertainties"
    elif 0.9 <= reduced_chi_squared <= 1.2:
        return "✅ **Good fit** - χ²/dof ≈ 1.0: การฟิตดีมาก โมเดลสอดคล้องกับข้อมูล"
    elif 1.2 < reduced_chi_squared <= 2.0:
        return "⚠️ **Marginal fit** - χ²/dof = 1.2-2.0: การฟิตพอใช้ได้ แต่อาจมี features ที่โมเดลจับไม่ได้"
    else:
        return "❌ **Bad fit** - χ²/dof > 2.0: โมเดลไม่เหมาะสม ควรพิจารณาเพิ่ม components หรือเปลี่ยนโมเดล"


def auto_estimate_parameters(energy, observed_rate, model_components):
    """
    ประมาณค่าพารามิเตอร์เริ่มต้นจากข้อมูลสเปกตรัม
    
    Parameters:
    -----------
    energy : array
        Energy array (keV)
    observed_rate : array
        Observed count rate (counts/s/keV)
    model_components : list
        List of model components to use
        
    Returns:
    --------
    estimated : dict
        ค่าประมาณสำหรับแต่ละพารามิเตอร์
    """
    estimated = {}
    
    # Filter valid data (positive values)
    valid = (observed_rate > 0) & (energy > 0)
    
    if np.sum(valid) < 10:
        # Not enough data, return defaults
        return {
            'pl_norm': 0.01,
            'photon_index': 2.0,
            'nH': 0.05,
            'refl_norm': 0.5,
            'line_energy': 6.4,
            'line_sigma': 0.1,
            'line_norm': 1.0,
            'bb_norm': 0.1,
            'kT': 0.5
        }
    
    energy_valid = energy[valid]
    rate_valid = observed_rate[valid]
    
    # Estimate normalization from mean rate
    # Typical X-ray normalization is mean_rate / (mean_energy^-2)
    mean_rate = np.mean(rate_valid)
    mean_energy = np.mean(energy_valid)
    estimated['pl_norm'] = mean_rate * (mean_energy ** 2) / 100.0
    
    # Clamp normalization to reasonable range
    estimated['pl_norm'] = max(0.001, min(10.0, estimated['pl_norm']))
    
    # Estimate photon index from log-log slope
    try:
        log_e = np.log10(energy_valid)
        log_r = np.log10(rate_valid)
        
        # Linear fit in log-log space: log(rate) = const - alpha * log(energy)
        # For power-law: rate ~ E^(-Gamma+1), so slope = 1 - Gamma
        coeffs = np.polyfit(log_e, log_r, 1)
        slope = coeffs[0]
        
        # Photon index: Gamma = 1 - slope
        estimated['photon_index'] = max(1.0, min(3.0, 1.0 - slope))
    except:
        estimated['photon_index'] = 2.0
    
    # Default values for other parameters
    # Absorption: typical Galactic nH
    estimated['nH'] = 0.05
    
    # Reflection: moderate contribution
    estimated['refl_norm'] = 0.5
    
    # Iron line at 6.4 keV
    estimated['line_energy'] = 6.4
    estimated['line_sigma'] = 0.1
    
    # Estimate line normalization from excess near 6.4 keV
    try:
        iron_mask = (energy_valid > 6.0) & (energy_valid < 7.0)
        if np.sum(iron_mask) > 0:
            iron_rate = np.mean(rate_valid[iron_mask])
            continuum_rate = np.mean(rate_valid[(energy_valid > 5.0) & (energy_valid < 6.0)])
            estimated['line_norm'] = max(0.1, (iron_rate - continuum_rate) * 10)
        else:
            estimated['line_norm'] = 1.0
    except:
        estimated['line_norm'] = 1.0
    
    # Blackbody parameters
    estimated['bb_norm'] = 0.1
    estimated['kT'] = 0.5
    
    return estimated


def get_job_hash(model_components, param_ranges, n_steps, energy_len, fixed_params=None):
    """
    สร้าง unique hash ID สำหรับงาน brute-force
    เพื่อใช้ในการแยก checkpoints
    """
    import hashlib
    import json
    
    # Create dictionary representing job configuration
    job_config = {
        'model_components': sorted(model_components),
        'param_ranges': {k: [float(v[0]), float(v[1])] for k, v in param_ranges.items()},
        'n_steps': int(n_steps),
        'energy_len': int(energy_len),
        'logic_version': 'v3_fixed_params', # Force new hash
        'fixed_params': {k: float(v) for k, v in fixed_params.items()} if fixed_params else {}
    }
    
    # Serialize to JSON string
    config_str = json.dumps(job_config, sort_keys=True)
    
    # Calculate SHA256 hash
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()


def load_checkpoint(job_hash, checkpoint_dir='data/checkpoints'):
    """
    Load checkpoint data if exists
    
    Parameters:
    -----------
    job_hash : str
        Unique job hash from get_job_hash()
    checkpoint_dir : str
        Directory to load checkpoint from
        
    Returns:
    --------
    checkpoint : dict or None
        Checkpoint data with keys:
        - completed_parts: list of completed part indices
        - best_result_so_far: dict with best_chi2_dof, best_params, best_result
        - last_updated: timestamp string
    """
    import json
    import os
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{job_hash}.json")
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    return None


def save_checkpoint(job_hash, completed_parts, best_result_so_far, checkpoint_dir='data/checkpoints'):
    """
    Save checkpoint after processing a part
    
    Parameters:
    -----------
    job_hash : str
        Unique job hash from get_job_hash()
    completed_parts : list
        List of completed part indices
    best_result_so_far : dict
        Dict with best_chi2_dof, best_params, best_result
    checkpoint_dir : str
        Directory to save checkpoint to
    """
    import json
    import os
    from datetime import datetime
    
    # Create directory if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{job_hash}.json")
    
    checkpoint_data = {
        'job_hash': job_hash,
        'completed_parts': sorted(completed_parts),
        'best_result_so_far': best_result_so_far,
        'last_updated': str(datetime.now())
    }
    
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


def prepare_effective_arf(response, energy):
    """
    Helper to calculate Effective ARF (ARF * dE)
    Pre-calculating this drastically improves performance and ensures unit consistency
    
    Returns:
    --------
    effective_arf : array or None
        Combined ARF * dE
    """
    if response is None or response.arf is None:
        return None
        
    arf = np.array(response.arf)
    
    # Calculate dE
    dE = None
    if response.energy_hi is not None and response.energy_lo is not None:
         if len(response.energy_hi) == len(arf):
             dE = response.energy_hi - response.energy_lo
    
    if dE is None and len(energy) > 1:
        # Fallback approximation
        dE = np.gradient(energy)
        
    if dE is not None:
        # Ensure dE matches arf length (if approximation was used on different grid)
        if len(dE) == len(arf):
            return arf * dE
            
    return arf # Return original if dE calc failed (better than nothing)



def brute_force_fit(energy, observed_rate, observed_error, 
                    model_components, param_ranges, 
                    n_steps=5, exposure=1.0, response=None, fixed_params=None):
    """
    Sequential Brute-force grid search
    """
    import itertools
    
    # Build parameter grids
    param_names = list(param_ranges.keys())
    param_grids = []
    
    for param_name in param_names:
        min_val, max_val = param_ranges[param_name]
        param_grids.append(np.linspace(min_val, max_val, n_steps))
    
    # Calculate total combinations
    total_combinations = 1
    for grid in param_grids:
        total_combinations *= len(grid)
        
    # Prepare Effective ARF
    effective_arf = prepare_effective_arf(response, energy)
        
    # Track best result
    best_chi2_dof = float('inf')
    best_params = None
    best_result = None
    
    count = 0
    
    # Generate all combinations
    all_combinations = itertools.product(*param_grids)
    
    for combo in all_combinations:
        count += 1
        current_params = dict(zip(param_names, combo))
        
        # Merge with fixed parameters
        if fixed_params:
            current_params.update(fixed_params)

        
        # Calculate model
        model_photon_flux = sm.combined_model(energy, current_params, model_components)
        
        # Fold through response
        if effective_arf is not None:
            model_rate = model_photon_flux * effective_arf
        else:
            model_rate = model_photon_flux
            
        # Chi-squared
        mask = observed_error > 0
        chi_sq = np.sum(((observed_rate[mask] - model_rate[mask]) / observed_error[mask]) ** 2)
        n_data = np.sum(mask)
        dof = n_data - len(param_names)
        chi2_dof = chi_sq / dof if dof > 0 else float('inf')
        
        # Check if best
        is_best = False
        if chi2_dof < best_chi2_dof:
            best_chi2_dof = chi2_dof
            best_params = current_params
            best_result = {
                'chi_squared': chi_sq,
                'dof': dof,
                'reduced_chi_squared': chi2_dof,
                'n_data_points': n_data
            }
            is_best = True
        
        # Yield progress
        # Update UI every 100 items or if found new best
        if count % 100 == 0 or is_best or count == total_combinations:
            yield {
                'iteration': count,
                'total': total_combinations,
                'progress': count / total_combinations,
                'current_params': current_params,
                'current_chi2_dof': chi2_dof,
                'is_best': is_best,
                'best_params': best_params,
                'best_chi2_dof': best_chi2_dof,
                'best_result': best_result,
                'status': 'running',
                'description': f"Processing {count}/{total_combinations}"
            }
            
    # Final result
    yield {
        'iteration': count,
        'total': total_combinations,
        'progress': 1.0,
        'current_params': best_params,
        'current_chi2_dof': best_chi2_dof,
        'is_best': True,
        'best_params': best_params,
        'best_chi2_dof': best_chi2_dof,
        'best_result': best_result,
        'status': 'complete',
        'description': f"เสร็จสิ้น! พบค่าที่ดีที่สุด χ²/dof = {best_chi2_dof:.3f}"
    }



def brute_force_fit_gpu(energy, observed_rate, observed_error, 
                        model_components, param_ranges, 
                        n_steps=5, exposure=1.0, response=None, batch_size=50000,
                        n_parts=100, skip_parts=None, checkpoint_dir='./checkpoints', fixed_params=None):
    """
    GPU Accelerated Brute-force grid search with partition and checkpoint support
    
    Parameters:
    -----------
    energy : array
        Energy array (keV)
    observed_rate : array
        Observed count rate
    observed_error : array
        Rate errors
    model_components : list
        Model components to use
    param_ranges : dict
        Parameter ranges {name: (min, max)}
    n_steps : int
        Grid steps per parameter
    exposure : float
        Exposure time
    response : ResponseData
        ARF response
    batch_size : int
        GPU batch size (default reduced to 50,000 to prevent TDR)
    n_parts : int
        Number of parts to split the search into (default 100)
    skip_parts : list
        List of part indices to skip (from checkpoint)
    checkpoint_dir : str
        Directory for checkpoint files
        
    Yields:
    -------
    dict with keys:
        - iteration, total, progress
        - compute_device: "GPU (CUDA)"
        - batch_compute_time_ms: compute time per batch
        - items_per_second: throughput
        - memory_used_mb: GPU memory usage
        - part_idx, n_parts: partition info
    """
    if not HAS_GPU:
        raise RuntimeError("CuPy not installed or GPU not detected")
        
    # Validation: Try to use GPU memory before starting
    try:
        _ = cp.array([1.0])
    except Exception as e:
        raise RuntimeError(f"GPU detected but not functional (missing DLLs?): {e}")
        
    import itertools
    import math
    
    # Get job hash for checkpoint
    energy_len = len(energy) if hasattr(energy, '__len__') else energy.shape[0]
    job_hash = get_job_hash(model_components, param_ranges, n_steps, energy_len, fixed_params)
    
    # Load existing checkpoint if skip_parts not provided
    if skip_parts is None:
        checkpoint = load_checkpoint(job_hash, checkpoint_dir)
        if checkpoint:
            skip_parts = checkpoint.get('completed_parts', [])
            # Also restore best result if available
            best_from_checkpoint = checkpoint.get('best_result_so_far', {})
        else:
            skip_parts = []
            best_from_checkpoint = {}
    else:
        best_from_checkpoint = {}
    
    # helper for moving to gpu
    def to_gpu(arr):
        return cp.asarray(arr, dtype=cp.float32)

    energy_gpu = to_gpu(energy)
    obs_rate_gpu = to_gpu(observed_rate)
    obs_err_gpu = to_gpu(observed_error)
    
    # Pre-calculate Effective ARF (ARF * dE)
    effective_arf_gpu = None
    
    if response is not None and response.arf is not None:
        eff_arf = prepare_effective_arf(response, energy)
        if eff_arf is not None:
            effective_arf_gpu = to_gpu(eff_arf)
        
    mask_gpu = obs_err_gpu > 0
    n_data = float(cp.sum(mask_gpu))
    
    # Prepare parameter grids
    n_data = float(cp.sum(mask_gpu))
    
    # Prepare parameter grids
    param_names = list(param_ranges.keys())
    param_grids = []
    
    for param_name in param_names:
        min_val, max_val = param_ranges[param_name]
        param_grids.append(np.linspace(min_val, max_val, n_steps))
        
    # Calculate total combinations
    total_combinations = n_steps ** len(param_names)
    
    # Calculate part size
    part_size = math.ceil(total_combinations / n_parts)
    
    # Initialize best result (from checkpoint or fresh)
    if best_from_checkpoint:
        best_chi2_dof = best_from_checkpoint.get('best_chi2_dof', float('inf'))
        best_params = best_from_checkpoint.get('best_params', None)
        best_result = best_from_checkpoint.get('best_result', None)
    else:
        best_chi2_dof = float('inf')
        best_params = None
        best_result = None
    
    # Track completed parts (copy to avoid mutating input)
    completed_parts = list(skip_parts)
    
    # Track total processed (including skipped)
    total_processed = len(skip_parts) * part_size
    
    # Last save time for throttling intermediate saves
    last_save_time = time.time()
    
    # Initial yield
    yield {
        'iteration': total_processed,
        'total': total_combinations,
        'progress': min(total_processed / total_combinations, 1.0),
        'current_params': best_params,
        'current_chi2_dof': best_chi2_dof,
        'is_best': False,
        'best_params': best_params,
        'best_chi2_dof': best_chi2_dof,
        'best_result': best_result,
        'status': 'starting',
        'description': f"🚀 GPU เริ่มต้น {n_parts} parts (ข้าม {len(skip_parts)} parts จาก checkpoint)...",
        'compute_device': 'GPU (CUDA)',
        'part_idx': 0,
        'n_parts': n_parts,
        'items_per_second': 0,
        'memory_used_mb': 0,
        'batch_compute_time_ms': 0,
        'job_hash': job_hash
    }
    
    # Generate all combinations iterator
    combination_generator = itertools.product(*param_grids)
    
    # Process by parts
    for part_idx in range(n_parts):
        # Calculate range for this part
        start_idx = part_idx * part_size
        end_idx = min(start_idx + part_size, total_combinations)
        current_part_size = end_idx - start_idx
        
        if current_part_size <= 0:
            break
            
        # Check if we should skip this part
        if part_idx in skip_parts:
            # Advance generator without processing
            list(itertools.islice(combination_generator, current_part_size))
            
            yield {
                'iteration': total_processed,
                'total': total_combinations,
                'progress': min(total_processed / total_combinations, 1.0),
                'current_params': best_params,
                'current_chi2_dof': best_chi2_dof,
                'is_best': False,
                'best_params': best_params,
                'best_chi2_dof': best_chi2_dof,
                'best_result': best_result,
                'status': 'running',
                'description': f"⏭️ ข้าม Part {part_idx+1}/{n_parts} (เสร็จแล้วจาก checkpoint)",
                'compute_device': 'GPU (CUDA) - Skipped',
                'part_idx': part_idx,
                'n_parts': n_parts,
                'skipped': True,
                'items_per_second': 0,
                'memory_used_mb': 0,
                'batch_compute_time_ms': 0,
                'job_hash': job_hash
            }
            continue
        
        # Process this part in batches
        part_start_time = time.time()
        part_processed = 0
        part_items_per_sec = 0
        
        while part_processed < current_part_size:
            current_batch_size = min(batch_size, current_part_size - part_processed)
            
            # Take next batch from generator
            batch = list(itertools.islice(combination_generator, current_batch_size))
            if not batch:
                break
            
            batch_start_time = time.time()
            
            # Process batch on GPU
            batch_array = np.array(batch, dtype=np.float32)
            batch_gpu = cp.asarray(batch_array)
            
            # Dictionary of arrays for model: {param_name: array(Batch, 1)}
            params_gpu = {}
            for idx, name in enumerate(param_names):
                params_gpu[name] = batch_gpu[:, idx].reshape(-1, 1)
            
            # Add fixed parameters
            if fixed_params:
                for key, val in fixed_params.items():
                    # Broadcast fixed scalar to (Batch, 1) or let combined_model handle scalar
                    # combined_model expects (Batch, 1) for batched params to trigger batch mode
                    # But if we mix (Batch, 1) and Scalar, it usually works in cupy/numpy
                    # However, strictly for safety, let's keep it as scalar if possible or broadcast
                    params_gpu[key] = val

                
            # Calculate Model on GPU (Vectorized)
            model_photon_flux = sm.combined_model(energy_gpu, params_gpu, model_components)
            
            # Fold through response
            # Note: effective_arf_gpu already includes dE
            model_rate = model_photon_flux # Start with flux
            
            if effective_arf_gpu is not None:
                model_rate = model_photon_flux * effective_arf_gpu
            
            # Calculate Chi-Squared
            diff = (obs_rate_gpu - model_rate)
            diff_masked = diff * mask_gpu
            err_masked = obs_err_gpu
            chi2_contrib = (diff_masked ** 2) / (err_masked ** 2 + 1e-20)
            chi2_contrib = chi2_contrib * mask_gpu
            
            chi_sq_batch = cp.sum(chi2_contrib, axis=1)
            
            # Synchronize GPU for accurate timing and safety
            cp.cuda.Stream.null.synchronize()
            batch_end_time = time.time()
            batch_compute_time_ms = (batch_end_time - batch_start_time) * 1000
            
            # Find min in batch
            min_idx_batch = cp.argmin(chi_sq_batch)
            min_chi2_batch = float(chi_sq_batch[min_idx_batch])
            
            dof = n_data - len(param_names)
            min_chi2_dof_batch = min_chi2_batch / dof if dof > 0 else float('inf')
            
            is_new_best = False
            if min_chi2_dof_batch < best_chi2_dof:
                best_chi2_dof = min_chi2_dof_batch
                best_param_values = batch[int(min_idx_batch)]
                best_params = dict(zip(param_names, best_param_values))
                if fixed_params:
                    best_params.update(fixed_params)
                
                best_result = {
                    'chi_squared': min_chi2_batch,
                    'dof': dof,
                    'reduced_chi_squared': min_chi2_dof_batch,
                    'n_data_points': int(n_data)
                }
                best_result = {
                    'chi_squared': min_chi2_batch,
                    'dof': dof,
                    'reduced_chi_squared': min_chi2_dof_batch,
                    'n_data_points': int(n_data)
                }
                is_new_best = True
                

                
                # Checkpoint immediately if it's a significant improvement or sufficient time passed
                # To protect against crashing after finding a good value but before part ends
                current_time = time.time()
                if current_time - last_save_time > 10.0: # Minimum 10s between intermediate saves
                    best_result_so_far = {
                        'best_chi2_dof': best_chi2_dof,
                        'best_params': best_params,
                        'best_result': best_result
                    }
                    # Note: We don't append current part to completed_parts yet
                    save_checkpoint(job_hash, completed_parts, best_result_so_far, checkpoint_dir)
                    last_save_time = current_time
            
            # Capture batch best for session tracking
            batch_best_params_values = batch[int(min_idx_batch)]
            batch_best_params = dict(zip(param_names, batch_best_params_values))
            if fixed_params:
                 batch_best_params.update(fixed_params)
            batch_best_chi2_dof = min_chi2_dof_batch

            part_processed += len(batch)
            total_processed += len(batch)
            
            # Calculate throughput
            elapsed_part = time.time() - part_start_time
            if elapsed_part > 0:
                part_items_per_sec = part_processed / elapsed_part
            
            # Get GPU memory usage
            try:
                mempool = cp.get_default_memory_pool()
                memory_used_mb = mempool.used_bytes() / (1024 * 1024)
            except:
                memory_used_mb = 0
            
            # Capture latest values for reporting before deleting GPU arrays
            latest_chi2_val = float(chi_sq_batch[-1])
            latest_params_val = dict(zip(param_names, batch[-1]))
            if fixed_params:
                latest_params_val.update(fixed_params)

            # Free up memory
            del batch_gpu, params_gpu, model_photon_flux, model_rate, chi_sq_batch
            cp.get_default_memory_pool().free_all_blocks()
            
            # Yield progress with detailed info
            yield {
                'iteration': total_processed,
                'total': total_combinations,
                'progress': total_processed / total_combinations,
                'current_params': latest_params_val,
                'current_chi2_dof': latest_chi2_val / dof if dof > 0 else float('inf'),
                'batch_best_params': batch_best_params,
                'batch_best_chi2_dof': batch_best_chi2_dof,
                'is_best': is_new_best,
                'best_params': best_params,
                'best_chi2_dof': best_chi2_dof,
                'best_result': best_result,
                'status': 'running',
                'description': f"🖥️ GPU Part {part_idx+1}/{n_parts}: {part_processed:,}/{current_part_size:,} | {part_items_per_sec:,.0f} items/s",
                'compute_device': 'GPU (CUDA)',
                'part_idx': part_idx,
                'n_parts': n_parts,
                'items_per_second': part_items_per_sec,
                'memory_used_mb': memory_used_mb,
                'batch_compute_time_ms': batch_compute_time_ms,
                'batch_size': len(batch),
                'job_hash': job_hash
            }
        
        # Part completed - save checkpoint
        completed_parts.append(part_idx)
        best_result_so_far = {
            'best_chi2_dof': best_chi2_dof,
            'best_params': best_params,
            'best_result': best_result
        }
        save_checkpoint(job_hash, completed_parts, best_result_so_far, checkpoint_dir)
        last_save_time = time.time()
        
        # Yield part completion
        yield {
            'iteration': total_processed,
            'total': total_combinations,
            'progress': total_processed / total_combinations,
            'current_params': best_params,
            'current_chi2_dof': best_chi2_dof,
            'is_best': False,
            'best_params': best_params,
            'best_chi2_dof': best_chi2_dof,
            'best_result': best_result,
            'status': 'part_complete',
            'description': f"✅ GPU Part {part_idx+1}/{n_parts} เสร็จสิ้น (บันทึก checkpoint แล้ว)",
            'compute_device': 'GPU (CUDA)',
            'part_idx': part_idx,
            'part_index': part_idx, # Add for compatibility with app.py
            'n_parts': n_parts,
            'part_complete': True,
            'items_per_second': part_items_per_sec,
            'memory_used_mb': 0,
            'batch_compute_time_ms': 0,
            'job_hash': job_hash
        }

    # Final result
    yield {
        'iteration': total_combinations,
        'total': total_combinations,
        'progress': 1.0,
        'current_params': best_params,
        'current_chi2_dof': best_chi2_dof,
        'is_best': True,
        'best_params': best_params,
        'best_chi2_dof': best_chi2_dof,
        'best_result': best_result,
        'status': 'complete',
        'description': f"🎉 GPU เสร็จสมบูรณ์! Best χ²/dof = {best_chi2_dof:.3f}",
        'compute_device': 'GPU (CUDA)',
        'part_idx': n_parts - 1,
        'n_parts': n_parts,
        'items_per_second': 0,
        'memory_used_mb': 0,
        'batch_compute_time_ms': 0,
        'job_hash': job_hash
    }


def check_gpu():
    """Check GPU availability and info"""
    info = {"available": False, "device": "None", "memory": "0 MB"}
    if HAS_GPU:
        try:
            # Try to force simple CUDA call to trigger latent DLL errors
            dev = cp.cuda.Device(0)
            dev.compute_capability # Trigger access
            
            info["available"] = True
            info["device"] = f"GPU {dev.id}"
            # Get simplified name
            try:
                # Attempt to get name if possible
                info["device_name"] = "NVIDIA GPU (CUDA)" 
            except:
                pass
                
            mem_info = dev.mem_info
            info["memory_free"] = f"{mem_info[0] / 1024**2:.0f} MB"
            info["memory_total"] = f"{mem_info[1] / 1024**2:.0f} MB"
            
        except Exception as e:
            err_msg = str(e)
            if "nvrtc" in err_msg.lower() or "dll" in err_msg.lower():
                info["error"] = "CuPy installed but CUDA/DLLs missing. Check NVIDIA Drivers."
            else:
                info["error"] = err_msg
            info["available"] = False
    return info


def benchmark_gpu(n_points=1000):
    """
    Benchmark CPU vs GPU for a simple calculation
    """
    results = {}
    
    # Create random data
    energy = np.linspace(0.1, 10, n_points)
    
    # 1. CPU Benchmark
    start_cpu = time.time()
    # Perform intensive calculation (e.g., 1000 powerlaws)
    for _ in range(100):
        _ = sm.powerlaw(energy, 1.0, 2.0)
    end_cpu = time.time()
    results['cpu_time'] = end_cpu - start_cpu
    
    if HAS_GPU:
        try:
            # 2. GPU Benchmark
            # Warmup
            energy_gpu = cp.asarray(energy)
            _ = sm.powerlaw(energy_gpu, 1.0, 2.0)
            cp.cuda.Stream.null.synchronize()
            
            start_gpu = time.time()
            for _ in range(100):
                _ = sm.powerlaw(energy_gpu, 1.0, 2.0)
            cp.cuda.Stream.null.synchronize()
            end_gpu = time.time()
            results['gpu_time'] = end_gpu - start_gpu
            results['speedup'] = results['cpu_time'] / results['gpu_time'] if results['gpu_time'] > 0 else 0
        except Exception as e:
            err_msg = str(e)
            if "nvrtc" in err_msg.lower():
                results['error'] = "Missing 'nvrtc' DLL. Check CUDA installation."
            else:
                results['error'] = err_msg
            
    return results


def _compute_chi2_for_params(args):
    """
    Helper function สำหรับคำนวณ chi-squared ของ parameter set เดียว
    ใช้สำหรับ multiprocessing (ต้องเป็น top-level function)
    """
    (param_values, param_names, energy, observed_rate, observed_error, 
     model_components, effective_arf, fixed_params) = args
    
    try:
        current_params = dict(zip(param_names, param_values))
        if fixed_params:
            current_params.update(fixed_params)
        
        # Calculate model
        model_photon_flux = sm.combined_model(energy, current_params, model_components)
        
        # Fold through response (effective_arf includes dE)
        if effective_arf is not None:
            model_rate = model_photon_flux * effective_arf
        else:
            model_rate = model_photon_flux
        
        # Calculate chi-squared
        mask = observed_error > 0
        
        # dynamic masking based on energy parameters if present
        if 'energy_min' in current_params or 'energy_max' in current_params:
             e_min = current_params.get('energy_min', -float('inf'))
             e_max = current_params.get('energy_max', float('inf'))
             energy_mask = (energy >= e_min) & (energy <= e_max)
             mask = mask & energy_mask

        chi_sq = np.sum(((observed_rate[mask] - model_rate[mask]) / observed_error[mask]) ** 2)
        n_data = np.sum(mask)
        dof = n_data - (len(param_names) - (2 if ('energy_min' in param_names or 'energy_max' in param_names) else 0)) 
        # Note: We subtract params (like energy_min) from DOF count? Argumentative, but usually valid.
        
        chi2_dof = chi_sq / dof if dof > 0 else float('inf')
        
        return {
            'params': current_params,
            'chi2_dof': chi2_dof,
            'chi_squared': chi_sq,
            'dof': dof,
            'n_data_points': n_data,
            'success': True
        }
    except Exception as e:
        return {
            'params': dict(zip(param_names, param_values)),
            'chi2_dof': float('inf'),
            'success': False,
            'error': str(e)
        }


def brute_force_fit_parallel(energy, observed_rate, observed_error, 
                             model_components, param_ranges, 
                             n_steps=5, n_workers=None, batch_size=100,
                             exposure=1.0, response=None, backend='threading',
                             n_parts=100, skip_parts=None, checkpoint_dir='./checkpoints', fixed_params=None):
    """
    Parallel Brute-force grid search with partitioning and checkpoint support
    
    Parameters:
    -----------
    n_parts : int
        Number of parts to split the search into (default 100)
    skip_parts : list
        List of part indices (0-based) to skip
    checkpoint_dir : str
        Directory for checkpoint files
    """
    import itertools
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import multiprocessing
    import os
    import math
    
    # Select executor class
    Executor = ThreadPoolExecutor if backend == 'threading' else ProcessPoolExecutor
    
    # Determine number of workers
    if n_workers is None:
        if backend == 'multiprocessing':
            n_workers = min(4, max(1, multiprocessing.cpu_count() - 1))
        else:
            n_workers = max(2, multiprocessing.cpu_count())
    
    # Get job hash for checkpoint
    energy_len = len(energy) if hasattr(energy, '__len__') else len(np.array(energy))
    job_hash = get_job_hash(model_components, param_ranges, n_steps, energy_len, fixed_params)
    
    # Load existing checkpoint if skip_parts not provided
    if skip_parts is None:
        checkpoint = load_checkpoint(job_hash, checkpoint_dir)
        if checkpoint:
            skip_parts = checkpoint.get('completed_parts', [])
            best_from_checkpoint = checkpoint.get('best_result_so_far', {})
        else:
            skip_parts = []
            best_from_checkpoint = {}
    else:
        best_from_checkpoint = {}
    
    # Build parameter grids
    param_names = list(param_ranges.keys())
    param_grids = []
    
    for param_name in param_names:
        min_val, max_val = param_ranges[param_name]
        param_grids.append(np.linspace(min_val, max_val, n_steps))
    
    # Calculate total combinations
    total_combinations = 1
    for grid in param_grids:
        total_combinations *= len(grid)
    
    # Calculate part size
    part_size = math.ceil(total_combinations / n_parts)
    
    # Prepare ARF values (Effective ARF)
    effective_arf = prepare_effective_arf(response, energy)
    
    # Convert arrays
    energy_arr = np.array(energy)
    rate_arr = np.array(observed_rate)
    error_arr = np.array(observed_error)
    
    # Track best result (from checkpoint or fresh)
    if best_from_checkpoint:
        best_chi2_dof = best_from_checkpoint.get('best_chi2_dof', float('inf'))
        best_params = best_from_checkpoint.get('best_params', None)
        best_result = best_from_checkpoint.get('best_result', None)
    else:
        best_chi2_dof = float('inf')
        best_params = None
        best_result = None
    
    # Track completed parts
    completed_parts = list(skip_parts)
    processed = len(skip_parts) * part_size
    
    # Initial yield
    yield {
        'iteration': processed,
        'total': total_combinations,
        'progress': min(processed / total_combinations, 1.0),
        'current_params': best_params,
        'current_chi2_dof': best_chi2_dof,
        'is_best': False,
        'best_params': best_params,
        'best_chi2_dof': best_chi2_dof,
        'best_result': best_result,
        'status': 'starting',
        'description': f"🚀 CPU เริ่มต้น ({backend}) {n_parts} parts (ข้าม {len(skip_parts)} parts จาก checkpoint)...",
        'compute_device': f'CPU ({backend.title()})',
        'n_workers': n_workers,
        'n_parts': n_parts,
        'part_size': part_size,
        'items_per_second': 0,
        'batch_compute_time_ms': 0,
        'job_hash': job_hash
    }
    
    # Process by parts
    combination_generator = itertools.product(*param_grids)
    
    for part_idx in range(n_parts):
        # Calculate range for this part
        start_idx = part_idx * part_size
        end_idx = min(start_idx + part_size, total_combinations)
        current_part_size = end_idx - start_idx
        
        if current_part_size <= 0:
            break
            
        # Check if we should skip this part
        if part_idx in skip_parts:
            # Advance generator without processing
            # Using islice to consume items efficiently
            # We need to consume 'current_part_size' items
            list(itertools.islice(combination_generator, current_part_size))
            
            yield {
                'iteration': processed,
                'total': total_combinations,
                'progress': min(processed / total_combinations, 1.0),
                'current_params': best_params,
                'current_chi2_dof': best_chi2_dof,
                'is_best': False,
                'best_params': best_params,
                'best_chi2_dof': best_chi2_dof,
                'best_result': best_result,
                'status': 'running',
                'description': f"⏭️ ข้าม Part {part_idx+1}/{n_parts} (เสร็จแล้วจาก checkpoint)",
                'compute_device': f'CPU ({backend.title()}) - Skipped',
                'part_idx': part_idx,
                'n_workers': n_workers,
                'n_parts': n_parts,
                'batch_processed': 0,
                'skipped': True,
                'items_per_second': 0,
                'batch_compute_time_ms': 0,
                'job_hash': job_hash
            }
            continue

        # Process batches within this part
        part_processed = 0
        part_start_time = time.time()
        part_items_per_sec = 0
        
        while part_processed < current_part_size:
            current_batch_size = min(batch_size, current_part_size - part_processed)
            
            # Take next batch from generator
            batch = list(itertools.islice(combination_generator, current_batch_size))
            if not batch:
                break
            
            # Prepare arguments
            batch_args = [
                (combo, param_names, energy_arr, rate_arr, error_arr, 
                 model_components, effective_arf, fixed_params)
                for combo in batch
            ]
            
            batch_start_time = time.time()
            
            # Process batch
            try:
                with Executor(max_workers=n_workers) as executor:
                    results = list(executor.map(_compute_chi2_for_params, batch_args))
                
                # Process results - Find Batch Best
                batch_best_chi2_dof = float('inf')
                batch_best_params = None
                
                for result in results:
                    if result['success']:
                         # Update local batch best
                         if result['chi2_dof'] < batch_best_chi2_dof:
                             batch_best_chi2_dof = result['chi2_dof']
                             batch_best_params = result['params']
                             
                         # Update global best
                         if result['chi2_dof'] < best_chi2_dof:
                            best_chi2_dof = result['chi2_dof']
                            best_params = result['params']
                            best_result = {
                                'chi_squared': result['chi_squared'],
                                'dof': result['dof'],
                                'reduced_chi_squared': result['chi2_dof'],
                                'n_data_points': result['n_data_points']
                            }
                
                part_processed += len(batch)
                processed += len(batch)
                
            except Exception as e:
                # Fallback to sequential
                for args in batch_args:
                    result = _compute_chi2_for_params(args)
                    if result['success'] and result['chi2_dof'] < best_chi2_dof:
                        best_chi2_dof = result['chi2_dof']
                        best_params = result['params']
                        best_result = {
                            'chi_squared': result['chi_squared'],
                            'dof': result['dof'],
                            'reduced_chi_squared': result['chi2_dof'],
                            'n_data_points': result['n_data_points']
                        }
                part_processed += len(batch)
                processed += len(batch)
            
            # Calculate timing
            batch_end_time = time.time()
            batch_compute_time_ms = (batch_end_time - batch_start_time) * 1000
            elapsed_part = time.time() - part_start_time
            if elapsed_part > 0:
                part_items_per_sec = part_processed / elapsed_part
            
            # Yield progress
            yield {
                'iteration': processed,
                'total': total_combinations,
                'progress': processed / total_combinations,
                'current_params': results[-1]['params'] if results else (dict(zip(param_names, batch[-1])) if batch else None),
                'current_chi2_dof': results[-1]['chi2_dof'] if results else float('inf'),
                'batch_best_params': batch_best_params,
                'batch_best_chi2_dof': batch_best_chi2_dof,
                'is_best': True,
                'best_params': best_params,
                'best_chi2_dof': best_chi2_dof,
                'best_result': best_result,
                'status': 'running',
                'description': f"💻 CPU Part {part_idx+1}/{n_parts}: {part_processed:,}/{current_part_size:,} | {part_items_per_sec:,.0f} items/s",
                'compute_device': f'CPU ({backend.title()})',
                'part_idx': part_idx,
                'n_workers': n_workers,
                'n_parts': n_parts,
                'batch_processed': len(batch),
                'items_per_second': part_items_per_sec,
                'batch_compute_time_ms': batch_compute_time_ms,
                'job_hash': job_hash
            }
        
        # Part completed - save checkpoint
        completed_parts.append(part_idx)
        best_result_so_far = {
            'best_chi2_dof': best_chi2_dof,
            'best_params': best_params,
            'best_result': best_result
        }
        save_checkpoint(job_hash, completed_parts, best_result_so_far, checkpoint_dir)
        
        # Yield part completion
        yield {
            'iteration': processed,
            'total': total_combinations,
            'progress': processed / total_combinations,
            'current_params': best_params,
            'current_chi2_dof': best_chi2_dof,
            'is_best': False,
            'best_params': best_params,
            'best_chi2_dof': best_chi2_dof,
            'best_result': best_result,
            'status': 'part_complete',
            'part_idx': part_idx,
            'description': f"✅ CPU Part {part_idx+1}/{n_parts} เสร็จสิ้น (บันทึก checkpoint แล้ว)",
            'compute_device': f'CPU ({backend.title()})',
            'n_workers': n_workers,
            'n_parts': n_parts,
            'part_complete': True,
            'items_per_second': part_items_per_sec,
            'batch_compute_time_ms': 0,
            'job_hash': job_hash
        }
    
    # Final result
    yield {
        'iteration': total_combinations,
        'total': total_combinations,
        'progress': 1.0,
        'current_params': best_params,
        'current_chi2_dof': best_chi2_dof,
        'is_best': True,
        'best_params': best_params,
        'best_chi2_dof': best_chi2_dof,
        'best_result': best_result,
        'status': 'complete',
        'description': f"🎉 CPU เสร็จสมบูรณ์! Best χ²/dof = {best_chi2_dof:.3f}",
        'compute_device': f'CPU ({backend.title()})',
        'part_idx': n_parts - 1,
        'n_workers': n_workers,
        'n_parts': n_parts,
        'items_per_second': 0,
        'batch_compute_time_ms': 0,
        'job_hash': job_hash
    }
