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
        # This gives count rate density that can be directly compared with observed rate
        predicted_rate = model_flux * response.arf
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
        
        # Fold model through ARF response to get predicted count rate (counts/s/keV)
        if response is not None and response.arf is not None:
            # Predicted count rate = Model flux × Effective area × Energy width
            # This converts photon flux (photons/cm²/s/keV) to count rate (counts/s/keV)
            # Units: [photons/cm²/s/keV] × [cm²] × [keV] = [counts/s]
            model_rate = fold_model_through_response(model_photon_flux, response)
        else:
            # Fallback: if no response, use photon flux directly (less accurate)
            model_rate = model_photon_flux
        
        # Chi-squared calculation in count rate space
        # Both observed_rate and model_rate are in counts/s/keV
        # observed_error is also in counts/s/keV
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
    
    # Fold through response if provided
    if response is not None and response.arf is not None:
        model_rate = fold_model_through_response(model_photon_flux, response)
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


def get_job_hash(model_components, param_ranges, n_steps, energy_len):
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
        'energy_len': int(energy_len)
    }
    
    # Serialize to JSON string
    config_str = json.dumps(job_config, sort_keys=True)
    
    # Calculate SHA256 hash
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()


def brute_force_fit(energy, observed_rate, observed_error, 
                    model_components, param_ranges, 
                    n_steps=5, exposure=1.0, response=None):
    """
    Brute-force grid search เพื่อหาค่าพารามิเตอร์ที่ดีที่สุด
    
    ฟังก์ชันนี้เป็น generator ที่ yield ผลลัพธ์แต่ละ iteration
    เพื่อให้ UI สามารถแสดง progress ได้
    
    Parameters:
    -----------
    energy : array
        Energy array (keV)
    observed_rate : array
        Observed count rate
    observed_error : array
        Errors on count rate
    model_components : list
        List of model components
    param_ranges : dict
        Parameter ranges: {param_name: (min, max)}
    n_steps : int
        Number of steps per parameter (total combinations = n_steps^n_params)
    exposure : float
        Exposure time
    response : ResponseData
        ARF response data
        
    Yields:
    -------
    dict: Progress info with current params, chi2, best so far, iteration count
    """
    import itertools
    
    # Build parameter grids
    param_names = list(param_ranges.keys())
    param_grids = []
    
    for param_name in param_names:
        min_val, max_val = param_ranges[param_name]
        param_grids.append(np.linspace(min_val, max_val, n_steps))
    
    # Calculate total combinations
    total_combinations = n_steps ** len(param_names)
    
    # Track best result
    best_chi2_dof = float('inf')
    best_params = None
    best_result = None
    
    # Iterate through all combinations
    for i, param_values in enumerate(itertools.product(*param_grids)):
        current_params = dict(zip(param_names, param_values))
        
        try:
            # Calculate model
            model_photon_flux = sm.combined_model(energy, current_params, model_components)
            
            # Fold through response
            if response is not None and response.arf is not None:
                model_rate = fold_model_through_response(model_photon_flux, response)
            else:
                model_rate = model_photon_flux
            
            # Calculate chi-squared
            mask = observed_error > 0
            chi_sq = np.sum(((observed_rate[mask] - model_rate[mask]) / observed_error[mask]) ** 2)
            n_data = np.sum(mask)
            dof = n_data - len(param_names)
            chi2_dof = chi_sq / dof if dof > 0 else float('inf')
            
            # Check if this is the best so far
            is_best = chi2_dof < best_chi2_dof
            if is_best:
                best_chi2_dof = chi2_dof
                best_params = current_params.copy()
                best_result = {
                    'chi_squared': chi_sq,
                    'dof': dof,
                    'reduced_chi_squared': chi2_dof,
                    'n_data_points': n_data
                }
            
            # Yield progress update
            yield {
                'iteration': i + 1,
                'total': total_combinations,
                'progress': (i + 1) / total_combinations,
                'current_params': current_params,
                'current_chi2_dof': chi2_dof,
                'is_best': is_best,
                'best_params': best_params,
                'best_chi2_dof': best_chi2_dof,
                'best_result': best_result,
                'status': 'running',
                'description': f"ทดสอบ: Γ={current_params.get('photon_index', 0):.2f}, Norm={current_params.get('pl_norm', 0):.4f}"
            }
            
        except Exception as e:
            # Skip failed combinations
            yield {
                'iteration': i + 1,
                'total': total_combinations,
                'progress': (i + 1) / total_combinations,
                'current_params': current_params,
                'current_chi2_dof': float('inf'),
                'is_best': False,
                'best_params': best_params,
                'best_chi2_dof': best_chi2_dof,
                'best_result': best_result,
                'status': 'error',
                'description': f"ข้ามเนื่องจากเกิดข้อผิดพลาด: {str(e)[:50]}"
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
        'description': f"เสร็จสิ้น! พบค่าที่ดีที่สุด χ²/dof = {best_chi2_dof:.3f}"
    }


def _compute_chi2_for_params(args):
    """
    Helper function สำหรับคำนวณ chi-squared ของ parameter set เดียว
    ใช้สำหรับ multiprocessing (ต้องเป็น top-level function)
    """
    (param_values, param_names, energy, observed_rate, observed_error, 
     model_components, arf_values) = args
    
    try:
        current_params = dict(zip(param_names, param_values))
        
        # Calculate model
        model_photon_flux = sm.combined_model(energy, current_params, model_components)
        
        # Fold through response (simplified - multiply by ARF)
        if arf_values is not None:
            model_rate = model_photon_flux * arf_values
        else:
            model_rate = model_photon_flux
        
        # Calculate chi-squared
        mask = observed_error > 0
        chi_sq = np.sum(((observed_rate[mask] - model_rate[mask]) / observed_error[mask]) ** 2)
        n_data = np.sum(mask)
        dof = n_data - len(param_names)
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
                             n_parts=10, skip_parts=None):
    """
    Parallel Brute-force grid search with partitioning support
    
    Parameters:
    -----------
    ...
    n_parts : int
        Number of parts to split the search into
    skip_parts : list
        List of part indices (0-based) to skip
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
    
    if skip_parts is None:
        skip_parts = []
    
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
    
    # Prepare ARF values
    arf_values = None
    if response is not None and response.arf is not None:
        arf_values = np.array(response.arf)
    
    # Convert arrays
    energy_arr = np.array(energy)
    rate_arr = np.array(observed_rate)
    error_arr = np.array(observed_error)
    
    # Track best result
    best_chi2_dof = float('inf')
    best_params = None
    best_result = None
    processed = 0
    
    # Initial yield
    yield {
        'iteration': 0,
        'total': total_combinations,
        'progress': 0.0,
        'current_params': None,
        'current_chi2_dof': float('inf'),
        'is_best': False,
        'best_params': None,
        'best_chi2_dof': float('inf'),
        'best_result': None,
        'status': 'starting',
        'description': f"เริ่มต้น Search ({backend}) {n_parts} parts...",
        'n_workers': n_workers,
        'n_parts': n_parts,
        'part_size': part_size
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
            
            processed += current_part_size
            yield {
                'iteration': processed,
                'total': total_combinations,
                'progress': processed / total_combinations,
                'current_params': best_params,
                'current_chi2_dof': best_chi2_dof,
                'is_best': False,
                'best_params': best_params,
                'best_chi2_dof': best_chi2_dof,
                'best_result': None,
                'status': 'running',
                'description': f"ข้าม Part {part_idx+1}/{n_parts} (ทำเสร็จแล้ว)",
                'n_workers': n_workers,
                'batch_processed': 0,
                'skipped': True
            }
            continue

        # Process batches within this part
        part_processed = 0
        while part_processed < current_part_size:
            current_batch_size = min(batch_size, current_part_size - part_processed)
            
            # Take next batch from generator
            batch = list(itertools.islice(combination_generator, current_batch_size))
            if not batch:
                break
            
            # Prepare arguments
            batch_args = [
                (combo, param_names, energy_arr, rate_arr, error_arr, 
                 model_components, arf_values)
                for combo in batch
            ]
            
            # Process batch
            try:
                with Executor(max_workers=n_workers) as executor:
                    results = list(executor.map(_compute_chi2_for_params, batch_args))
                
                # Process results
                for result in results:
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
            
            # Yield progress
            yield {
                'iteration': processed,
                'total': total_combinations,
                'progress': processed / total_combinations,
                'current_params': best_params,
                'current_chi2_dof': best_chi2_dof,
                'is_best': True, # Always yield current best
                'best_params': best_params,
                'best_chi2_dof': best_chi2_dof,
                'best_result': best_result,
                'status': 'running',
                'description': f"Part {part_idx+1}/{n_parts}: {part_processed:,}/{current_part_size:,}",
                'n_workers': n_workers,
                'batch_processed': len(batch)
            }
        
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
            'part_index': part_idx,
            'description': f"✅ เสร็จสิ้น Part {part_idx+1}/{n_parts}",
            'n_workers': n_workers
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
        'description': f"เสร็จสิ้น! พบค่าที่ดีที่สุด χ²/dof = {best_chi2_dof:.3f}",
        'n_workers': n_workers
    }
