"""
Multi-epoch Spectral Comparison Module
สำหรับการเปรียบเทียบสเปกตรัมหลาย epochs

โมดูลนี้ประกอบด้วย:
1. Loading multiple spectra
2. Parameter evolution tracking  
3. Statistical comparison (F-test, χ² comparison)
4. Spectral evolution visualization
"""

import numpy as np
from astropy.io import fits
from scipy import stats
import json
import os
from datetime import datetime


class EpochData:
    """Container for single epoch spectral data"""
    def __init__(self, name=None):
        self.name = name
        self.obs_id = None
        self.date = None
        self.exposure = None
        self.energy = None
        self.rate = None
        self.error = None
        self.fit_result = None
        

def load_multiple_spectra(file_list, names=None):
    """
    Load multiple spectrum files for comparison
    
    Parameters:
    -----------
    file_list : list of str
        Paths to spectrum FITS files
    names : list of str or None
        Optional names for each epoch
        
    Returns:
    --------
    list of EpochData
        List of spectral data objects
    """
    epochs = []
    
    for i, filepath in enumerate(file_list):
        epoch = EpochData()
        epoch.name = names[i] if names and i < len(names) else f"Epoch {i+1}"
        
        try:
            with fits.open(filepath) as hdul:
                # Read spectrum extension
                if len(hdul) > 1:
                    data = hdul[1].data
                    header = hdul[1].header
                    
                    # Get observation info
                    epoch.obs_id = header.get('OBS_ID', 'Unknown')
                    epoch.date = header.get('DATE-OBS', header.get('DATE', None))
                    epoch.exposure = header.get('EXPOSURE', None)
                    
                    # Get spectral data
                    if 'CHANNEL' in data.columns.names:
                        epoch.energy = data['CHANNEL'].astype(float)
                    if 'COUNTS' in data.columns.names:
                        counts = data['COUNTS'].astype(float)
                        if epoch.exposure:
                            epoch.rate = counts / epoch.exposure
                            epoch.error = np.sqrt(counts) / epoch.exposure
                        else:
                            epoch.rate = counts
                            epoch.error = np.sqrt(counts)
                    elif 'RATE' in data.columns.names:
                        epoch.rate = data['RATE']
                        if 'ERROR' in data.columns.names:
                            epoch.error = data['ERROR']
                            
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            continue
            
        epochs.append(epoch)
    
    return epochs


def compare_fit_results(fit_results_list, param_names=None):
    """
    Compare fitting results across epochs
    
    Parameters:
    -----------
    fit_results_list : list of dict
        List of fit results from brute-force or fitting
    param_names : list of str or None
        Parameters to compare
        
    Returns:
    --------
    dict
        Comparison statistics
    """
    if not fit_results_list:
        return {}
    
    # Get all parameter names
    if param_names is None:
        param_names = list(fit_results_list[0].get('best_params', {}).keys())
    
    comparison = {
        'parameters': {},
        'chi2': [],
        'epochs': len(fit_results_list)
    }
    
    for param in param_names:
        values = []
        for result in fit_results_list:
            params = result.get('best_params', {})
            if param in params:
                values.append(params[param])
        
        if values:
            comparison['parameters'][param] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }
    
    # Chi-squared values
    for result in fit_results_list:
        chi2 = result.get('reduced_chi_squared', 
                         result.get('best_chi2_dof', None))
        if chi2 is not None:
            comparison['chi2'].append(chi2)
    
    return comparison


def statistical_comparison(epoch1_result, epoch2_result, dof1=None, dof2=None):
    """
    Statistical comparison between two epochs using F-test
    
    Parameters:
    -----------
    epoch1_result : dict
        Fit result for epoch 1
    epoch2_result : dict
        Fit result for epoch 2
    dof1, dof2 : int or None
        Degrees of freedom (if not in results)
        
    Returns:
    --------
    dict
        F-statistic, p-value, and interpretation
    """
    chi1 = epoch1_result.get('chi_squared', 
                             epoch1_result.get('best_result', {}).get('chi_squared', None))
    chi2 = epoch2_result.get('chi_squared',
                             epoch2_result.get('best_result', {}).get('chi_squared', None))
    
    dof1 = dof1 or epoch1_result.get('dof',
                                      epoch1_result.get('best_result', {}).get('dof', 100))
    dof2 = dof2 or epoch2_result.get('dof',
                                      epoch2_result.get('best_result', {}).get('dof', 100))
    
    if chi1 is None or chi2 is None:
        return {'error': 'Missing chi-squared values'}
    
    # F-test
    F = (chi1 / dof1) / (chi2 / dof2 + 1e-10)
    p_value = 1 - stats.f.cdf(F, dof1, dof2)
    
    # Interpretation
    if p_value < 0.01:
        interpretation = "❌ **Significant difference** (p < 0.01): สเปกตรัมเปลี่ยนแปลงอย่างมีนัยสำคัญ"
    elif p_value < 0.05:
        interpretation = "⚠️ **Marginal difference** (p < 0.05): อาจมีการเปลี่ยนแปลงเล็กน้อย"
    else:
        interpretation = "✅ **No significant difference** (p > 0.05): สเปกตรัมคงที่"
    
    return {
        'F_statistic': F,
        'p_value': p_value,
        'chi2_1': chi1,
        'chi2_2': chi2,
        'dof_1': dof1,
        'dof_2': dof2,
        'interpretation': interpretation
    }


def track_parameter_evolution(epochs, param_name):
    """
    Track a parameter's evolution across epochs
    
    Parameters:
    -----------
    epochs : list of EpochData
        Epoch data with fit_result attached
    param_name : str
        Parameter to track
        
    Returns:
    --------
    dict
        times, values, errors for plotting
    """
    times = []
    values = []
    errors = []
    names = []
    
    for i, epoch in enumerate(epochs):
        if epoch.fit_result is None:
            continue
            
        params = epoch.fit_result.get('best_params', {})
        if param_name in params:
            times.append(i)
            values.append(params[param_name])
            names.append(epoch.name)
            
            # Try to get error if available
            param_errors = epoch.fit_result.get('param_errors', {})
            if param_name in param_errors:
                errors.append(param_errors[param_name])
            else:
                errors.append(0)
    
    return {
        'epochs': names,
        'times': np.array(times),
        'values': np.array(values),
        'errors': np.array(errors),
        'param_name': param_name
    }


def interpret_spectral_evolution(comparison):
    """
    Interpret spectral evolution results
    
    Parameters:
    -----------
    comparison : dict
        Output from compare_fit_results
        
    Returns:
    --------
    str
        Physical interpretation
    """
    lines = []
    
    if not comparison or 'parameters' not in comparison:
        return "ไม่สามารถวิเคราะห์ได้"
    
    params = comparison['parameters']
    
    # Photon index evolution
    if 'photon_index' in params or 'Gamma' in params:
        gamma = params.get('photon_index', params.get('Gamma', {}))
        if 'std' in gamma:
            if gamma['std'] > 0.2:
                lines.append(f"📈 **Photon Index ผันผวน** (σ = {gamma['std']:.2f})")
                lines.append("   → Corona temperature/optical depth เปลี่ยนแปลง")
            else:
                lines.append(f"📊 **Photon Index คงที่** (σ = {gamma['std']:.2f})")
    
    # Reflection evolution
    if 'refl_frac' in params or 'refl_norm' in params:
        refl = params.get('refl_frac', params.get('refl_norm', {}))
        if 'range' in refl and refl['range'] > 0.5:
            lines.append(f"🪞 **Reflection เปลี่ยนแปลง** (range = {refl['range']:.2f})")
            lines.append("   → Corona geometry หรือ disk ionization เปลี่ยน")
    
    # Ionization evolution
    if 'xi' in params:
        xi = params['xi']
        if 'std' in xi and xi['mean'] > 0:
            rel_var = xi['std'] / xi['mean']
            if rel_var > 0.3:
                lines.append(f"⚡ **Ionization ผันผวน** ({rel_var*100:.0f}%)")
                lines.append("   → Luminosity หรือ disk density เปลี่ยนแปลง")
    
    # Chi-squared trend
    chi2_list = comparison.get('chi2', [])
    if chi2_list:
        avg_chi2 = np.mean(chi2_list)
        if avg_chi2 < 1.5:
            lines.append(f"\n✅ **ค่าเฉลี่ย χ²/dof = {avg_chi2:.2f}** - Model fits ดีทุก epochs")
        else:
            lines.append(f"\n⚠️ **ค่าเฉลี่ย χ²/dof = {avg_chi2:.2f}** - อาจต้องปรับปรุง model")
    
    return "\n".join(lines) if lines else "ไม่พบ patterns ที่ชัดเจน"


def save_multi_epoch_results(epochs, comparison, filepath):
    """Save multi-epoch analysis results to JSON"""
    
    data = {
        'epochs': [],
        'comparison': comparison,
        'saved_at': str(datetime.now())
    }
    
    for epoch in epochs:
        epoch_data = {
            'name': epoch.name,
            'obs_id': epoch.obs_id,
            'date': epoch.date,
            'exposure': epoch.exposure,
            'fit_result': epoch.fit_result
        }
        data['epochs'].append(epoch_data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def load_multi_epoch_results(filepath):
    """Load multi-epoch analysis results from JSON"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    epochs = []
    for epoch_data in data.get('epochs', []):
        epoch = EpochData(name=epoch_data.get('name'))
        epoch.obs_id = epoch_data.get('obs_id')
        epoch.date = epoch_data.get('date')
        epoch.exposure = epoch_data.get('exposure')
        epoch.fit_result = epoch_data.get('fit_result')
        epochs.append(epoch)
    
    return epochs, data.get('comparison', {})
