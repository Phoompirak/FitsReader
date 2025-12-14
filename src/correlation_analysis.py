"""
Parameter Correlation Analysis Module
สำหรับวิเคราะห์ความสัมพันธ์ระหว่างพารามิเตอร์

โมดูลนี้ประกอบด้วย:
1. Correlation matrix calculation
2. Γ vs ξ vs R analysis
3. Corner plot generation
4. Degeneracy analysis
5. MCMC posterior sampling (optional)
"""

import numpy as np
from scipy import stats
import json


# ============================================================
# Correlation Matrix
# ============================================================

def compute_correlation_matrix(fit_results_list, param_names=None):
    """
    Compute correlation matrix between parameters from multiple fits
    
    Parameters:
    -----------
    fit_results_list : list of dict
        List of fit results
    param_names : list of str or None
        Parameters to include
        
    Returns:
    --------
    dict
        correlation_matrix, covariance_matrix, param_names
    """
    if not fit_results_list:
        return {}
    
    # Get parameter names
    if param_names is None:
        param_names = list(fit_results_list[0].get('best_params', {}).keys())
    
    # Extract parameter values
    n_params = len(param_names)
    n_samples = len(fit_results_list)
    
    param_values = np.zeros((n_samples, n_params))
    
    for i, result in enumerate(fit_results_list):
        params = result.get('best_params', {})
        for j, name in enumerate(param_names):
            param_values[i, j] = params.get(name, np.nan)
    
    # Remove rows with NaN
    valid_mask = ~np.isnan(param_values).any(axis=1)
    param_values = param_values[valid_mask]
    
    if len(param_values) < 2:
        return {'error': 'Not enough valid samples'}
    
    # Compute correlation and covariance
    corr_matrix = np.corrcoef(param_values.T)
    cov_matrix = np.cov(param_values.T)
    
    return {
        'param_names': param_names,
        'correlation': corr_matrix,
        'covariance': cov_matrix,
        'n_samples': len(param_values),
        'values': param_values
    }


def analyze_known_degeneracies(correlation_result):
    """
    Analyze known parameter degeneracies in X-ray spectral fitting
    
    Parameters:
    -----------
    correlation_result : dict
        Output from compute_correlation_matrix
        
    Returns:
    --------
    list of str
        Degeneracy warnings and interpretations
    """
    warnings = []
    
    param_names = correlation_result.get('param_names', [])
    corr = correlation_result.get('correlation', np.array([]))
    
    if len(corr) == 0:
        return warnings
    
    # Create param index mapping
    param_idx = {name: i for i, name in enumerate(param_names)}
    
    # Known degeneracies in X-ray spectral fitting
    
    # 1. Γ - nH degeneracy (photon index vs absorption)
    if 'photon_index' in param_idx and 'nH' in param_idx:
        i, j = param_idx['photon_index'], param_idx['nH']
        r = corr[i, j]
        if abs(r) > 0.5:
            warnings.append(f"⚠️ **Γ-nH Degeneracy** (r = {r:.2f})")
            warnings.append("   → Steeper power-law สามารถชดเชยด้วย less absorption")
            warnings.append("   → ควรใช้ constraint จาก Galactic nH")
    
    # 2. Γ - Reflection degeneracy
    gamma_names = ['photon_index', 'Gamma']
    refl_names = ['refl_frac', 'refl_norm', 'R']
    
    for g_name in gamma_names:
        for r_name in refl_names:
            if g_name in param_idx and r_name in param_idx:
                i, j = param_idx[g_name], param_idx[r_name]
                r = corr[i, j]
                if abs(r) > 0.5:
                    warnings.append(f"⚠️ **Γ-Reflection Degeneracy** (r = {r:.2f})")
                    warnings.append("   → Harder continuum mimics stronger reflection")
                    warnings.append("   → ลองใช้ broader band data หรือ flix refl_frac")
    
    # 3. ξ - Reflection degeneracy
    if 'xi' in param_idx:
        for r_name in refl_names:
            if r_name in param_idx:
                i, j = param_idx['xi'], param_idx[r_name]
                r = corr[i, j]
                if abs(r) > 0.5:
                    warnings.append(f"⚠️ **ξ-Reflection Degeneracy** (r = {r:.2f})")
                    warnings.append("   → Higher ionization reduces reflection features")
                    warnings.append("   → Fe line profile ช่วยแยกได้")
    
    # 4. Spin - Inclination degeneracy
    if 'spin' in param_idx and 'incl' in param_idx:
        i, j = param_idx['spin'], param_idx['incl']
        r = corr[i, j]
        if abs(r) > 0.4:
            warnings.append(f"⚠️ **Spin-Inclination Degeneracy** (r = {r:.2f})")
            warnings.append("   → High spin + high inclination ให้ line profile คล้ายกัน")
    
    # 5. Norm - Γ degeneracy
    norm_names = ['pl_norm', 'norm', 'Norm']
    for n_name in norm_names:
        for g_name in gamma_names:
            if n_name in param_idx and g_name in param_idx:
                i, j = param_idx[n_name], param_idx[g_name]
                r = corr[i, j]
                if abs(r) > 0.7:
                    warnings.append(f"⚠️ **Norm-Γ Degeneracy** (r = {r:.2f})")
                    warnings.append("   → Mathematically coupled at pivot energy")
    
    return warnings


def analyze_gamma_xi_relation(fit_results_list):
    """
    Analyze Γ vs log(ξ) relation
    
    Physical expectation: 
    - Higher ionization (ξ) → softer X-ray spectrum (higher Γ)
    - Due to more efficient cooling
    
    Parameters:
    -----------
    fit_results_list : list of dict
        Fit results
        
    Returns:
    --------
    dict
        Correlation analysis results
    """
    gammas = []
    xis = []
    
    for result in fit_results_list:
        params = result.get('best_params', {})
        gamma = params.get('photon_index', params.get('Gamma', None))
        xi = params.get('xi', None)
        
        if gamma is not None and xi is not None and xi > 0:
            gammas.append(gamma)
            xis.append(np.log10(xi))
    
    if len(gammas) < 3:
        return {'error': 'Not enough data points'}
    
    gammas = np.array(gammas)
    log_xis = np.array(xis)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_xis, gammas)
    
    interpretation = []
    if r_value > 0.3 and p_value < 0.05:
        interpretation.append("✅ **Positive Γ-ξ correlation detected**")
        interpretation.append("   → สอดคล้องกับ photoionization physics")
        interpretation.append(f"   → Γ = {intercept:.2f} + {slope:.2f} × log(ξ)")
    elif r_value < -0.3 and p_value < 0.05:
        interpretation.append("⚠️ **Negative Γ-ξ correlation detected**")
        interpretation.append("   → อาจมี degeneracy ใน fitting")
    else:
        interpretation.append("📊 **No significant Γ-ξ correlation**")
    
    return {
        'gamma': gammas,
        'log_xi': log_xis,
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'interpretation': "\n".join(interpretation)
    }


def analyze_reflection_correlations(fit_results_list):
    """
    Analyze correlations involving reflection fraction
    
    Parameters:
    -----------
    fit_results_list : list of dict
        Fit results
        
    Returns:
    --------
    dict
        Correlation analysis
    """
    results = {}
    
    # Extract parameters
    refls = []
    gammas = []
    xis = []
    
    for result in fit_results_list:
        params = result.get('best_params', {})
        refl = params.get('refl_frac', params.get('refl_norm', None))
        gamma = params.get('photon_index', params.get('Gamma', None))
        xi = params.get('xi', None)
        
        if refl is not None:
            refls.append(refl)
            gammas.append(gamma)
            xis.append(xi if xi else np.nan)
    
    if len(refls) < 3:
        return {'error': 'Not enough data points'}
    
    refls = np.array(refls)
    
    # R vs Γ
    valid = [g is not None for g in gammas]
    if sum(valid) > 2:
        g_arr = np.array([g for g, v in zip(gammas, valid) if v])
        r_arr = np.array([r for r, v in zip(refls, valid) if v])
        r_gamma = np.corrcoef(r_arr, g_arr)[0, 1]
        results['R_Gamma_correlation'] = r_gamma
    
    # R vs log(ξ)
    valid_xi = [x is not None and x > 0 for x in xis]
    if sum(valid_xi) > 2:
        xi_arr = np.log10([x for x, v in zip(xis, valid_xi) if v])
        r_arr = np.array([r for r, v in zip(refls, valid_xi) if v])
        r_xi = np.corrcoef(r_arr, xi_arr)[0, 1]
        results['R_logxi_correlation'] = r_xi
    
    # Interpretation
    lines = []
    if 'R_Gamma_correlation' in results:
        r = results['R_Gamma_correlation']
        if r > 0.3:
            lines.append(f"📈 **R-Γ positive** (r = {r:.2f}): Softer → more reflection")
        elif r < -0.3:
            lines.append(f"📉 **R-Γ negative** (r = {r:.2f}): Harder → more reflection")
    
    if 'R_logxi_correlation' in results:
        r = results['R_logxi_correlation']
        if abs(r) > 0.3:
            lines.append(f"⚡ **R-ξ correlation** (r = {r:.2f})")
    
    results['interpretation'] = "\n".join(lines) if lines else "ไม่พบ correlations ที่ชัดเจน"
    
    return results


# ============================================================
# MCMC Posterior Sampling (Simplified)
# ============================================================

def simple_mcmc_posterior(objective_func, initial_params, param_bounds,
                          n_steps=1000, n_walkers=None, burn_in=200):
    """
    Simple MCMC posterior sampling using Metropolis-Hastings
    
    Parameters:
    -----------
    objective_func : callable
        Function to minimize (returns chi-squared)
    initial_params : dict
        Initial parameter values
    param_bounds : dict
        Parameter bounds {name: (min, max)}
    n_steps : int
        Number of MCMC steps
    n_walkers : int
        Number of walkers
    burn_in : int
        Burn-in steps to discard
        
    Returns:
    --------
    dict
        Posterior samples, mean, std
    """
    param_names = list(initial_params.keys())
    n_params = len(param_names)
    
    if n_walkers is None:
        n_walkers = max(2 * n_params, 4)
    
    # Initialize chains near initial parameters
    chains = np.zeros((n_walkers, n_steps, n_params))
    chi2_chains = np.zeros((n_walkers, n_steps))
    
    # Initial positions with small scatter
    for w in range(n_walkers):
        for i, name in enumerate(param_names):
            val = initial_params[name]
            lo, hi = param_bounds.get(name, (val * 0.5, val * 2))
            # Small random offset
            scatter = 0.05 * (hi - lo)
            chains[w, 0, i] = np.clip(val + np.random.normal(0, scatter), lo, hi)
    
    # Calculate initial chi2
    for w in range(n_walkers):
        params = dict(zip(param_names, chains[w, 0]))
        try:
            chi2_chains[w, 0] = objective_func(params)
        except:
            chi2_chains[w, 0] = 1e10
    
    # Proposal scale
    scales = np.array([0.05 * (param_bounds.get(n, (0.1, 10))[1] - 
                              param_bounds.get(n, (0.1, 10))[0]) 
                      for n in param_names])
    
    # MCMC iteration
    acceptance = np.zeros(n_walkers)
    
    for step in range(1, n_steps):
        for w in range(n_walkers):
            # Propose new position
            current = chains[w, step-1]
            proposal = current + np.random.normal(0, scales)
            
            # Apply bounds
            for i, name in enumerate(param_names):
                lo, hi = param_bounds.get(name, (-np.inf, np.inf))
                proposal[i] = np.clip(proposal[i], lo, hi)
            
            # Calculate chi2 for proposal
            params = dict(zip(param_names, proposal))
            try:
                chi2_proposal = objective_func(params)
            except:
                chi2_proposal = 1e10
            
            # Metropolis acceptance
            delta_chi2 = chi2_proposal - chi2_chains[w, step-1]
            
            if delta_chi2 < 0 or np.random.random() < np.exp(-0.5 * delta_chi2):
                # Accept
                chains[w, step] = proposal
                chi2_chains[w, step] = chi2_proposal
                acceptance[w] += 1
            else:
                # Reject
                chains[w, step] = current
                chi2_chains[w, step] = chi2_chains[w, step-1]
    
    # Remove burn-in and flatten
    samples = chains[:, burn_in:, :].reshape(-1, n_params)
    
    # Calculate statistics
    posterior = {}
    for i, name in enumerate(param_names):
        posterior[name] = {
            'samples': samples[:, i],
            'mean': np.mean(samples[:, i]),
            'std': np.std(samples[:, i]),
            'median': np.median(samples[:, i]),
            'q16': np.percentile(samples[:, i], 16),
            'q84': np.percentile(samples[:, i], 84)
        }
    
    return {
        'posterior': posterior,
        'param_names': param_names,
        'samples': samples,
        'acceptance_rate': acceptance / (n_steps - 1),
        'n_samples': len(samples)
    }


# ============================================================
# Visualization Helpers
# ============================================================

def prepare_corner_plot_data(posterior_result):
    """
    Prepare data for corner plot visualization
    
    Parameters:
    -----------
    posterior_result : dict
        Output from simple_mcmc_posterior
        
    Returns:
    --------
    dict
        Data formatted for corner plot
    """
    return {
        'samples': posterior_result['samples'],
        'labels': posterior_result['param_names'],
        'truths': [posterior_result['posterior'][n]['median'] 
                  for n in posterior_result['param_names']],
        'quantiles': [0.16, 0.5, 0.84]
    }


def interpret_posterior_results(posterior_result):
    """
    Interpret MCMC posterior results
    
    Parameters:
    -----------
    posterior_result : dict
        Output from simple_mcmc_posterior
        
    Returns:
    --------
    str
        Interpretation text
    """
    lines = []
    
    posterior = posterior_result.get('posterior', {})
    acceptance = posterior_result.get('acceptance_rate', [])
    
    # Acceptance rate check
    avg_acceptance = np.mean(acceptance)
    if avg_acceptance < 0.1:
        lines.append("⚠️ **Low acceptance rate** ({:.1%}): Proposal too large".format(avg_acceptance))
    elif avg_acceptance > 0.7:
        lines.append("⚠️ **High acceptance rate** ({:.1%}): Proposal too small".format(avg_acceptance))
    else:
        lines.append("✅ **Good acceptance rate** ({:.1%})".format(avg_acceptance))
    
    lines.append("\n**Parameter Constraints:**")
    
    for name, stats in posterior.items():
        mean = stats['mean']
        std = stats['std']
        rel_error = std / abs(mean) if abs(mean) > 1e-10 else 0
        
        if rel_error < 0.1:
            constraint = "🎯 Well constrained"
        elif rel_error < 0.3:
            constraint = "📊 Moderately constrained"
        else:
            constraint = "⚠️ Poorly constrained"
        
        lines.append(f"   {name}: {mean:.3f} ± {std:.3f} ({constraint})")
    
    return "\n".join(lines)
