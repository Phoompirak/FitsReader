"""
X-ray Timing Analysis Module
Spectral-timing analysis สำหรับ X-ray observations

โมดูลนี้ประกอบด้วย:
1. Light curve loading and processing
2. Power spectrum analysis (PSD)
3. Lag-frequency and lag-energy spectra
4. RMS and covariance spectra

ใช้ Stingray library สำหรับการวิเคราะห์หลัก
"""

import numpy as np
from astropy.io import fits
import warnings

# Try importing stingray
try:
    from stingray import Lightcurve, Powerspectrum, AveragedPowerspectrum
    from stingray import Crossspectrum, AveragedCrossspectrum
    HAS_STINGRAY = True
except ImportError:
    HAS_STINGRAY = False
    warnings.warn("Stingray not installed. Install with: pip install stingray")


# ============================================================
# Light Curve Loading
# ============================================================

class TimingData:
    """Container for timing analysis data"""
    def __init__(self):
        self.time = None
        self.rate = None
        self.error = None
        self.energy_lo = None
        self.energy_hi = None
        self.dt = None
        self.exposure = None
        self.gti = None  # Good Time Intervals
        

def load_lightcurve_fits(filepath, time_col='TIME', rate_col='RATE', 
                          error_col='ERROR', extension=1):
    """
    Load light curve from FITS file
    
    Parameters:
    -----------
    filepath : str
        Path to FITS file
    time_col : str
        Column name for time
    rate_col : str
        Column name for count rate
    error_col : str
        Column name for error (optional)
    extension : int
        FITS extension number
        
    Returns:
    --------
    TimingData
        Light curve data container
    """
    data = TimingData()
    
    with fits.open(filepath) as hdul:
        if extension < len(hdul):
            table = hdul[extension].data
            header = hdul[extension].header
            
            if time_col in table.columns.names:
                data.time = table[time_col]
            else:
                # Try common alternatives
                for col in ['TIME', 'Time', 'MJD', 'TSTART']:
                    if col in table.columns.names:
                        data.time = table[col]
                        break
            
            if rate_col in table.columns.names:
                data.rate = table[rate_col]
            elif 'COUNTS' in table.columns.names:
                # Convert counts to rate
                if 'TIMEDEL' in header:
                    data.rate = table['COUNTS'] / header['TIMEDEL']
                else:
                    data.rate = table['COUNTS']
            
            if error_col in table.columns.names:
                data.error = table[error_col]
            elif data.rate is not None:
                # Poisson error
                data.error = np.sqrt(np.maximum(data.rate, 1))
            
            # Try to get bin size
            if data.time is not None and len(data.time) > 1:
                data.dt = np.median(np.diff(data.time))
            elif 'TIMEDEL' in header:
                data.dt = header['TIMEDEL']
            
            # Exposure
            if 'EXPOSURE' in header:
                data.exposure = header['EXPOSURE']
    
    return data


def load_event_file(filepath, energy_col='PI', time_col='TIME'):
    """
    Load event file and create light curve
    
    Parameters:
    -----------
    filepath : str
        Path to event FITS file
    energy_col : str
        Column for energy (PI or ENERGY)
    time_col : str
        Column for time
        
    Returns:
    --------
    dict
        Dictionary with time, energy arrays
    """
    with fits.open(filepath) as hdul:
        # Find events extension
        for ext in hdul:
            if hasattr(ext, 'data') and ext.data is not None:
                if time_col in ext.columns.names:
                    events = {
                        'time': ext.data[time_col],
                        'energy': ext.data.get(energy_col, None),
                        'pi': ext.data.get('PI', None),
                    }
                    return events
    return None


def bin_events_to_lightcurve(events, dt=1.0, energy_range=None):
    """
    Bin events into light curve
    
    Parameters:
    -----------
    events : dict
        Event data from load_event_file
    dt : float
        Time bin size (seconds)
    energy_range : tuple or None
        (E_min, E_max) in keV or PI channels
        
    Returns:
    --------
    TimingData
        Binned light curve
    """
    time = events['time']
    
    # Filter by energy if requested
    if energy_range is not None and events.get('pi') is not None:
        mask = (events['pi'] >= energy_range[0]) & (events['pi'] <= energy_range[1])
        time = time[mask]
    
    # Create bins
    t_start = time.min()
    t_end = time.max()
    bins = np.arange(t_start, t_end + dt, dt)
    
    # Histogram
    counts, _ = np.histogram(time, bins=bins)
    
    data = TimingData()
    data.time = (bins[:-1] + bins[1:]) / 2
    data.rate = counts / dt
    data.error = np.sqrt(counts) / dt
    data.dt = dt
    
    return data


# ============================================================
# Power Spectrum Analysis
# ============================================================

def compute_power_spectrum(lightcurve_data, segment_size=None, 
                           normalization='rms', rebin_factor=None):
    """
    Compute power spectrum (PSD)
    
    Parameters:
    -----------
    lightcurve_data : TimingData or Lightcurve
        Light curve data
    segment_size : float
        Segment size for averaging (seconds)
    normalization : str
        'rms' - fractional RMS squared
        'leahy' - Leahy normalization
        'none' - unnormalized
    rebin_factor : float
        Logarithmic rebinning factor
        
    Returns:
    --------
    dict
        frequency, power, error
    """
    if not HAS_STINGRAY:
        return _compute_power_spectrum_simple(lightcurve_data)
    
    # Convert to Stingray Lightcurve if needed
    if isinstance(lightcurve_data, TimingData):
        lc = Lightcurve(lightcurve_data.time, lightcurve_data.rate, 
                       dt=lightcurve_data.dt)
    else:
        lc = lightcurve_data
    
    # Compute power spectrum
    if segment_size is not None:
        ps = AveragedPowerspectrum(lc, segment_size=segment_size, 
                                    norm=normalization)
    else:
        ps = Powerspectrum(lc, norm=normalization)
    
    # Rebin if requested
    if rebin_factor is not None:
        ps = ps.rebin_log(rebin_factor)
    
    return {
        'frequency': ps.freq,
        'power': ps.power,
        'error': ps.power_err if hasattr(ps, 'power_err') else None,
        'normalization': normalization
    }


def _compute_power_spectrum_simple(lightcurve_data):
    """Simple FFT-based power spectrum without Stingray"""
    rate = lightcurve_data.rate
    dt = lightcurve_data.dt
    n = len(rate)
    
    # FFT
    fft = np.fft.rfft(rate - np.mean(rate))
    power = (2 * dt / n) * np.abs(fft)**2
    freq = np.fft.rfftfreq(n, dt)
    
    # RMS normalization
    power = power / np.mean(rate)**2
    
    return {
        'frequency': freq[1:],  # Skip DC component
        'power': power[1:],
        'error': None,
        'normalization': 'rms'
    }


# ============================================================
# Cross Spectrum and Lag Analysis
# ============================================================

def compute_cross_spectrum(lc1, lc2, segment_size=None):
    """
    Compute cross spectrum between two light curves
    
    Parameters:
    -----------
    lc1 : TimingData
        Reference band light curve
    lc2 : TimingData
        Subject band light curve
    segment_size : float
        Segment size for averaging
        
    Returns:
    --------
    dict
        frequency, cross_power, phase, time_lag
    """
    if not HAS_STINGRAY:
        return _compute_cross_spectrum_simple(lc1, lc2)
    
    # Convert to Stingray Lightcurves
    if isinstance(lc1, TimingData):
        lc1 = Lightcurve(lc1.time, lc1.rate, dt=lc1.dt)
    if isinstance(lc2, TimingData):
        lc2 = Lightcurve(lc2.time, lc2.rate, dt=lc2.dt)
    
    if segment_size is not None:
        cs = AveragedCrossspectrum(lc1, lc2, segment_size=segment_size)
    else:
        cs = Crossspectrum(lc1, lc2)
    
    # Compute time lags
    freq = cs.freq
    phase = np.angle(cs.power)
    time_lag = phase / (2 * np.pi * freq)
    
    return {
        'frequency': freq,
        'cross_power': np.abs(cs.power),
        'phase': phase,
        'time_lag': time_lag,
        'coherence': cs.coherence() if hasattr(cs, 'coherence') else None
    }


def _compute_cross_spectrum_simple(lc1, lc2):
    """Simple cross spectrum without Stingray"""
    rate1 = lc1.rate - np.mean(lc1.rate)
    rate2 = lc2.rate - np.mean(lc2.rate)
    dt = lc1.dt
    n = len(rate1)
    
    fft1 = np.fft.rfft(rate1)
    fft2 = np.fft.rfft(rate2)
    
    cross = fft1 * np.conj(fft2)
    freq = np.fft.rfftfreq(n, dt)
    
    phase = np.angle(cross)
    time_lag = phase / (2 * np.pi * freq + 1e-10)
    
    return {
        'frequency': freq[1:],
        'cross_power': np.abs(cross[1:]),
        'phase': phase[1:],
        'time_lag': time_lag[1:],
        'coherence': None
    }


def compute_lag_frequency_spectrum(lc_ref, lc_subj, segment_size, 
                                    freq_range=None):
    """
    Compute lag-frequency spectrum
    
    Parameters:
    -----------
    lc_ref : TimingData
        Reference band light curve
    lc_subj : TimingData
        Subject band light curve
    segment_size : float
        Segment size (seconds)
    freq_range : tuple
        (f_min, f_max) to select frequency range
        
    Returns:
    --------
    dict
        frequency, lag, lag_error
    """
    cs = compute_cross_spectrum(lc_ref, lc_subj, segment_size)
    
    freq = cs['frequency']
    lag = cs['time_lag']
    
    # Apply frequency filter
    if freq_range is not None:
        mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
        freq = freq[mask]
        lag = lag[mask]
    
    return {
        'frequency': freq,
        'lag': lag,
        'phase': cs['phase'],
        'coherence': cs['coherence']
    }


def compute_lag_energy_spectrum(event_data, energy_bands, ref_band, 
                                 dt=1.0, segment_size=256):
    """
    Compute lag-energy spectrum
    
    Parameters:
    -----------
    event_data : dict
        Event data from load_event_file
    energy_bands : list of tuples
        List of (E_lo, E_hi) for each energy band
    ref_band : tuple
        Energy range for reference band (E_lo, E_hi)
    dt : float
        Light curve time bin
    segment_size : float
        Segment size for cross spectrum
        
    Returns:
    --------
    dict
        energy, lag, lag_error
    """
    # Create reference light curve
    lc_ref = bin_events_to_lightcurve(event_data, dt=dt, energy_range=ref_band)
    
    energies = []
    lags = []
    
    for E_lo, E_hi in energy_bands:
        # Subject band light curve
        lc_subj = bin_events_to_lightcurve(event_data, dt=dt, 
                                           energy_range=(E_lo, E_hi))
        
        # Compute cross spectrum
        cs = compute_cross_spectrum(lc_ref, lc_subj, segment_size)
        
        # Average lag over frequencies
        avg_lag = np.nanmean(cs['time_lag'])
        
        energies.append((E_lo + E_hi) / 2)
        lags.append(avg_lag)
    
    return {
        'energy': np.array(energies),
        'lag': np.array(lags),
        'ref_band': ref_band
    }


# ============================================================
# RMS and Covariance Spectra
# ============================================================

def compute_rms_spectrum(event_data, energy_bands, dt=1.0, 
                          freq_range=None):
    """
    Compute fractional RMS spectrum vs energy
    
    Parameters:
    -----------
    event_data : dict
        Event data from load_event_file
    energy_bands : list of tuples
        List of (E_lo, E_hi) for each energy band
    dt : float
        Light curve time bin
    freq_range : tuple
        Frequency range to integrate RMS (f_min, f_max)
        
    Returns:
    --------
    dict
        energy, rms, rms_error
    """
    energies = []
    rms_values = []
    
    for E_lo, E_hi in energy_bands:
        # Light curve in this energy band
        lc = bin_events_to_lightcurve(event_data, dt=dt, 
                                      energy_range=(E_lo, E_hi))
        
        # Power spectrum
        ps = compute_power_spectrum(lc, normalization='rms')
        
        freq = ps['frequency']
        power = ps['power']
        
        # Integrate over frequency range
        if freq_range is not None:
            mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            power = power[mask]
            df = np.diff(freq[mask])
            if len(df) > 0:
                rms_sq = np.sum(power[:-1] * df)
            else:
                rms_sq = 0
        else:
            df = np.diff(freq)
            rms_sq = np.sum(power[:-1] * df)
        
        rms = np.sqrt(rms_sq) if rms_sq > 0 else 0
        
        energies.append((E_lo + E_hi) / 2)
        rms_values.append(rms)
    
    return {
        'energy': np.array(energies),
        'rms': np.array(rms_values),
        'freq_range': freq_range
    }


def compute_covariance_spectrum(event_data, ref_band, energy_bands,
                                 dt=1.0, segment_size=256):
    """
    Compute covariance spectrum
    
    Parameters:
    -----------
    event_data : dict
        Event data
    ref_band : tuple
        Reference energy band (E_lo, E_hi)
    energy_bands : list of tuples
        Subject energy bands
    dt : float
        Time bin
    segment_size : float
        Segment size
        
    Returns:
    --------
    dict
        energy, covariance
    """
    # Reference light curve
    lc_ref = bin_events_to_lightcurve(event_data, dt=dt, energy_range=ref_band)
    ref_var = np.var(lc_ref.rate)
    
    energies = []
    covariances = []
    
    for E_lo, E_hi in energy_bands:
        lc_subj = bin_events_to_lightcurve(event_data, dt=dt,
                                           energy_range=(E_lo, E_hi))
        
        # Covariance = correlation * std_ref * std_subj
        cov = np.cov(lc_ref.rate, lc_subj.rate)[0, 1]
        
        energies.append((E_lo + E_hi) / 2)
        covariances.append(cov)
    
    return {
        'energy': np.array(energies),
        'covariance': np.array(covariances),
        'ref_band': ref_band
    }


# ============================================================
# Utility Functions
# ============================================================

def check_stingray():
    """Check if Stingray is available and return info"""
    if HAS_STINGRAY:
        import stingray
        return {
            'available': True,
            'version': stingray.__version__ if hasattr(stingray, '__version__') else 'unknown'
        }
    return {
        'available': False,
        'install_cmd': 'pip install stingray'
    }


def interpret_lag_results(lag_energy_result):
    """
    Interpret lag-energy spectrum results
    
    Parameters:
    -----------
    lag_energy_result : dict
        Output from compute_lag_energy_spectrum
        
    Returns:
    --------
    str
        Physical interpretation
    """
    energy = lag_energy_result['energy']
    lag = lag_energy_result['lag']
    
    lines = []
    
    # Check for reverberation signature
    # Soft lags (soft photons arrive later) indicate reverberation
    soft_mask = energy < 2  # < 2 keV
    hard_mask = energy > 5  # > 5 keV
    
    if len(lag[soft_mask]) > 0 and len(lag[hard_mask]) > 0:
        soft_lag = np.mean(lag[soft_mask])
        hard_lag = np.mean(lag[hard_mask])
        
        if soft_lag > hard_lag:
            lines.append("🔄 **พบสัญญาณ Reverberation**: Soft X-rays มาถึงหลัง Hard X-rays")
            lines.append(f"   Soft lag: {soft_lag*1000:.1f} ms, Hard lag: {hard_lag*1000:.1f} ms")
            lines.append("   → บ่งชี้ reflection จาก disk ที่อยู่ใกล้หลุมดำ")
        else:
            lines.append("🔄 **Hard Lags พบ**: Hard X-rays มาถึงหลัง")
            lines.append("   → บ่งชี้ propagation ของ fluctuations ใน disk/corona")
    
    # Check Fe K region
    fe_mask = (energy > 6) & (energy < 7)
    if len(lag[fe_mask]) > 0:
        fe_lag = np.mean(lag[fe_mask])
        lines.append(f"\n⚡ **Fe K region lag**: {fe_lag*1000:.1f} ms")
        if fe_lag > 0:
            lines.append("   → Fe line responds to continuum → reflection confirmed")
    
    return "\n".join(lines) if lines else "ไม่สามารถตีความผลได้อย่างชัดเจน"
