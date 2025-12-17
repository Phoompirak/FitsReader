# -*- coding: utf-8 -*-
"""
Data Loader Module for X-ray Spectral Analysis (Google Colab)

โมดูลนี้ใช้สำหรับอ่านไฟล์ FITS ที่ใช้ในการวิเคราะห์สเปกตรัมรังสีเอกซ์:
- 📊 Source Spectrum: ไฟล์สเปกตรัมหลักของ source
- 🌌 Background Spectrum: ไฟล์สัญญาณรบกวนพื้นหลัง
- 🔬 Background Subtraction: ลบ background ออกจาก source
- 📈 ARF File: Ancillary Response File (Effective Area)
- 🔲 RMF File: Redistribution Matrix File

Dependencies:
    pip install astropy numpy
"""

import numpy as np
from astropy.io import fits


# ============================================================
# Data Classes - เก็บข้อมูลที่โหลดมา
# ============================================================

class SpectrumData:
    """
    Class สำหรับเก็บข้อมูลสเปกตรัม
    
    Attributes:
        channel (array): Channel numbers
        counts (array): Photon counts
        exposure (float): Exposure time (seconds)
        backscal (float): Background scaling factor
        grouping (array): Grouping info
        quality (array): Quality flags
    """
    def __init__(self):
        self.channel = None   # เลข channel
        self.counts = None    # จำนวน counts
        self.exposure = None  # เวลา exposure (วินาที)
        self.backscal = 1.0   # ค่า scaling สำหรับ background
        self.grouping = None  # grouping info
        self.quality = None   # quality flags
        
    def count_rate(self):
        """คืนค่า count rate (counts/s)"""
        if self.exposure is not None and self.exposure > 0:
            return self.counts / self.exposure
        return self.counts
    
    def count_rate_error(self):
        """คืนค่า error (Poisson statistics: sqrt(counts))"""
        if self.exposure is not None and self.exposure > 0:
            return np.sqrt(np.maximum(self.counts, 1.0)) / self.exposure
        return np.sqrt(np.maximum(self.counts, 1.0))


class ResponseData:
    """
    Class สำหรับเก็บข้อมูล Response (ARF และ RMF)
    
    Attributes:
        energy_lo (array): Lower energy bounds (keV)
        energy_hi (array): Upper energy bounds (keV)
        energy_mid (array): Midpoint energies (keV)
        arf (array): Effective area (cm²)
        rmf_matrix (array): Response matrix
        channel_lo (array): Channel lower bounds
        channel_hi (array): Channel upper bounds
    """
    def __init__(self):
        self.energy_lo = None    # ขอบล่างของ energy bin (keV)
        self.energy_hi = None    # ขอบบนของ energy bin (keV)
        self.energy_mid = None   # ค่ากลางของ energy bin (keV)
        self.arf = None          # Effective area (cm²)
        self.rmf_matrix = None   # Response matrix
        self.channel_lo = None   # ขอบล่างของ channel
        self.channel_hi = None   # ขอบบนของ channel


# ============================================================
# File Reading Functions - ฟังก์ชันอ่านไฟล์
# ============================================================

def read_spectrum_file(filepath):
    """
    📊 อ่านไฟล์สเปกตรัม FITS (.pha หรือ .fits)
    
    Algorithm:
    1. เปิดไฟล์ FITS และอ่าน HDU 1 (SPECTRUM extension)
    2. ดึงข้อมูล columns: CHANNEL, COUNTS, GROUPING, QUALITY
    3. อ่าน header keywords: EXPOSURE, BACKSCAL
    
    Parameters:
        filepath (str): Path ไปยังไฟล์ spectrum
        
    Returns:
        SpectrumData: Object ที่มีข้อมูลสเปกตรัม หรือ None ถ้า error
        
    Example:
        >>> source = read_spectrum_file('/path/to/source.pha')
        >>> print(f"Channels: {len(source.channel)}")
        >>> print(f"Exposure: {source.exposure} seconds")
    """
    spectrum = SpectrumData()
    
    try:
        with fits.open(filepath) as hdul:
            # อ่านจาก SPECTRUM extension (มักอยู่ที่ HDU 1)
            if len(hdul) > 1:
                data = hdul[1].data
                header = hdul[1].header
                
                # อ่าน columns ที่จำเป็น
                if 'CHANNEL' in data.columns.names:
                    spectrum.channel = data['CHANNEL']
                if 'COUNTS' in data.columns.names:
                    spectrum.counts = data['COUNTS'].astype(float)
                if 'GROUPING' in data.columns.names:
                    spectrum.grouping = data['GROUPING']
                if 'QUALITY' in data.columns.names:
                    spectrum.quality = data['QUALITY']
                    
                # อ่าน header keywords
                if 'EXPOSURE' in header:
                    spectrum.exposure = header['EXPOSURE']
                if 'BACKSCAL' in header:
                    spectrum.backscal = header['BACKSCAL']
                    
    except Exception as e:
        print(f"❌ Error reading spectrum file: {e}")
        return None
        
    return spectrum


def read_arf_file(filepath):
    """
    📈 อ่านไฟล์ ARF (Ancillary Response File)
    
    ARF File บอก Effective Area ของเครื่องมือที่แต่ละ energy
    - หน่วย: cm²
    - ใช้ในการแปลง photon flux เป็น count rate
    
    Algorithm:
    1. เปิดไฟล์ ARF (มักเป็น .arf)
    2. อ่าน columns: ENERG_LO, ENERG_HI, SPECRESP
    3. คำนวณ energy midpoints
    
    Parameters:
        filepath (str): Path ไปยังไฟล์ ARF
        
    Returns:
        ResponseData: Object ที่มีข้อมูล ARF หรือ None ถ้า error
        
    Example:
        >>> arf = read_arf_file('/path/to/file.arf')
        >>> print(f"Energy range: {arf.energy_lo.min()}-{arf.energy_hi.max()} keV")
    """
    response = ResponseData()
    
    try:
        with fits.open(filepath) as hdul:
            if len(hdul) > 1:
                data = hdul[1].data
                
                # อ่าน energy bounds และ effective area
                if 'ENERG_LO' in data.columns.names:
                    response.energy_lo = data['ENERG_LO']
                if 'ENERG_HI' in data.columns.names:
                    response.energy_hi = data['ENERG_HI']
                if 'SPECRESP' in data.columns.names:
                    response.arf = data['SPECRESP']
                    
                # คำนวณ energy midpoints
                if response.energy_lo is not None and response.energy_hi is not None:
                    response.energy_mid = (response.energy_lo + response.energy_hi) / 2.0
                    
    except Exception as e:
        print(f"❌ Error reading ARF file: {e}")
        return None
        
    return response


def read_rmf_file(filepath):
    """
    🔲 อ่านไฟล์ RMF (Redistribution Matrix File)
    
    RMF File บอกการกระจายของ energy ไปยัง channels
    - EBOUNDS: mapping ของ channel กับ energy
    - MATRIX: redistribution matrix
    
    Algorithm:
    1. เปิดไฟล์ RMF
    2. หา EBOUNDS extension สำหรับ channel info
    3. หา MATRIX extension สำหรับ energy info
    4. คำนวณ energy midpoints
    
    Parameters:
        filepath (str): Path ไปยังไฟล์ RMF
        
    Returns:
        ResponseData: Object ที่มีข้อมูล RMF หรือ None ถ้า error
        
    Note:
        การอ่าน full RMF matrix ซับซ้อน ที่นี่ใช้ simplified version
    """
    response = ResponseData()
    
    try:
        with fits.open(filepath) as hdul:
            ebounds_hdu = None
            matrix_hdu = None
            
            # หา extensions ที่ต้องการ
            for hdu in hdul:
                if hdu.name == 'EBOUNDS':
                    ebounds_hdu = hdu
                elif hdu.name in ['MATRIX', 'SPECRESP MATRIX']:
                    matrix_hdu = hdu
                    
            # อ่าน channel boundaries จาก EBOUNDS
            if ebounds_hdu is not None:
                data = ebounds_hdu.data
                if 'CHANNEL' in data.columns.names:
                    response.channel_lo = data['CHANNEL']
                    response.channel_hi = data['CHANNEL']
                    
            # อ่าน energy data จาก MATRIX
            if matrix_hdu is not None:
                data = matrix_hdu.data
                
                if 'ENERG_LO' in data.columns.names:
                    response.energy_lo = data['ENERG_LO']
                if 'ENERG_HI' in data.columns.names:
                    response.energy_hi = data['ENERG_HI']
                    
                # คำนวณ energy midpoints
                if response.energy_lo is not None and response.energy_hi is not None:
                    response.energy_mid = (response.energy_lo + response.energy_hi) / 2.0
                    
    except Exception as e:
        print(f"❌ Error reading RMF file: {e}")
        return None
        
    return response


# ============================================================
# Background Subtraction - ลบ Background
# ============================================================

def subtract_background(source_spec, bkg_spec):
    """
    🔬 ลบ Background ออกจาก Source Spectrum
    
    Formula:
        net_counts = source_counts - (bkg_counts × scale_factor)
        scale_factor = source_backscal / bkg_backscal
    
    Algorithm:
    1. คำนวณ scaling factor จาก BACKSCAL
    2. Scale background counts ตาม exposure time
    3. ลบ scaled background จาก source
    4. คำนวณ error propagation
    
    Parameters:
        source_spec (SpectrumData): Source spectrum
        bkg_spec (SpectrumData): Background spectrum
        
    Returns:
        tuple: (net_counts, net_error) หรือ (None, None) ถ้า error
        
    Example:
        >>> source = read_spectrum_file('source.pha')
        >>> bkg = read_spectrum_file('background.pha')
        >>> net_counts, net_error = subtract_background(source, bkg)
    """
    if source_spec is None or bkg_spec is None:
        print("❌ Source or background spectrum is None")
        return None, None
    
    if source_spec.counts is None or bkg_spec.counts is None:
        print("❌ Counts data is missing")
        return None, None
    
    # คำนวณ scaling factor
    # ปรับตาม backscal และ exposure time
    src_backscal = source_spec.backscal if source_spec.backscal else 1.0
    bkg_backscal = bkg_spec.backscal if bkg_spec.backscal else 1.0
    scale_factor = src_backscal / bkg_backscal
    
    # ปรับตาม exposure time ถ้ามี
    if source_spec.exposure and bkg_spec.exposure:
        time_scale = source_spec.exposure / bkg_spec.exposure
        scale_factor *= time_scale
    
    # ลบ background
    # net_counts = source - scaled_background
    scaled_bkg = bkg_spec.counts * scale_factor
    net_counts = source_spec.counts - scaled_bkg
    
    # Error propagation (Poisson errors add in quadrature)
    # σ_net² = σ_source² + (scale_factor × σ_bkg)²
    src_error = np.sqrt(np.maximum(source_spec.counts, 1.0))
    bkg_error = np.sqrt(np.maximum(bkg_spec.counts, 1.0)) * scale_factor
    net_error = np.sqrt(src_error**2 + bkg_error**2)
    
    return net_counts, net_error


# ============================================================
# Utility Functions - ฟังก์ชันช่วยเหลือ
# ============================================================

def get_energy_from_response(arf_data, rmf_data=None):
    """
    สร้าง energy array จาก response files
    
    Parameters:
        arf_data (ResponseData): ARF data
        rmf_data (ResponseData): RMF data (optional)
        
    Returns:
        array: Energy midpoints (keV)
    """
    if arf_data is not None and arf_data.energy_mid is not None:
        return np.array(arf_data.energy_mid)
    elif rmf_data is not None and rmf_data.energy_mid is not None:
        return np.array(rmf_data.energy_mid)
    else:
        print("⚠️ No energy information available, using default range")
        return np.linspace(0.3, 10.0, 100)


def filter_energy_range(energy, data, error=None, e_min=0.3, e_max=10.0):
    """
    กรองข้อมูลตามช่วง energy ที่ต้องการ
    
    Parameters:
        energy (array): Energy array (keV)
        data (array): Data array (counts or rate)
        error (array): Error array (optional)
        e_min (float): Minimum energy (keV)
        e_max (float): Maximum energy (keV)
        
    Returns:
        tuple: (filtered_energy, filtered_data, filtered_error)
    """
    # ⚠️ Check for dimension mismatch
    if len(energy) != len(data):
        print(f"⚠️ Warning: Dimension mismatch! Energy: {len(energy)}, Data: {len(data)}")
        
        min_len = min(len(energy), len(data))
        print(f"   -> Truncating/Aligning to {min_len} bins")
        
        energy = energy[:min_len]
        data = data[:min_len]
        if error is not None:
            error = error[:min_len]

    mask = (energy >= e_min) & (energy <= e_max)
    
    filtered_energy = energy[mask]
    filtered_data = data[mask]
    filtered_error = error[mask] if error is not None else None
    
    return filtered_energy, filtered_data, filtered_error


def fold_model_through_response(model_flux, response):
    """
    Fold model spectrum ผ่าน instrument response
    
    คำนวณ predicted count rate จาก model photon flux
    
    Formula:
        predicted_rate = model_flux × ARF × dE
        
    Units:
        - model_flux: photons/cm²/s/keV
        - ARF: cm²
        - dE: keV
        - predicted_rate: counts/s
    
    Parameters:
        model_flux (array): Model photon flux (photons/cm²/s/keV)
        response (ResponseData): Response data (ARF)
        
    Returns:
        array: Predicted count rate (counts/s)
    """
    if response.arf is None:
        return model_flux
    
    # Predicted = Model × ARF
    predicted_rate = model_flux * response.arf
    
    # Multiply by energy bin width (dE)
    if response.energy_hi is not None and response.energy_lo is not None:
        if len(response.energy_hi) == len(predicted_rate):
            dE = response.energy_hi - response.energy_lo
            predicted_rate *= dE
    
    return predicted_rate


# ============================================================
# Quick Test / Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📊 Data Loader Module for X-ray Spectral Analysis")
    print("=" * 60)
    print()
    print("Available functions:")
    print("  • read_spectrum_file(filepath) - อ่าน source/background spectrum")
    print("  • read_arf_file(filepath)      - อ่าน ARF file")
    print("  • read_rmf_file(filepath)      - อ่าน RMF file")
    print("  • subtract_background(src, bkg) - ลบ background")
    print()
    print("Example usage in Colab:")
    print("  from data_loader import *")
    print("  source = read_spectrum_file('/content/drive/MyDrive/data/source.pha')")
    print("  arf = read_arf_file('/content/drive/MyDrive/data/source.arf')")
