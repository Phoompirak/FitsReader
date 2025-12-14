import streamlit as st
import numpy as np
from astropy.io import fits
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pandas as pd
import spectral_models as sm
import spectral_fitting as sf
from datetime import datetime

st.set_page_config(page_title="X-ray Spectrum Analyzer", layout="wide")


def fix_byte_order(data):
    """Fix byte order for FITS data to avoid Arrow serialization issues (NumPy 2.0 safe)"""
    if hasattr(data, 'dtype') and data.dtype.byteorder not in ('=', '|', '<'):
        # Change byte order safely
        swapped = data.byteswap()
        return swapped.view(swapped.dtype.newbyteorder('='))
    return data


def json_numpy_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    import numpy as np
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def fits_table_to_dataframe(fits_data, max_rows=None):
    """Convert FITS table data to pandas DataFrame with proper byte order"""
    data_dict = {}
    for col_name in fits_data.columns.names:
        col_data = fits_data[col_name]
        if max_rows:
            col_data = col_data[:max_rows]
        data_dict[col_name] = fix_byte_order(col_data)
    return pd.DataFrame(data_dict)


st.title("🔭 X-ray Spectrum Data Analyzer")
st.markdown("### เครื่องมือวิเคราะห์ข้อมูลสเปกตรัม X-ray จาก XMM-Newton")

# JSON file path for storing brute-force results
RESULTS_FILE = Path("data/brute_force_results.json")

def load_brute_force_results():
    """โหลดผลลัพธ์ brute-force จากไฟล์ JSON"""
    try:
        if RESULTS_FILE.exists():
            import json
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.sidebar.warning(f"⚠️ ไม่สามารถโหลดผลลัพธ์: {e}")
    return {"best_results": [], "last_updated": None}

def save_brute_force_result(result_data, run_id=None):
    """
    บันทึกผลลัพธ์ brute-force ลงไฟล์ JSON
    ผลลัพธ์จะถูกเรียงจากค่า chi²/dof ต่ำสุด (ดีที่สุด) ไปสูงสุด
    
    Parameters:
    -----------
    result_data : dict
        Brute-force result data
    run_id : str, optional
        Unique ID for the current run. If provided and matches an existing entry,
        that entry will be updated instead of creating a new one.
    """
    import json
    from datetime import datetime
    
    data = load_brute_force_results()
    
    chi2_value = result_data.get('best_chi2_dof')
    
    # Skip saving if chi² is None or inf
    if chi2_value is None or chi2_value == float('inf'):
        return False
    
    # Add new result with timestamp
    new_result = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "chi2_dof": chi2_value,
        "params": result_data.get('best_params'),
        "n_combinations": result_data.get('total'),
        "model_components": result_data.get('model_components', []),
        "varied_models": result_data.get('varied_models', []),
        "fixed_params": result_data.get('fixed_params', {})
    }
    
    # Check if we should update an existing entry with the same run_id
    updated = False
    if run_id:
        for i, entry in enumerate(data["best_results"]):
            if entry.get("run_id") == run_id:
                # Update existing entry
                data["best_results"][i] = new_result
                updated = True
                break
    
    if not updated:
        # Add new entry
        data["best_results"].append(new_result)
    
    # Sort by chi²/dof value (ascending - lowest/best first)
    def sort_key(x):
        chi2 = x.get('chi2_dof')
        if chi2 is None:
            return float('inf')
        return chi2
    
    data["best_results"].sort(key=sort_key)
    
    # Keep only the best 10 results
    data["best_results"] = data["best_results"][:10]
        
    data["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=json_numpy_serializer)
        return True
    except Exception as e:
        st.error(f"❌ ไม่สามารถบันทึกผลลัพธ์: {e}")
        return False

# Sidebar for file selection
st.sidebar.header("📂 เลือกไฟล์ข้อมูล")

# System Status Section
with st.sidebar.expander("🖥️ System Status (GPU)", expanded=False):
    # GPU Status
    if st.button("ตรวจสอบ GPU"):
        info = sf.check_gpu()
        if info['available']:
            st.success(f"✅ GPU Detected: {info.get('device_name', 'Unknown')}")
            st.write(f"Memory: {info.get('memory_free')} / {info.get('memory_total')}")
        else:
            st.error(f"❌ No GPU Detected: {info.get('error', 'Unknown error')}")
            
    # Benchmark
    if st.button("🚀 Run Benchmark"):
        with st.spinner("Running benchmark..."):
            res = sf.benchmark_gpu()
            if 'error' in res:
                st.error(f"Benchmark Failed: {res['error']}")
            else:
                st.write("**Results (1000 pts, 100 iters):**")
                st.write(f"CPU: {res.get('cpu_time', 0):.4f}s")
                st.write(f"GPU: {res.get('gpu_time', 0):.4f}s")
                speedup = res.get('speedup', 0)
                if speedup > 1:
                    st.success(f"⚡ Speedup: {speedup:.1f}x")
                else:
                    st.warning(f"Speedup: {speedup:.1f}x (GPU might be slower for small tasks)")

# Check for attached files
attached_dir = Path("data/attached_assets")
attached_files = {}

if attached_dir.exists():
    for file in attached_dir.glob("*"):
        if file.suffix in ['.fits', '.arf', '.rmf']:
            attached_files[file.name] = str(file)

# File upload option
upload_option = st.sidebar.radio("เลือกวิธีการโหลดไฟล์:",
                                 ["ใช้ไฟล์ที่แนบมา", "อัพโหลดไฟล์ใหม่"])

# File descriptions
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 คำอธิบายไฟล์:")
st.sidebar.markdown("""
- **FITS (Source):** สเปกตรัม X-ray จากแหล่ง
- **FITS (Background):** สเปกตรัมพื้นหลัง
- **ARF:** ประสิทธิภาพการรับแสง
- **RMF:** การกระจายพลังงาน
""")

# Display saved brute-force results
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 ผลลัพธ์ Brute-Force ที่บันทึกไว้")

saved_results = load_brute_force_results()
if saved_results["best_results"]:
    # Sort results by chi2_dof (ascending) to show best ranks first
    saved_results["best_results"].sort(key=lambda x: x.get('chi2_dof', float('inf')))
    
    for i, result in enumerate(saved_results["best_results"][:5]):
        with st.sidebar.expander(f"#{i+1} χ²/dof = {result['chi2_dof']:.4f}" if result['chi2_dof'] else f"#{i+1} Result"):
            # Parse and display timestamp
            if result.get('timestamp'):
                from datetime import datetime
                try:
                    ts = datetime.fromisoformat(result['timestamp'])
                    st.caption(f"📅 {ts.strftime('%Y-%m-%d %H:%M')}")
                except:
                    pass
            
            # Display parameters
            if result.get('params'):
                st.markdown("**Parameters:**")
                for param, value in result['params'].items():
                    st.text(f"  {param}: {value:.4f}")
            
            # Display model components
            if result.get('model_components'):
                st.caption(f"Models: {', '.join(result['model_components'])}")
            
            # Display varied/fixed info if available
            if result.get('varied_models'):
                st.caption(f"Varied: {', '.join(result['varied_models'])}")
            
            if result.get('fixed_params'):
                with st.expander("Fixed Params", expanded=False):
                    for k, v in result['fixed_params'].items():
                        st.caption(f"{k}: {v:.4f}")

            # Display combinations
            if result.get('n_combinations'):
                st.caption(f"Tested: {result['n_combinations']:,} combinations")
    
    # Clear button
    if st.sidebar.button("🗑️ ลบผลลัพธ์ทั้งหมด", key="clear_results"):
        try:
            import json
            with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump({"best_results": [], "last_updated": None}, f)
            st.sidebar.success("✅ ลบผลลัพธ์แล้ว!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")
else:
    st.sidebar.info("ยังไม่มีผลลัพธ์ที่บันทึกไว้")


def read_fits_file(file_path):
    """Read FITS file and return HDU list"""
    try:
        hdul = fits.open(file_path)
        return hdul
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ได้: {e}")
        return None


def display_fits_header(hdul, hdu_index=1):
    """Display FITS header information"""
    if hdul and len(hdul) > hdu_index:
        header = hdul[hdu_index].header
        st.subheader("📋 Header Information")

        # Convert header to dictionary for display
        header_data = []
        for key in header.keys():
            if key and key.strip():  # Skip empty keys
                value = header[key]
                comment = header.comments[key]
                header_data.append({
                    "Keyword": key,
                    "Value": str(value),
                    "Comment": comment
                })

        if header_data:
            df = pd.DataFrame(header_data)
            st.dataframe(df, width='stretch', height=300)


def plot_spectrum(hdul, title="Spectrum", show_options=True):
    """Plot spectrum data from FITS file"""
    try:
        if len(hdul) > 1:
            data = hdul[1].data

            # Check available columns
            if data is not None:
                st.subheader(f"📊 {title}")

                # Display column names
                col_names = data.columns.names
                st.write("**ข้อมูลที่มีในไฟล์:**", ", ".join(col_names))

                # Create plot based on available data
                if 'CHANNEL' in col_names and 'COUNTS' in col_names:
                    channels = data['CHANNEL']
                    counts = data['COUNTS']

                    # Visualization options
                    if show_options:
                        col_opt1, col_opt2, col_opt3 = st.columns(3)
                        with col_opt1:
                            use_log_y = st.checkbox(
                                "ใช้ Logarithmic Scale (แกน Y)",
                                value=False,
                                key=f"log_{title}")
                        with col_opt2:
                            show_errors = st.checkbox("แสดง Error Bars",
                                                      value=False,
                                                      key=f"err_{title}")
                        with col_opt3:
                            show_markers = st.checkbox("แสดง Markers",
                                                       value=False,
                                                       key=f"mkr_{title}")
                    else:
                        use_log_y = False
                        show_errors = False
                        show_markers = False

                    # Calculate error bars (Poisson statistics: error = sqrt(counts))
                    if show_errors:
                        errors = np.sqrt(np.maximum(
                            counts, 0))  # Avoid sqrt of negative

                    fig = go.Figure()

                    if show_errors:
                        fig.add_trace(
                            go.Scatter(x=channels,
                                       y=counts,
                                       mode='lines+markers'
                                       if show_markers else 'lines',
                                       name='Counts',
                                       line=dict(width=1.5),
                                       error_y=dict(type='data',
                                                    array=errors,
                                                    visible=True,
                                                    color='rgba(0,0,0,0.3)')))
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=channels,
                                y=counts,
                                mode='lines+markers'
                                if show_markers else 'lines',
                                name='Counts',
                                line=dict(width=1.5),
                                marker=dict(size=3) if show_markers else None))

                    yaxis_type = 'log' if use_log_y else 'linear'

                    fig.update_layout(title=title,
                                      xaxis_title="Channel",
                                      yaxis_title="Counts" +
                                      (" (log scale)" if use_log_y else ""),
                                      yaxis_type=yaxis_type,
                                      hovermode='x unified',
                                      template='plotly_white')

                    st.plotly_chart(fig, width='stretch')

                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Counts", f"{np.sum(counts):,.0f}")
                    with col2:
                        st.metric("Mean Counts", f"{np.mean(counts):.2f}")
                    with col3:
                        st.metric("Max Counts", f"{np.max(counts):,.0f}")
                    with col4:
                        st.metric("Channels", len(channels))

                    # Energy band selection and flux calculation
                    if show_options:
                        with st.expander(
                                "⚡ Energy Band Selection & Flux Calculation"):
                            st.write(
                                "เลือกช่วง Channel เพื่อคำนวณ counts และ flux ในแถบพลังงานนั้น"
                            )

                            col_band1, col_band2 = st.columns(2)
                            with col_band1:
                                min_channel = st.number_input(
                                    "Channel ต่ำสุด",
                                    min_value=int(np.min(channels)),
                                    max_value=int(np.max(channels)),
                                    value=int(np.min(channels)),
                                    key=f"min_ch_{title}")
                            with col_band2:
                                max_channel = st.number_input(
                                    "Channel สูงสุด",
                                    min_value=int(np.min(channels)),
                                    max_value=int(np.max(channels)),
                                    value=int(np.max(channels)),
                                    key=f"max_ch_{title}")

                            # Filter data for selected range
                            mask = (channels >= min_channel) & (channels
                                                                <= max_channel)
                            selected_channels = channels[mask]
                            selected_counts = counts[mask]

                            if len(selected_counts) > 0:
                                st.write(
                                    f"**ผลลัพธ์สำหรับ Channels {min_channel} - {max_channel}:**"
                                )

                                col_flux1, col_flux2, col_flux3, col_flux4 = st.columns(
                                    4)
                                with col_flux1:
                                    st.metric(
                                        "Total Counts",
                                        f"{np.sum(selected_counts):,.0f}")
                                with col_flux2:
                                    st.metric(
                                        "Mean Counts",
                                        f"{np.mean(selected_counts):.2f}")
                                with col_flux3:
                                    st.metric(
                                        "Max Counts",
                                        f"{np.max(selected_counts):,.0f}")
                                with col_flux4:
                                    st.metric("Channels Selected",
                                              len(selected_counts))

                                # Simple flux calculation (counts per channel)
                                flux = np.sum(selected_counts) / len(
                                    selected_counts) if len(
                                        selected_counts) > 0 else 0
                                st.write(
                                    f"**Average Flux:** {flux:.2f} counts/channel"
                                )

                                # Plot selected region
                                fig_band = go.Figure()
                                fig_band.add_trace(
                                    go.Scatter(x=channels,
                                               y=counts,
                                               mode='lines',
                                               name='Full Spectrum',
                                               line=dict(width=1,
                                                         color='lightgray'),
                                               opacity=0.5))
                                fig_band.add_trace(
                                    go.Scatter(x=selected_channels,
                                               y=selected_counts,
                                               mode='lines',
                                               name='Selected Band',
                                               line=dict(width=2, color='red'),
                                               fill='tozeroy'))
                                fig_band.update_layout(
                                    title=
                                    f"Selected Energy Band: Channels {min_channel}-{max_channel}",
                                    xaxis_title="Channel",
                                    yaxis_title="Counts",
                                    template='plotly_white',
                                    height=400)
                                st.plotly_chart(fig_band,
                                                width='stretch')
                            else:
                                st.warning("ไม่มีข้อมูลในช่วงที่เลือก")

                    # Show data table and export
                    with st.expander("📊 ดูข้อมูลตาราง และ Export"):
                        df = fits_table_to_dataframe(data)
                        st.dataframe(df, width='stretch', height=300)

                        # Export options
                        st.write("**💾 Export Data:**")
                        col_exp1, col_exp2 = st.columns(2)

                        with col_exp1:
                            csv_data = df.to_csv(index=False)
                            st.download_button(label="📥 Download CSV",
                                               data=csv_data,
                                               file_name="spectrum_data.csv",
                                               mime="text/csv",
                                               key=f"csv_{title}")

                        with col_exp2:
                            # Text format (space-separated)
                            txt_data = df.to_csv(index=False, sep='\t')
                            st.download_button(label="📄 Download TXT",
                                               data=txt_data,
                                               file_name="spectrum_data.txt",
                                               mime="text/plain",
                                               key=f"txt_{title}")
                else:
                    # Display all available data
                    df = fits_table_to_dataframe(data)
                    st.dataframe(df, width='stretch', height=400)

    except Exception as e:
        st.error(f"ไม่สามารถแสดงกราฟได้: {e}")


def plot_arf(file_path):
    """Plot ARF (Ancillary Response File) - Effective Area"""
    try:
        hdul = fits.open(file_path)
        if len(hdul) > 1:
            data = hdul[1].data
            st.subheader("📈 ARF - Effective Area (ประสิทธิภาพการรับแสง)")

            # Display available columns
            col_names = data.columns.names
            st.write("**ข้อมูลที่มีในไฟล์:**", ", ".join(col_names))

            if 'ENERG_LO' in col_names and 'ENERG_HI' in col_names and 'SPECRESP' in col_names:
                energ_lo = data['ENERG_LO']
                energ_hi = data['ENERG_HI']
                specresp = data['SPECRESP']

                # Calculate energy midpoints
                energy = (energ_lo + energ_hi) / 2.0

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=energy,
                               y=specresp,
                               mode='lines',
                               name='Effective Area',
                               line=dict(width=2, color='blue')))

                fig.update_layout(title="ARF: Effective Area vs Energy",
                                  xaxis_title="Energy (keV)",
                                  yaxis_title="Effective Area (cm²)",
                                  hovermode='x unified',
                                  template='plotly_white')

                st.plotly_chart(fig, width='stretch')

                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Energy Range",
                        f"{np.min(energ_lo):.3f} - {np.max(energ_hi):.3f} keV")
                with col2:
                    st.metric("Max Effective Area",
                              f"{np.max(specresp):.2f} cm²")
                with col3:
                    st.metric("Mean Effective Area",
                              f"{np.mean(specresp):.2f} cm²")

                # Export ARF data
                with st.expander("💾 Export ARF Data"):
                    export_df = pd.DataFrame({
                        'ENERGY_LOW_keV':
                        fix_byte_order(energ_lo),
                        'ENERGY_HIGH_keV':
                        fix_byte_order(energ_hi),
                        'ENERGY_MID_keV':
                        fix_byte_order(energy),
                        'EFFECTIVE_AREA_cm2':
                        fix_byte_order(specresp)
                    })

                    st.dataframe(export_df.head(20), width='stretch')

                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        csv_data = export_df.to_csv(index=False)
                        st.download_button(label="📥 Download CSV",
                                           data=csv_data,
                                           file_name="arf_effective_area.csv",
                                           mime="text/csv")
                    with col_exp2:
                        txt_data = export_df.to_csv(index=False, sep='\t')
                        st.download_button(label="📄 Download TXT",
                                           data=txt_data,
                                           file_name="arf_effective_area.txt",
                                           mime="text/plain")

                # Display header
                with st.expander("📋 ดู Header Information"):
                    display_fits_header(hdul, 1)
            else:
                st.write("แสดงข้อมูลที่มีในไฟล์:")
                df = fits_table_to_dataframe(data)
                st.dataframe(df, width='stretch', height=400)

        hdul.close()
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ ARF ได้: {e}")


def plot_rmf(file_path):
    """Plot RMF (Response Matrix File) - Energy Redistribution"""
    try:
        hdul = fits.open(file_path)
        st.subheader("🔲 RMF - Response Matrix (การกระจายพลังงาน)")

        # Display file structure
        st.write("**โครงสร้างไฟล์ RMF:**")
        for i, hdu in enumerate(hdul):
            st.write(f"HDU {i}: {hdu.name} ({type(hdu).__name__})")

        # Try to read EBOUNDS extension (HDU 1) for energy information
        if len(hdul) > 1 and 'EBOUNDS' in [hdu.name for hdu in hdul]:
            ebounds_idx = [
                i for i, hdu in enumerate(hdul) if hdu.name == 'EBOUNDS'
            ][0]
            ebounds_data = hdul[ebounds_idx].data

            st.write("### 📊 EBOUNDS Extension (Energy Boundaries)")
            col_names = ebounds_data.columns.names
            st.write("**คอลัมน์ที่มี:**", ", ".join(col_names))

            if 'E_MIN' in col_names and 'E_MAX' in col_names:
                e_min = ebounds_data['E_MIN']
                e_max = ebounds_data['E_MAX']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("จำนวน Energy Channels", len(e_min))
                with col2:
                    st.metric("Energy Range (min)", f"{np.min(e_min):.3f} keV")
                with col3:
                    st.metric("Energy Range (max)", f"{np.max(e_max):.3f} keV")

                with st.expander("📋 ดู EBOUNDS Data"):
                    df_ebounds = fits_table_to_dataframe(ebounds_data,
                                                         max_rows=20)
                    st.dataframe(df_ebounds, width='stretch')

        # Try to read MATRIX extension (usually HDU 2) for response matrix
        matrix_found = False
        if len(hdul) > 2:
            for i, hdu in enumerate(hdul):
                if hdu.name in ['MATRIX', 'SPECRESP MATRIX'] or i == 2:
                    try:
                        matrix_data = hdul[i].data
                        if matrix_data is not None:
                            st.write(
                                f"### 🔲 Response Matrix Extension (HDU {i}: {hdu.name})"
                            )

                            col_names = matrix_data.columns.names
                            st.write("**คอลัมน์ที่มี:**", ", ".join(col_names))

                            # Check for matrix column
                            matrix_col = None
                            for possible_name in [
                                    'MATRIX', 'SPECRESP MATRIX', 'F_CHAN',
                                    'RESPONSE'
                            ]:
                                if possible_name in col_names:
                                    matrix_col = possible_name
                                    break

                            if matrix_col:
                                st.success(
                                    f"✅ พบ Response Matrix ใน column '{matrix_col}'"
                                )

                                # Get energy information
                                if 'ENERG_LO' in col_names and 'ENERG_HI' in col_names:
                                    energ_lo = matrix_data['ENERG_LO']
                                    energ_hi = matrix_data['ENERG_HI']
                                    energy_mid = (energ_lo + energ_hi) / 2.0

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("จำนวน Energy Bins",
                                                  len(energ_lo))
                                    with col2:
                                        st.metric(
                                            "Energy Range",
                                            f"{np.min(energ_lo):.3f} - {np.max(energ_hi):.3f} keV"
                                        )

                                    # Try to visualize matrix elements
                                    st.write("**การแสดงผล Response Matrix:**")

                                    # Extract matrix values for visualization
                                    try:
                                        # Build a simplified visualization of the matrix
                                        # For each energy bin, get the redistribution response
                                        matrix_values = matrix_data[matrix_col]

                                        # Sample a subset for visualization (matrix can be very large)
                                        sample_size = min(
                                            50, len(matrix_values))
                                        sample_indices = np.linspace(
                                            0,
                                            len(matrix_values) - 1,
                                            sample_size,
                                            dtype=int)

                                        # Create a 2D array for heatmap
                                        max_channels = max([
                                            len(row) if hasattr(
                                                row, '__len__') else 1 for row
                                            in matrix_values[sample_indices]
                                        ])
                                        max_channels = min(
                                            max_channels,
                                            100)  # Limit for visualization

                                        matrix_2d = np.zeros(
                                            (sample_size, max_channels))
                                        for i, idx in enumerate(
                                                sample_indices):
                                            row = matrix_values[idx]
                                            if hasattr(row, '__len__'):
                                                length = min(
                                                    len(row), max_channels)
                                                matrix_2d[
                                                    i, :length] = row[:length]
                                            else:
                                                matrix_2d[i, 0] = row

                                        # Create heatmap
                                        fig = go.Figure(data=go.Heatmap(
                                            z=matrix_2d,
                                            x=list(range(max_channels)),
                                            y=energy_mid[sample_indices],
                                            colorscale='Viridis',
                                            colorbar=dict(title="Response")))

                                        fig.update_layout(
                                            title=
                                            "Response Matrix Heatmap (Sampled)",
                                            xaxis_title="PHA Channel",
                                            yaxis_title="Energy (keV)")

                                        st.plotly_chart(
                                            fig, width='stretch')

                                        st.info(
                                            "ℹ️ Heatmap แสดงความสัมพันธ์ระหว่างพลังงานจริงของ photon (แกน Y) กับช่องพลังงานที่ตรวจจับได้ (แกน X)"
                                        )

                                    except Exception as e:
                                        st.warning(
                                            f"ไม่สามารถสร้าง heatmap ได้: {e}")
                                        st.write("แสดงข้อมูลตัวอย่างแทน:")
                                        df_sample = fits_table_to_dataframe(
                                            matrix_data, max_rows=10)
                                        st.dataframe(df_sample,
                                                     width='stretch')

                                # Display header
                                with st.expander("📋 ดู Header Information"):
                                    display_fits_header(hdul, i)

                                matrix_found = True
                                break
                            else:
                                # Show available data even without matrix column
                                df = fits_table_to_dataframe(matrix_data,
                                                             max_rows=20)
                                st.dataframe(df, width='stretch')

                    except Exception as e:
                        st.warning(f"ไม่สามารถอ่าน HDU {i} ได้: {e}")
                        continue

        if not matrix_found:
            st.warning("⚠️ ไม่พบ Response Matrix extension ในไฟล์นี้")
            st.info(
                "ℹ️ Response Matrix File (RMF) ปกติจะมีข้อมูลความสัมพันธ์ระหว่างพลังงานจริงของ photon กับช่องพลังงานที่กล้องวัดได้"
            )

        hdul.close()
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ RMF ได้: {e}")


# Main application
if upload_option == "ใช้ไฟล์ที่แนบมา":
    if attached_files:
        st.sidebar.success(f"พบไฟล์ที่แนบมา {len(attached_files)} ไฟล์")

        # Categorize files
        source_files = [
            f for f in attached_files.keys()
            if 'source_spectrum' in f and f.endswith('.fits')
        ]
        bkg_files = [f for f in attached_files.keys() if 'bkg_spectrum' in f]
        arf_files = [f for f in attached_files.keys() if f.endswith('.arf')]
        rmf_files = [f for f in attached_files.keys() if f.endswith('.rmf')]

        # Create tabs for different file types
        tabs = st.tabs([
            "📊 Source Spectrum", "🌌 Background Spectrum",
            "🔬 Background Subtraction", "📈 ARF File", "🔲 RMF File",
            "🎯 Spectral Fitting Analysis"
        ])

        # Tab 1: Source Spectrum
        with tabs[0]:
            if source_files:
                selected_source = st.selectbox("เลือก Source Spectrum File:",
                                               source_files)
                if selected_source:
                    file_path = attached_files[selected_source]
                    st.write(f"**ไฟล์:** `{selected_source}`")

                    hdul = read_fits_file(file_path)
                    if hdul:
                        # Display file structure
                        st.write("**โครงสร้างไฟล์:**")
                        for i, hdu in enumerate(hdul):
                            st.write(
                                f"HDU {i}: {hdu.name} ({type(hdu).__name__})")

                        # Plot spectrum
                        plot_spectrum(
                            hdul, "Source Spectrum - สเปกตรัมจากแหล่ง X-ray")

                        # Display header
                        with st.expander("📋 ดู Header Information"):
                            display_fits_header(hdul, 1)

                        hdul.close()
            else:
                st.warning("ไม่พบไฟล์ Source Spectrum")

        # Tab 2: Background Spectrum
        with tabs[1]:
            if bkg_files:
                selected_bkg = st.selectbox("เลือก Background Spectrum File:",
                                            bkg_files)
                if selected_bkg:
                    file_path = attached_files[selected_bkg]
                    st.write(f"**ไฟล์:** `{selected_bkg}`")

                    hdul = read_fits_file(file_path)
                    if hdul:
                        # Display file structure
                        st.write("**โครงสร้างไฟล์:**")
                        for i, hdu in enumerate(hdul):
                            st.write(
                                f"HDU {i}: {hdu.name} ({type(hdu).__name__})")

                        # Plot spectrum
                        plot_spectrum(
                            hdul, "Background Spectrum - สเปกตรัมพื้นหลัง")

                        # Display header
                        with st.expander("📋 ดู Header Information"):
                            display_fits_header(hdul, 1)

                        hdul.close()
            else:
                st.warning("ไม่พบไฟล์ Background Spectrum")

        # Tab 3: Background Subtraction
        with tabs[2]:
            if source_files and bkg_files:
                st.subheader("🔬 Background Subtraction Analysis")
                st.write(
                    "ลบสัญญาณพื้นหลัง (background) ออกจากสเปกตรัมแหล่ง (source) เพื่อดูสัญญาณที่แท้จริง"
                )

                col1, col2 = st.columns(2)
                with col1:
                    selected_source_sub = st.selectbox(
                        "เลือก Source Spectrum:",
                        source_files,
                        key="source_sub")
                with col2:
                    selected_bkg_sub = st.selectbox(
                        "เลือก Background Spectrum:", bkg_files, key="bkg_sub")

                if selected_source_sub and selected_bkg_sub:
                    try:
                        # Read source spectrum
                        source_hdul = read_fits_file(
                            attached_files[selected_source_sub])
                        bkg_hdul = read_fits_file(
                            attached_files[selected_bkg_sub])

                        if source_hdul and bkg_hdul and len(
                                source_hdul) > 1 and len(bkg_hdul) > 1:
                            source_data = source_hdul[1].data
                            bkg_data = bkg_hdul[1].data

                            if ('CHANNEL' in source_data.columns.names
                                    and 'COUNTS' in source_data.columns.names
                                    and 'CHANNEL' in bkg_data.columns.names
                                    and 'COUNTS' in bkg_data.columns.names):

                                source_channels = source_data['CHANNEL']
                                source_counts = source_data['COUNTS']
                                bkg_counts = bkg_data['COUNTS']

                                # Ensure arrays are compatible
                                min_len = min(len(source_counts),
                                              len(bkg_counts))
                                source_channels = source_channels[:min_len]
                                source_counts = source_counts[:min_len]
                                bkg_counts = bkg_counts[:min_len]

                                # Calculate background-subtracted spectrum
                                subtracted_counts = source_counts - bkg_counts

                                # Create comparison plot
                                fig = go.Figure()

                                fig.add_trace(
                                    go.Scatter(x=source_channels,
                                               y=source_counts,
                                               mode='lines',
                                               name='Source (แหล่ง)',
                                               line=dict(width=1.5,
                                                         color='blue'),
                                               opacity=0.7))

                                fig.add_trace(
                                    go.Scatter(x=source_channels,
                                               y=bkg_counts,
                                               mode='lines',
                                               name='Background (พื้นหลัง)',
                                               line=dict(width=1.5,
                                                         color='red'),
                                               opacity=0.7))

                                fig.add_trace(
                                    go.Scatter(x=source_channels,
                                               y=subtracted_counts,
                                               mode='lines',
                                               name='Subtracted (ลบพื้นหลัง)',
                                               line=dict(width=2,
                                                         color='green')))

                                fig.update_layout(
                                    title="Background Subtraction Comparison",
                                    xaxis_title="Channel",
                                    yaxis_title="Counts",
                                    hovermode='x unified',
                                    template='plotly_white',
                                    height=500)

                                st.plotly_chart(fig, width='stretch')

                                # Display statistics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Source Total",
                                              f"{np.sum(source_counts):,.0f}")
                                with col2:
                                    st.metric("Background Total",
                                              f"{np.sum(bkg_counts):,.0f}")
                                with col3:
                                    st.metric(
                                        "Subtracted Total",
                                        f"{np.sum(subtracted_counts):,.0f}")
                                with col4:
                                    bkg_fraction = (
                                        np.sum(bkg_counts) /
                                        np.sum(source_counts)) * 100 if np.sum(
                                            source_counts) > 0 else 0
                                    st.metric("Background %",
                                              f"{bkg_fraction:.1f}%")

                                # Show subtracted spectrum only
                                st.subheader(
                                    "📊 Background-Subtracted Spectrum")

                                fig2 = go.Figure()
                                fig2.add_trace(
                                    go.Scatter(x=source_channels,
                                               y=subtracted_counts,
                                               mode='lines',
                                               name='Background-Subtracted',
                                               line=dict(width=2,
                                                         color='darkgreen'),
                                               fill='tozeroy',
                                               fillcolor='rgba(0,100,0,0.2)'))

                                fig2.update_layout(
                                    title="Background-Subtracted Spectrum",
                                    xaxis_title="Channel",
                                    yaxis_title="Net Counts",
                                    hovermode='x unified',
                                    template='plotly_white')

                                st.plotly_chart(fig2, width='stretch')

                                # Export option
                                with st.expander(
                                        "💾 Export Background-Subtracted Data"):
                                    export_df = pd.DataFrame({
                                        'CHANNEL':
                                        fix_byte_order(source_channels),
                                        'SOURCE_COUNTS':
                                        fix_byte_order(source_counts),
                                        'BACKGROUND_COUNTS':
                                        fix_byte_order(bkg_counts),
                                        'NET_COUNTS':
                                        fix_byte_order(subtracted_counts)
                                    })

                                    csv_data = export_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download as CSV",
                                        data=csv_data,
                                        file_name=
                                        "background_subtracted_spectrum.csv",
                                        mime="text/csv")

                                    st.dataframe(export_df.head(20),
                                                 width='stretch')
                            else:
                                st.error(
                                    "ไฟล์ไม่มีคอลัมน์ CHANNEL และ COUNTS ที่จำเป็น"
                                )

                            source_hdul.close()
                            bkg_hdul.close()
                        else:
                            st.error("ไม่สามารถอ่านไฟล์สเปกตรัมได้")

                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
            else:
                st.warning(
                    "⚠️ ต้องมีทั้ง Source และ Background Spectrum เพื่อทำ Background Subtraction"
                )
                if not source_files:
                    st.info("ไม่พบไฟล์ Source Spectrum")
                if not bkg_files:
                    st.info("ไม่พบไฟล์ Background Spectrum")

        # Tab 4: ARF
        with tabs[3]:
            if arf_files:
                selected_arf = st.selectbox("เลือก ARF File:", arf_files)
                if selected_arf:
                    file_path = attached_files[selected_arf]
                    st.write(f"**ไฟล์:** `{selected_arf}`")
                    plot_arf(file_path)
            else:
                st.warning("ไม่พบไฟล์ ARF")

        # Tab 5: RMF
        with tabs[4]:
            if rmf_files:
                selected_rmf = st.selectbox("เลือก RMF File:", rmf_files)
                if selected_rmf:
                    file_path = attached_files[selected_rmf]
                    st.write(f"**ไฟล์:** `{selected_rmf}`")
                    plot_rmf(file_path)
            else:
                st.warning("ไม่พบไฟล์ RMF")
        
        # Tab 6: Spectral Fitting Analysis
        with tabs[5]:
            st.subheader("🎯 Spectral Fitting Analysis")
            st.markdown("### การฟิตสเปกตรัม X-ray ของ Fairall 9")
            
            if source_files and arf_files:
                st.info("ℹ️ **หมายเหตุ:** การฟิตสเปกตรัมต้องใช้เวลาในการคำนวณ กรุณารอสักครู่หลังกดปุ่ม 'เริ่มการฟิต'")
                
                # File selection
                col1, col2 = st.columns(2)
                with col1:
                    selected_spec = st.selectbox("เลือก Source Spectrum:", source_files, key="fit_source")
                with col2:
                    selected_arf_fit = st.selectbox("เลือก ARF File:", arf_files, key="fit_arf")
                
                # Background Subtraction Option
                st.markdown("---")
                st.write("**Background Subtraction (การลบพื้นหลัง):**")
                use_bkg_sub = st.checkbox("✅ Enable Background Subtraction", value=True, 
                                          help="ลบสัญญาณ Background ออกจาก Source เพื่อความแม่นยำ")
                
                selected_bkg_fit = None
                if use_bkg_sub:
                    if bkg_files:
                        selected_bkg_fit = st.selectbox("เลือก Background Spectrum:", bkg_files, key="fit_bkg")
                    else:
                        st.warning("⚠️ ไม่พบไฟล์ Background (.fits) กรุณาแนบไฟล์")
                
                # Model selection
                st.markdown("---")
                st.markdown("### 🔧 เลือก Spectral Models")
                st.write("เลือก components ทางฟิสิกส์ที่ต้องการรวมในโมเดล:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    use_powerlaw = st.checkbox("✅ Power-law Continuum", value=True, 
                                              help="X-ray continuum จาก Comptonization")
                    use_absorption = st.checkbox("Photoelectric Absorption (tbabs)", value=True,
                                                help="การดูดกลืนโดย neutral hydrogen")
                with col2:
                    use_reflection = st.checkbox("X-ray Reflection", value=True,
                                                help="รังสีเอกซ์ที่สะท้อนจาก accretion disk")
                    use_gaussian = st.checkbox("Gaussian Line (Fe Kα)", value=True,
                                              help="เส้นสเปกตรัมจาก iron fluorescence")
                with col3:
                    use_blackbody = st.checkbox("Blackbody (Thermal)", value=False,
                                               help="Thermal emission จาก accretion disk")
                
                # Build model components list
                model_components = []
                if use_powerlaw:
                    model_components.append('powerlaw')
                if use_absorption:
                    model_components.append('tbabs')
                if use_reflection:
                    model_components.append('reflection')
                if use_gaussian:
                    model_components.append('gaussian')
                if use_blackbody:
                    model_components.append('blackbody')
                
                if not model_components:
                    st.warning("⚠️ กรุณาเลือก spectral model อย่างน้อย 1 ตัว")
                else:
                    st.success(f"✅ เลือกโมเดล: {', '.join(model_components)}")
                    
                    # Show model descriptions
                    with st.expander("📖 คำอธิบาย Spectral Models"):
                        for comp in model_components:
                            desc = sm.get_model_description(comp)
                            if desc:
                                st.markdown(f"**{desc.get('name', comp)}**")
                                st.write(f"- *องค์ประกอบทางกายภาพ:* {desc.get('physics', 'N/A')}")
                                if 'parameters' in desc:
                                    st.write("- *พารามิเตอร์:*")
                                    for param, param_desc in desc['parameters'].items():
                                        st.write(f"  - `{param}`: {param_desc}")
                                st.markdown("---")
                    
                    # Energy Range Selection
                    st.markdown("### 📐 เลือกช่วงพลังงาน (Energy Range)")
                    st.write("เลือกช่วงพลังงานที่ต้องการใช้ในการฟิต:")
                    
                    energy_col1, energy_col2 = st.columns(2)
                    with energy_col1:
                        energy_min = st.slider(
                            "Energy ต่ำสุด (keV)", 
                            min_value=0.1, max_value=5.0, value=0.3, step=0.1,
                            key="energy_min",
                            help="พลังงานต่ำสุดที่ใช้ในการฟิต (ค่าทั่วไป: 0.3 keV)")
                    with energy_col2:
                        energy_max = st.slider(
                            "Energy สูงสุด (keV)", 
                            min_value=2.0, max_value=15.0, value=10.0, step=0.5,
                            key="energy_max",
                            help="พลังงานสูงสุดที่ใช้ในการฟิต (ค่าทั่วไป: 10 keV)")
                    
                    st.info(f"📊 ช่วงพลังงานที่เลือก: **{energy_min:.1f} - {energy_max:.1f} keV**")
                    
                    st.markdown("---")
                    
                    # Parameter settings
                    st.markdown("### ⚙️ ตั้งค่าพารามิเตอร์เริ่มต้น")
                    
                    initial_params = {}
                    
                    col1, col2 = st.columns(2)
                    
                    if use_powerlaw:
                        with col1:
                            st.markdown("**Power-law Parameters:**")
                            initial_params['pl_norm'] = st.number_input(
                                "Normalization", value=0.01, min_value=0.0001, 
                                max_value=10.0, format="%.4f", step=0.0001, key="pl_norm")
                            initial_params['photon_index'] = st.number_input(
                                "Photon Index (Γ)", value=2.0, min_value=1.0, 
                                max_value=3.0, format="%.4f", step=0.01, key="photon_idx")
                    
                    if use_absorption:
                        with col2:
                            st.markdown("**Absorption Parameters:**")
                            initial_params['nH'] = st.number_input(
                                "nH (10²² cm⁻²)", value=0.05, min_value=0.0, 
                                max_value=10.0, format="%.3f", key="nH")
                    
                    if use_reflection:
                        with col1:
                            st.markdown("**Reflection Parameters:**")
                            initial_params['refl_norm'] = st.number_input(
                                "Reflection Norm", value=0.5, min_value=0.0, 
                                max_value=5.0, format="%.2f", key="refl_norm")
                    
                    if use_gaussian:
                        with col2:
                            st.markdown("**Gaussian Line Parameters:**")
                            initial_params['line_energy'] = st.number_input(
                                "Line Energy (keV)", value=6.4, min_value=6.0, 
                                max_value=7.0, format="%.2f", key="line_e")
                            initial_params['line_sigma'] = st.number_input(
                                "Line Width σ (keV)", value=0.1, min_value=0.01, 
                                max_value=0.5, format="%.2f", key="line_sig")
                            initial_params['line_norm'] = st.number_input(
                                "Line Norm", value=1.0, min_value=0.0, 
                                max_value=100.0, format="%.2f", key="line_norm")
                    
                    if use_blackbody:
                        with col1:
                            st.markdown("**Blackbody Parameters:**")
                            initial_params['bb_norm'] = st.number_input(
                                "BB Normalization", value=0.1, min_value=0.0, 
                                max_value=10.0, format="%.2f", key="bb_norm")
                            initial_params['kT'] = st.number_input(
                                "kT (keV)", value=0.5, min_value=0.05, 
                                max_value=3.0, format="%.2f", key="kT")
                    
                    st.markdown("---")
                    
                    # Auto-estimate button
                    st.markdown("### 🔄 ประมาณค่าพารามิเตอร์อัตโนมัติ")
                    if st.button("🔄 คำนวณค่าเริ่มต้นจากข้อมูล", key="auto_estimate"):
                        try:
                            # Load data for estimation
                            spec_path = attached_files[selected_spec]
                            arf_path = attached_files[selected_arf_fit]
                            
                            spectrum = sf.read_spectrum_file(spec_path)
                            arf_data = sf.read_arf_file(arf_path)
                            
                            if spectrum is not None and arf_data is not None:
                                energy = arf_data.energy_mid
                                observed_rate = spectrum.count_rate()
                                
                                # Filter to selected energy range
                                min_len = min(len(energy), len(observed_rate))
                                energy = energy[:min_len]
                                observed_rate = observed_rate[:min_len]
                                
                                energy_mask = (energy > energy_min) & (energy < energy_max)
                                energy_filtered = energy[energy_mask]
                                rate_filtered = observed_rate[energy_mask]
                                
                                # Get estimated parameters
                                estimated = sf.auto_estimate_parameters(
                                    energy_filtered, rate_filtered, model_components)
                                
                                # Display estimated values
                                st.success("✅ ประมาณค่าเสร็จสิ้น! กรุณาคัดลอกค่าด้านล่างไปใส่ในช่อง input ด้านบน:")
                                
                                est_col1, est_col2 = st.columns(2)
                                with est_col1:
                                    if 'powerlaw' in model_components:
                                        st.write(f"**Normalization:** `{estimated['pl_norm']:.4f}`")
                                        st.write(f"**Photon Index:** `{estimated['photon_index']:.2f}`")
                                    if 'reflection' in model_components:
                                        st.write(f"**Reflection Norm:** `{estimated['refl_norm']:.2f}`")
                                with est_col2:
                                    if 'tbabs' in model_components:
                                        st.write(f"**nH:** `{estimated['nH']:.3f}`")
                                    if 'gaussian' in model_components:
                                        st.write(f"**Line Energy:** `{estimated['line_energy']:.2f}` keV")
                                        st.write(f"**Line Sigma:** `{estimated['line_sigma']:.2f}` keV")
                                        st.write(f"**Line Norm:** `{estimated['line_norm']:.2f}`")
                            else:
                                st.error("❌ ไม่สามารถอ่านไฟล์ข้อมูลได้")
                        except Exception as e:
                            st.error(f"❌ เกิดข้อผิดพลาด: {e}")
                    
                    st.markdown("---")
                    
                    # Brute-force optimization section
                    st.markdown("### 🔥 Brute-Force Optimization")
                    st.write("ค้นหาค่าพารามิเตอร์ที่ดีที่สุดโดยการทดสอบทุกๆ combinations:")

                    # Select models to vary
                    varied_models = st.multiselect(
                        "เลือกโมเดลที่ต้องการทำ Brute Force (โมเดลที่ไม่เลือกจะถูก fix ไว้ที่ค่าเริ่มต้น):",
                        options=model_components,
                        default=model_components
                    )

                    
                    bf_col1, bf_col2, bf_col3 = st.columns(3)
                    with bf_col1:
                        bf_steps = st.slider(
                            "จำนวน Steps ต่อ Parameter", 
                            min_value=3, max_value=15, value=5, step=1,
                            key="bf_steps",
                            help="จำนวน combinations ทั้งหมด = steps^n_params (เช่น 5^2 = 25, 5^3 = 125)")
                    
                    with bf_col2:
                        # Computation Mode Selection
                        st.markdown("**Computation Mode:**")
                        comp_mode = st.radio(
                            "เลือกโหมดการคำนวณ:",
                            ["CPU (Parallel)", "GPU (CUDA)", "CPU (Sequential)"],
                            index=1 if sf.HAS_GPU else 0, # Default to GPU if available
                            help="GPU เร็วที่สุดสำหรับ grid ขนาดใหญ่, CPU Parallel ใช้ทุก cores"
                        )
                        
                        n_workers = 1
                        if comp_mode == "CPU (Parallel)":
                             import multiprocessing
                             max_cpus = multiprocessing.cpu_count()
                             n_workers = st.slider(
                                "จำนวน Workers", 
                                min_value=1, max_value=max_cpus, value=max(1, max_cpus - 1),
                                key="n_workers",
                                help="จำนวน CPU cores ที่จะใช้")
                        elif comp_mode == "GPU (CUDA)":
                            if not sf.HAS_GPU:
                                st.error("⚠️ ไม่พบ GPU (CuPy) ระบบจะใช้ CPU แทน")
                                comp_mode = "CPU (Sequential)"
                            else:
                                st.success("🚀 พร้อมใช้งาน GPU Acceleration")
                    
                    # Calculate total combinations
                    active_params = 0
                    if 'powerlaw' in varied_models:
                        active_params += 2  # pl_norm, photon_index
                    if 'tbabs' in varied_models:
                        active_params += 1  # nH
                    if 'reflection' in varied_models:
                        active_params += 1  # refl_norm
                    if 'gaussian' in varied_models:
                        active_params += 3  # line_energy, line_sigma, line_norm
                    if 'blackbody' in varied_models:
                        active_params += 2 # bb_norm, kT

                    total_combos = bf_steps ** active_params if active_params > 0 else 0

                    
                    with bf_col3:
                        st.metric("Total Combinations", f"{total_combos:,}")
                        if comp_mode == "GPU (CUDA)":
                             # Estimate roughly 1M combos per second on GPU (very rough)
                             est_time = total_combos / 1_000_000 
                             st.caption(f"⚡ ~{est_time:.2f}s (GPU)")
                        elif comp_mode == "CPU (Parallel)" and n_workers > 1:
                            est_time = total_combos * 0.005 / n_workers
                            st.caption(f"⚡ ~{est_time:.1f}s (CPU {n_workers}x)")
                        else:
                            st.caption(f"~{total_combos * 0.01:.1f}s (CPU Seq)")
                    
                    # Build parameter ranges and fixed params
                    param_ranges = {}
                    fixed_params = {}
                    
                    # Dynamic Parameter Ranges Config
                    st.markdown("---")
                    with st.expander("⚙️ ตั้งค่าช่วงการค้นหา (Brute Force Ranges)", expanded=True):
                        
                        # Auto-Detect Toggle
                        use_auto_ranges = st.checkbox("⚡ ใช้ช่วงการค้นหาอัตโนมัติ (Wide Search + Energy Opt)", value=False, help="ระบบจะกำหนดช่วงกว้างๆ ให้เอง และค้นหาช่วงพลังงานที่ดีที่สุด (energy_min/max) โดยอัตโนมัติ")
                        
                        if use_auto_ranges:
                            st.success("✅ **Auto Mode Active**: ระบบจะค้นหาพารามิเตอร์ในช่วงกว้างและปรับช่วงพลังงานอัตโนมัติ (0.1-12.0 keV)")
                        else:
                            st.info("กำหนดช่วง Min/Max สำหรับพารามิเตอร์ที่ต้องการ vary (ค่า step จะถูกคำนวณอัตโนมัติจากจำนวน steps)")
                        
                        # Create columns for better layout
                        r_col1, r_col2 = st.columns(2)
                        
                        # Helper to create range inputs
                        def create_range_input(label, key_prefix, default_min, default_max, format="%.4f", step=0.01):
                            if use_auto_ranges:
                                return (default_min, default_max)
                            
                            c1, c2 = st.columns(2)
                            min_val = c1.number_input(f"Min {label}", value=float(default_min), format=format, step=step, key=f"min_{key_prefix}")
                            max_val = c2.number_input(f"Max {label}", value=float(default_max), format=format, step=step, key=f"max_{key_prefix}")
                            return (min_val, max_val)

                        if use_powerlaw and 'powerlaw' in varied_models:
                            st.markdown("**Power-law**")
                            d_min, d_max = (0.0001, 10.0) if use_auto_ranges else (0.001, 1.0)
                            param_ranges['pl_norm'] = create_range_input("Norm", "pl_norm", d_min, d_max, format="%.4f", step=0.0001)
                            
                            d_min, d_max = (0.5, 4.0) if use_auto_ranges else (1.2, 2.8)
                            param_ranges['photon_index'] = create_range_input("Index", "pho_idx", d_min, d_max)
                        elif use_powerlaw:
                            fixed_params['pl_norm'] = initial_params.get('pl_norm', 0.01)
                            fixed_params['photon_index'] = initial_params.get('photon_index', 2.0)
                            
                        if use_absorption:
                            if 'tbabs' in varied_models:
                                st.markdown("**Absorption**")
                                d_min, d_max = (0.0, 5.0) if use_auto_ranges else (0.01, 1.0)
                                param_ranges['nH'] = create_range_input("nH", "nH", d_min, d_max)
                            else:
                                fixed_params['nH'] = initial_params.get('nH', 0.05)
                                
                        if use_reflection:
                            if 'reflection' in varied_models:
                                st.markdown("**Reflection**")
                                d_min, d_max = (0.01, 10.0) if use_auto_ranges else (0.1, 2.0)
                                param_ranges['refl_norm'] = create_range_input("Refl Norm", "refl", d_min, d_max)
                            else:
                                fixed_params['refl_norm'] = initial_params.get('refl_norm', 0.5)
                                
                        if use_gaussian:
                            if 'gaussian' in varied_models:
                                st.markdown("**Gaussian Line**")
                                d_min, d_max = (3.0, 9.0) if use_auto_ranges else (6.2, 6.6)
                                param_ranges['line_energy'] = create_range_input("Energy (keV)", "line_e", d_min, d_max)
                                
                                d_min, d_max = (0.01, 2.0) if use_auto_ranges else (0.05, 0.3)
                                param_ranges['line_sigma'] = create_range_input("Sigma (keV)", "line_s", d_min, d_max)
                                
                                d_min, d_max = (0.01, 10.0) if use_auto_ranges else (0.1, 5.0)
                                param_ranges['line_norm'] = create_range_input("Norm", "line_n", d_min, d_max)
                            else:
                                fixed_params['line_energy'] = initial_params.get('line_energy', 6.4)
                                fixed_params['line_sigma'] = initial_params.get('line_sigma', 0.1)
                                fixed_params['line_norm'] = initial_params.get('line_norm', 1.0)
                                
                        if use_blackbody:
                            if 'blackbody' in varied_models:
                                st.markdown("**Blackbody**")
                                d_min, d_max = (0.01, 20.0) if use_auto_ranges else (0.1, 10.0)
                                param_ranges['bb_norm'] = create_range_input("BB Norm", "bb_n", d_min, d_max)
                                
                                d_min, d_max = (0.01, 5.0) if use_auto_ranges else (0.05, 1.5)
                                param_ranges['kT'] = create_range_input("kT (keV)", "bb_kt", d_min, d_max)
                            else:
                                fixed_params['bb_norm'] = initial_params.get('bb_norm', 1.0)
                                fixed_params['kT'] = initial_params.get('kT', 0.1)

                        # Inject Energy Search Ranges if Auto Mode
                        if use_auto_ranges:
                            param_ranges['energy_min'] = (0.1, 3.0)
                            param_ranges['energy_max'] = (3.0, 15.0)

                    if st.button("🔥 เริ่ม Brute-Force Search", key="brute_force_btn"):
                        if total_combos > 10000:
                            st.warning(f"⚠️ จำนวน combinations สูงมาก ({total_combos:,}) อาจใช้เวลานาน!")
                        
                        try:
                            # Generate Run ID for auto-save
                            import uuid
                            run_id = str(uuid.uuid4())
                            st.info(f"💾 เริ่มต้นการ Auto-save... ระบบจะบันทึกค่าที่ดีที่สุดให้ตลอดเวลา")
                            st.warning("💡 **สามารถกดปุ่ม Stop (บนขวา) ได้ทุกเมื่อ** ค่าที่ดีที่สุดจะถูกบันทึกไว้ในแถบข้างเสมอ")
                            
                            # Load data
                            spec_path = attached_files[selected_spec]
                            arf_path = attached_files[selected_arf_fit]
                            
                            spectrum = sf.read_spectrum_file(spec_path)
                            arf_data = sf.read_arf_file(arf_path)
                            
                            if spectrum is not None and arf_data is not None:
                                # Prepare data
                                energy = arf_data.energy_mid
                                observed_rate = spectrum.count_rate()
                                observed_error = spectrum.count_rate_error()
                                
                                # Background Subtraction Logic
                                if use_bkg_sub and selected_bkg_fit:
                                    try:
                                        bkg_path = attached_files[selected_bkg_fit]
                                        bkg_spectrum = sf.read_spectrum_file(bkg_path)
                                        if bkg_spectrum:
                                            bkg_rate = bkg_spectrum.count_rate()
                                            
                                            # Match lengths
                                            min_len_bkg = min(len(observed_rate), len(bkg_rate))
                                            # Subtract background
                                            # Note: Error propagation: sqrt(err_src^2 + err_bkg^2)
                                            # But simplifying here: New Rate = Src Rate - Bkg Rate
                                            
                                            # Ensure non-negative rate (optional but good for physical consistency)
                                            observed_rate[:min_len_bkg] = np.maximum(observed_rate[:min_len_bkg] - bkg_rate[:min_len_bkg], 0.0)
                                            
                                            # Update errors (add in quadrature)
                                            bkg_error = bkg_spectrum.count_rate_error()
                                            observed_error[:min_len_bkg] = np.sqrt(observed_error[:min_len_bkg]**2 + bkg_error[:min_len_bkg]**2)
                                            
                                            st.success(f"✅ Applied Background Subtraction using {selected_bkg_fit}")
                                    except Exception as e:
                                        st.error(f"⚠️ Background Subtraction Failed: {e}")

                                min_len = min(len(energy), len(observed_rate))
                                energy = energy[:min_len]
                                observed_rate = observed_rate[:min_len]
                                observed_error = observed_error[:min_len]
                                
                                if use_auto_ranges:
                                    # Auto Mode: Use wide/full range (e.g. 0.1 - 15.0) so Brute Force can explore
                                    safe_min, safe_max = 0.1, 15.0
                                    energy_mask = (energy > safe_min) & (energy < safe_max)
                                else:
                                    # Manual Mode: Use slider values
                                    energy_mask = (energy > energy_min) & (energy < energy_max)
                                
                                energy = energy[energy_mask]
                                observed_rate = observed_rate[energy_mask]
                                observed_error = observed_error[energy_mask]
                                
                                # Filter ARF with Energy Bounds (for correct dE calculation)
                                arf_filtered = sf.ResponseData()
                                arf_filtered.energy_mid = energy
                                arf_filtered.arf = arf_data.arf[:min_len][energy_mask]
                                
                                # Preserve Energy Bounds if available
                                if arf_data.energy_lo is not None and arf_data.energy_hi is not None:
                                    arf_filtered.energy_lo = arf_data.energy_lo[:min_len][energy_mask]
                                    arf_filtered.energy_hi = arf_data.energy_hi[:min_len][energy_mask]
                                


                                
                                # UI placeholders
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                current_params_display = st.empty()
                                best_so_far = st.empty()
                                sound_placeholder = st.empty()
                                
                                st.markdown("---")
                                results_container = st.container()
                                
                                # Run brute-force (parallel or sequential)
                                final_result = None
                                
                            # Create checkpoints directory
                            import os
                            import json
                            checkpoint_dir = Path("data/checkpoints")
                            checkpoint_dir.mkdir(exist_ok=True)
                            
                            # Calculate Job Hash
                            job_hash = sf.get_job_hash(model_components, param_ranges, bf_steps, len(energy), fixed_params)
                            checkpoint_file = checkpoint_dir / f"{job_hash}.json"
                            
                            st.info(f"🔑 Job ID: `{job_hash[:8]}` (ใช้สำหรับ Checkpoints)")
                            
                            # Load existing checkpoint
                            completed_parts = []
                            accumulated_best = None
                            
                            if checkpoint_file.exists():
                                try:
                                    with open(checkpoint_file, 'r') as f:
                                        checkpoint_data = json.load(f)
                                        completed_parts = checkpoint_data.get('completed_parts', [])
                                        accumulated_best = checkpoint_data.get('best_result_so_far', None)
                                        
                                        # Restore best result if valid
                                        if accumulated_best:
                                            # Update best_so_far UI immediately
                                            best_str = " | ".join([f"{k}={v:.4f}" for k, v in accumulated_best['best_params'].items()])
                                            best_so_far.success(f"""
                                            🏆 **ค่าที่ดีที่สุดจากครั้งก่อน:**  
                                            χ²/dof = **{accumulated_best['best_chi2_dof']:.4f}**  
                                            `{best_str}`
                                            """)
                                            
                                    st.info(f"📂 พบข้อมูลเก่า: ทำเสร็จแล้ว {len(completed_parts)} ส่วน ({', '.join(map(str, sorted(completed_parts)))})")
                                except:
                                    st.warning("⚠️ ไฟล์ Checkpoint เสียหาย จะเริ่มใหม่ทั้งหมด")
                            
                            # Run brute-force
                            final_result = None
                            
                            if comp_mode == "GPU (CUDA)" and sf.HAS_GPU:
                                # Run GPU Fit
                                st.info("🚀 Running on GPU...")
                                brute_force_generator = sf.brute_force_fit_gpu(
                                    energy, observed_rate, observed_error,
                                    model_components, param_ranges,
                                    n_steps=bf_steps, response=arf_filtered,
                                    fixed_params=fixed_params
                                )
                            elif comp_mode == "CPU (Parallel)":
                                # Use parallel processing
                                n_parts = 100
                                part_size = (total_combos + n_parts - 1) // n_parts
                                dynamic_batch = max(10, min(200, part_size // 10))
                                
                                brute_force_generator = sf.brute_force_fit_parallel(
                                    energy, observed_rate, observed_error,
                                    model_components, param_ranges,
                                    n_steps=bf_steps, n_workers=n_workers,
                                    batch_size=dynamic_batch,
                                    response=arf_filtered,
                                    backend='threading', # Defaulting to threading for safety
                                    n_parts=n_parts,
                                    skip_parts=completed_parts,
                                    fixed_params=fixed_params
                                )
                            else:
                                # Sequential
                                brute_force_generator = sf.brute_force_fit(
                                    energy, observed_rate, observed_error,
                                    model_components, param_ranges,
                                    n_steps=bf_steps, response=arf_filtered,
                                    fixed_params=fixed_params
                                )
                            
                            # Initialize Session Best Tracking (Reset every run)
                            session_best_accumulated = {
                                'best_chi2_dof': float('inf'),
                                'best_params': None
                            }
                            
                            for update in brute_force_generator:
                                # Update progress bar
                                progress_bar.progress(update['progress'])
                                
                                # Update status text
                                if update.get('skipped'):
                                    status_text.warning(update['description'])
                                else:
                                    # Enhanced Status Display
                                    part_info = ""
                                    if 'part_idx' in update and 'n_parts' in update:
                                        part_info = f"📦 **Part:** {update['part_idx']+1} / {update['n_parts']}"
                                    
                                    status_text.markdown(f"""
                                    **สถานะ:** {update['description']}  
                                    {part_info}  
                                    **Progress:** {update['iteration']:,} / {update['total']:,} ({update['progress']*100:.1f}%)
                                    """)
                                
                                # Show current parameters (Latest Checked)
                                if update['current_params']:
                                    params_str = " | ".join([f"{k}={v:.4f}" for k, v in update['current_params'].items()])
                                    curr_chi2 = update.get('current_chi2_dof', float('inf'))
                                    
                                    # Format string based on chi2 value (handle inf)
                                    chi2_str = f"{curr_chi2:.4f}" if curr_chi2 != float('inf') else "inf"
                                    
                                    current_params_display.info(f"""
                                    🔄 **กำลังทดสอบ (ล่าสุด):**  
                                    χ²/dof = **{chi2_str}**  
                                    `{params_str}`
                                    """)
                                
                                # Handle best result updates using accumulated best
                                current_best_data = None
                                
                                # If we have a new best in this run
                                if update.get('is_best') and update['best_chi2_dof'] < float('inf'):
                                    # Compare with accumulated best
                                    if accumulated_best is None or update['best_chi2_dof'] < accumulated_best['best_chi2_dof']:
                                        accumulated_best = {
                                            'best_chi2_dof': update['best_chi2_dof'],
                                            'best_params': update['best_params'],
                                            'best_result': update.get('best_result')
                                        }
                                # Handle best result updates (SESSION ONLY)
                                # We now track the best result found IN THIS RUN exclusively
                                batch_best_chi2 = update.get('batch_best_chi2_dof', float('inf'))
                                batch_best_params_val = update.get('batch_best_params')
                                
                                if batch_best_params_val and batch_best_chi2 < session_best_accumulated['best_chi2_dof']:
                                    session_best_accumulated['best_chi2_dof'] = batch_best_chi2
                                    session_best_accumulated['best_params'] = batch_best_params_val

                                # Background: Save Global Best if found
                                if update.get('is_best') and update['best_chi2_dof'] < float('inf'):
                                    # This ensures that if we find a new GLOBAL best, we still save it
                                    # even if we are strictly showing session best in the UI
                                    if accumulated_best is None or update['best_chi2_dof'] < accumulated_best['best_chi2_dof']:
                                         accumulated_best = {
                                            'best_chi2_dof': update['best_chi2_dof'],
                                            'best_params': update['best_params'],
                                            'best_result': update.get('best_result')
                                         }
                                         save_data = update.copy()
                                         save_data['model_components'] = model_components
                                         save_data['varied_models'] = varied_models
                                         save_data['fixed_params'] = fixed_params
                                         save_brute_force_result(save_data, run_id=run_id)

                                # Display Session Best
                                if session_best_accumulated['best_params']:
                                    best_str = " | ".join([f"{k}={v:.4f}" for k, v in session_best_accumulated['best_params'].items()])
                                    best_so_far.success(f"""
                                    🏆 **ค่าที่ดีที่สุดตอนนี้ (รอบปัจจุบัน):**  
                                    χ²/dof = **{session_best_accumulated['best_chi2_dof']:.4f}**  
                                    `{best_str}`
                                    """)

                                # Save Checkpoint on Part Completion
                                if update.get('status') == 'part_complete':
                                    part_idx = update['part_index']

                                    # Play sound effect
                                    sound_file = Path("data/sounds/Twitch Bits Donation Sound Effect  SFX.mp3")
                                    if sound_file.exists():
                                        try:
                                            import base64
                                            import time
                                            audio_bytes = sound_file.read_bytes()
                                            audio_base64 = base64.b64encode(audio_bytes).decode()
                                            # Add timestamp to force re-render
                                            unique_id = f"audio_{part_idx}_{int(time.time()*1000)}"
                                            audio_html = f'<audio id="{unique_id}" src="data:audio/mp3;base64,{audio_base64}" autoplay="autoplay" style="display:none;"></audio>'
                                            with sound_placeholder:
                                                # Empty placeholder first to ensure clean state
                                                sound_placeholder.empty()
                                                st.markdown(audio_html, unsafe_allow_html=True)
                                        except Exception as e:
                                            sound_placeholder.error(f"เล่นเสียงไม่ได้: {e}")
                                    else:
                                        sound_placeholder.warning(f"⚠️ ไม่พบไฟล์เสียง: {sound_file.name}")

                                    if part_idx not in completed_parts:
                                        completed_parts.append(part_idx)
                                        
                                        # Save to file
                                        ckpt_data = {
                                            'job_hash': job_hash,
                                            'completed_parts': completed_parts,
                                            'best_result_so_far': accumulated_best,
                                            'last_updated': str(datetime.now())
                                        }
                                        with open(checkpoint_file, 'w') as f:
                                            json.dump(ckpt_data, f, indent=4, default=json_numpy_serializer)
                                
                                final_result = update
                                
                                # Show final results
                                if final_result and final_result['status'] == 'complete':
                                    with results_container:
                                        st.balloons()
                                        st.markdown("## 🎉 Brute-Force เสร็จสิ้น!")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Best χ²/dof", f"{final_result['best_chi2_dof']:.4f}")
                                        with col2:
                                            st.metric("Combinations Tested", f"{final_result['total']:,}")
                                        with col3:
                                            interp = sf.goodness_of_fit_interpretation(final_result['best_chi2_dof'])
                                            st.markdown(interp)
                                        
                                        st.markdown("### 🎯 Best-fit Parameters")
                                        if final_result['best_params']:
                                            df_best = pd.DataFrame([
                                                {"Parameter": k, "Value": f"{v:.6f}"}
                                                for k, v in final_result['best_params'].items()
                                            ])
                                            st.dataframe(df_best, width='stretch')
                                            
                                            st.info("💡 **Tip:** คัดลอกค่าด้านบนไปใส่ในช่อง Parameter Settings แล้วกด 'เริ่มการฟิตสเปกตรัม' เพื่อ refine ผลลัพธ์")
                                        
                                        # Save result to JSON file
                                        final_result['model_components'] = model_components
                                        final_result['varied_models'] = varied_models
                                        final_result['fixed_params'] = fixed_params
                                        if save_brute_force_result(final_result):
                                            st.success("💾 บันทึกผลลัพธ์ไปยังไฟล์ JSON แล้ว!")
                            else:
                                st.error("❌ ไม่สามารถอ่านไฟล์ข้อมูลได้")
                                
                        except Exception as e:
                            st.error(f"❌ เกิดข้อผิดพลาด: {e}")
                    
                    st.markdown("---")
                    
                    # Fitting button
                    if st.button("🚀 เริ่มการฟิตสเปกตรัม", type="primary"):
                        with st.spinner("กำลังฟิตสเปกตรัม... กรุณารอสักครู่"):
                            try:
                                # Read spectrum and ARF data
                                spec_path = attached_files[selected_spec]
                                arf_path = attached_files[selected_arf_fit]
                                
                                # Load spectrum
                                spectrum = sf.read_spectrum_file(spec_path)
                                arf_data = sf.read_arf_file(arf_path)
                                
                                if spectrum is None or arf_data is None:
                                    st.error("❌ ไม่สามารถอ่านไฟล์ข้อมูลได้")
                                else:
                                    # Prepare data for fitting
                                    # Use energy from ARF
                                    energy = arf_data.energy_mid
                                    
                                    # Get count rate and errors
                                    observed_rate = spectrum.count_rate()
                                    observed_error = spectrum.count_rate_error()
                                    
                                    # Ensure compatible lengths
                                    min_len = min(len(energy), len(observed_rate))
                                    energy = energy[:min_len]
                                    observed_rate = observed_rate[:min_len]
                                    observed_error = observed_error[:min_len]
                                    
                                    # Filter energy range using user-selected values
                                    energy_mask = (energy > energy_min) & (energy < energy_max)
                                    energy = energy[:min_len][energy_mask]
                                    observed_rate = observed_rate[:min_len][energy_mask]
                                    observed_error = observed_error[:min_len][energy_mask]
                                    
                                    # Filter ARF data to match energy range
                                    arf_data_filtered = sf.ResponseData()
                                    arf_data_filtered.energy_lo = arf_data.energy_lo[:min_len][energy_mask]
                                    arf_data_filtered.energy_hi = arf_data.energy_hi[:min_len][energy_mask]
                                    arf_data_filtered.energy_mid = energy
                                    arf_data_filtered.arf = arf_data.arf[:min_len][energy_mask]
                                    
                                    # Perform fitting with ARF response
                                    fit_result = sf.fit_spectrum(
                                        energy, observed_rate, observed_error,
                                        model_components, initial_params,
                                        exposure=spectrum.exposure,
                                        response=arf_data_filtered
                                    )
                                    
                                    # Display results
                                    st.markdown("---")
                                    st.markdown("## 📊 ผลการฟิตสเปกตรัม")
                                    
                                    if fit_result['success']:
                                        st.success("✅ การฟิตสำเร็จ!")
                                        
                                        # Display goodness of fit
                                        st.markdown("### 📈 Goodness of Fit")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("χ²", f"{fit_result['chi_squared']:.2f}")
                                        with col2:
                                            st.metric("DOF", f"{fit_result['dof']}")
                                        with col3:
                                            st.metric("χ²/DOF", f"{fit_result['reduced_chi_squared']:.3f}")
                                        with col4:
                                            st.metric("Data Points", f"{fit_result['n_data_points']}")
                                        
                                        # Interpretation
                                        interpretation = sf.goodness_of_fit_interpretation(
                                            fit_result['reduced_chi_squared'])
                                        st.markdown(interpretation)
                                        
                                        # Best-fit parameters
                                        st.markdown("### 🎯 Best-fit Parameters")
                                        
                                        param_data = []
                                        for param, value in fit_result['best_params'].items():
                                            error = fit_result['param_errors'].get(param)
                                            if error is not None:
                                                param_data.append({
                                                    'Parameter': param,
                                                    'Value': f"{value:.4f}",
                                                    'Error': f"± {error:.4f}"
                                                })
                                            else:
                                                param_data.append({
                                                    'Parameter': param,
                                                    'Value': f"{value:.4f}",
                                                    'Error': "N/A"
                                                })
                                        
                                        df_params = pd.DataFrame(param_data)
                                        st.dataframe(df_params, width='stretch')
                                        
                                        # Calculate best-fit model (folded through ARF response)
                                        model_rate = sf.calculate_model_spectrum(
                                            energy, fit_result['best_params'], model_components,
                                            response=arf_data_filtered)
                                        
                                        # Plot: Data and Model
                                        st.markdown("### 📉 Spectrum และ Best-fit Model")
                                        
                                        fig = go.Figure()
                                        
                                        # Observed data with error bars
                                        fig.add_trace(go.Scatter(
                                            x=energy,
                                            y=observed_rate,
                                            error_y=dict(type='data', array=observed_error, visible=True),
                                            mode='markers',
                                            name='Observed Data',
                                            marker=dict(size=4, color='blue'),
                                            line=dict(width=0)
                                        ))
                                        
                                        # Best-fit model
                                        fig.add_trace(go.Scatter(
                                            x=energy,
                                            y=model_rate,
                                            mode='lines',
                                            name='Best-fit Model',
                                            line=dict(width=2, color='red')
                                        ))
                                        
                                        fig.update_layout(
                                            title="Spectrum with Best-fit Model",
                                            xaxis_title="Energy (keV)",
                                            yaxis_title="Count Rate (counts/s/keV)",
                                            yaxis_type="log",
                                            hovermode='x unified',
                                            template='plotly_white',
                                            height=500
                                        )
                                        
                                        st.plotly_chart(fig, width='stretch')
                                        
                                        # Plot: Residuals
                                        st.markdown("### 📊 Residuals (ค่าเหลือจากการฟิต)")
                                        
                                        residuals = sf.calculate_residuals(
                                            observed_rate, model_rate, observed_error)
                                        
                                        fig_res = go.Figure()
                                        
                                        fig_res.add_trace(go.Scatter(
                                            x=energy,
                                            y=residuals,
                                            mode='markers',
                                            name='Residuals',
                                            marker=dict(size=5, color='darkgreen'),
                                        ))
                                        
                                        # Zero line
                                        fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
                                        fig_res.add_hline(y=3, line_dash="dot", line_color="orange", 
                                                         annotation_text="±3σ")
                                        fig_res.add_hline(y=-3, line_dash="dot", line_color="orange")
                                        
                                        fig_res.update_layout(
                                            title="Residuals: (Data - Model) / Error",
                                            xaxis_title="Energy (keV)",
                                            yaxis_title="Residuals (σ)",
                                            hovermode='x unified',
                                            template='plotly_white',
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_res, width='stretch')
                                        
                                        st.info("""
                                        ℹ️ **การตีความ Residuals:**
                                        - ค่า residuals ควรกระจายแบบสุ่มรอบ 0
                                        - ถ้ามี systematic pattern แสดงว่าโมเดลอาจไม่เหมาะสม
                                        - ค่าส่วนใหญ่ควรอยู่ในช่วง ±3σ
                                        """)
                                        
                                        # Physical Interpretation
                                        st.markdown("---")
                                        st.markdown("## 🔬 การอภิปรายผลการฟิตแบบละเอียด")
                                        
                                        st.markdown("""
                                        ### องค์ประกอบทางกายภาพของสเปกตรัม Fairall 9
                                        
                                        Fairall 9 เป็น Active Galactic Nucleus (AGN) ประเภท Seyfert 1 ซึ่งมีหลุมดำมวลยิ่งยวด
                                        (Supermassive Black Hole) ที่ศูนย์กลาง สเปกตรัม X-ray ของ Fairall 9 แสดงให้เห็นองค์ประกอบ
                                        ทางกายภาพที่สำคัญหลายประการ:
                                        """)
                                        
                                        # Discuss each component
                                        if use_powerlaw:
                                            photon_idx = fit_result['best_params'].get('photon_index', 2.0)
                                            st.markdown(f"""
                                            #### 1. **Power-law Continuum** (Γ = {photon_idx:.2f})
                                            
                                            - **ความหมาย:** Power-law continuum เป็นองค์ประกอบหลักของรังสีเอกซ์ใน AGN 
                                              เกิดจากกระบวนการ **Inverse Compton scattering** ในบริเวณ corona (บริเวณร้อนจัด) 
                                              ที่อยู่เหนือ accretion disk
                                            
                                            - **Photon index (Γ = {photon_idx:.2f}):** 
                                              - ค่า Γ ≈ 1.7-2.0 เป็นค่าทั่วไปสำหรับ AGN
                                              - ค่า Γ ที่สูงกว่า → corona ร้อนกว่า หรือมี optical depth ต่ำกว่า
                                              - ค่า Γ ที่ต่ำกว่า → corona เย็นกว่า หรือมี optical depth สูงกว่า
                                            
                                            - **กระบวนการทางฟิสิกส์:** โฟตอนจาก accretion disk ถูก upscatter โดย
                                              อิเล็กตรอนความร้อนสูงใน corona ทำให้ได้โฟตอนพลังงานสูง (X-ray)
                                            """)
                                        
                                        if use_absorption:
                                            nH_val = fit_result['best_params'].get('nH', 0.0)
                                            st.markdown(f"""
                                            #### 2. **Photoelectric Absorption** (nH = {nH_val:.3f} × 10²² cm⁻²)
                                            
                                            - **ความหมาย:** การดูดกลืนรังสีเอกซ์โดย neutral hydrogen ในทางเดินของแสง
                                            
                                            - **แหล่งที่มาของ Absorption:**
                                              - **Galactic absorption:** จากทาง Milky Way ของเรา (~ 0.01-0.1 × 10²² cm⁻²)
                                              - **Intrinsic absorption:** จาก host galaxy ของ Fairall 9
                                            
                                            - **ผลกระทบ:** Absorption ส่งผลมากในช่วง soft X-ray (< 2 keV) ทำให้
                                              flux ลดลงอย่างมีนัยสำคัญที่พลังงานต่ำ
                                            
                                            - **nH = {nH_val:.3f} × 10²² cm⁻²:** ค่านี้{"สูง" if nH_val > 1.0 else "ปานกลาง" if nH_val > 0.1 else "ต่ำ"}
                                              แสดงว่า Fairall 9 {"มีสสารดูดกลืนจำนวนมากในทางเดินของแสง" if nH_val > 1.0 else "มีสสารดูดกลืนปานกลาง" if nH_val > 0.1 else "มีสสารดูดกลืนค่อนข้างน้อย"}
                                            """)
                                        
                                        if use_reflection:
                                            refl_val = fit_result['best_params'].get('refl_norm', 0.0)
                                            st.markdown(f"""
                                            #### 3. **X-ray Reflection** (R = {refl_val:.2f})
                                            
                                            - **ความหมาย:** รังสีเอกซ์จาก corona ส่องลงไปที่ accretion disk และสะท้อนกลับมา
                                            
                                            - **องค์ประกอบของ Reflection:**
                                              - **Compton hump:** โครงสร้างที่ ~ 20-40 keV จาก Compton scattering
                                              - **Iron Kα line:** เส้นสเปกตรัมที่ ~ 6.4 keV จาก fluorescence
                                              - **Relativistic effects:** การเบี่ยงเบนจากความเร็วสูงและ gravitational redshift
                                            
                                            - **Reflection strength (R = {refl_val:.2f}):**
                                              - R ~ 0: ไม่มี reflection
                                              - R ~ 1: Reflection ปานกลาง (disk ครอบคลุมมุม ~ 2π steradians)
                                              - R > 1: Strong reflection (มี light bending จาก strong gravity)
                                            
                                            - **การตีความ:** R = {refl_val:.2f} แสดงว่า Fairall 9 
                                              {"มี reflection component ที่แข็งแรง" if refl_val > 1.0 else "มี reflection ปานกลาง" if refl_val > 0.3 else "มี reflection ค่อนข้างอ่อน"}
                                              ซึ่งบ่งชี้{"การมีอิทธิพลของ strong gravity" if refl_val > 1.0 else "geometry ของระบบที่สมเหตุสมผล"}
                                            """)
                                        
                                        if use_gaussian:
                                            line_e = fit_result['best_params'].get('line_energy', 6.4)
                                            line_w = fit_result['best_params'].get('line_sigma', 0.1)
                                            st.markdown(f"""
                                            #### 4. **Iron Kα Emission Line** (E = {line_e:.2f} keV, σ = {line_w:.2f} keV)
                                            
                                            - **ความหมาย:** เส้นสเปกตรัมจาก fluorescence ของ iron ใน accretion disk
                                            
                                            - **Energy ({line_e:.2f} keV):**
                                              - Neutral iron (Fe I): 6.4 keV
                                              - He-like iron (Fe XXV): 6.7 keV
                                              - H-like iron (Fe XXVI): 6.97 keV
                                              - ค่า {line_e:.2f} keV แสดงว่าเป็น {"neutral/low-ionization iron" if line_e < 6.5 else "moderately ionized iron" if line_e < 6.8 else "highly ionized iron"}
                                            
                                            - **Line width (σ = {line_w:.2f} keV):**
                                              - σ ~ 0.01 keV: เส้นแคบ (narrow line) จาก torus ที่ไกลออกไป
                                              - σ ~ 0.1-0.5 keV: เส้นกว้าง (broad line) จาก accretion disk
                                              - Velocity ~ {(line_w/line_e * 3e5):.0f} km/s
                                            
                                            - **การตีความ:** Line width บอกความเร็วของสสารที่ปล่อยเส้นสเปกตรัม
                                              ซึ่ง{"บ่งชี้ว่ามาจาก disk ที่ใกล้หลุมดำ (high velocity)" if line_w > 0.15 else "อาจมาจาก disk หรือ torus (moderate velocity)"}
                                            """)
                                        
                                        # Overall interpretation
                                        st.markdown("""
                                        ---
                                        ### สรุปการตีความโดยรวม
                                        
                                        สเปกตรัม X-ray ของ Fairall 9 แสดงให้เห็นคุณสมบัติที่เป็นเอกลักษณ์ของ AGN:
                                        
                                        1. **Corona-Disk System:** Power-law continuum แสดงการมีอยู่ของ hot corona 
                                           ที่ทำ Comptonization ของโฟตอนจาก disk
                                        
                                        2. **Accretion Disk Reflection:** Reflection component และ iron line แสดงว่า
                                           มี accretion disk ที่สะท้อนรังสีเอกซ์จาก corona
                                        
                                        3. **Line-of-sight Absorption:** Photoelectric absorption แสดงการมีอยู่ของ
                                           neutral material ในทางเดินของแสง
                                        
                                        4. **Black Hole Environment:** พารามิเตอร์ต่างๆ ที่ได้จากการฟิตช่วยให้เราเข้าใจ
                                           สภาพแวดล้อมรอบหลุมดำมวลยิ่งยวดใน Fairall 9
                                        
                                        **ข้อจำกัดของการวิเคราะห์:**
                                        - โมเดลที่ใช้เป็น simplified models สำหรับการศึกษาเบื้องต้น
                                        - การวิเคราะห์ที่สมบูรณ์ควรใช้ advanced models เช่น relxill (relativistic reflection)
                                        - ควรพิจารณา systematic uncertainties และทำการวิเคราะห์ความไม่แน่นอนอย่างละเอียด
                                        
                                        **คำแนะนำสำหรับการศึกษาต่อ:**
                                        - ใช้ XSPEC กับ relxill model สำหรับ reflection แบบ relativistic
                                        - วิเคราะห์ timing properties เพื่อศึกษา variability
                                        - เปรียบเทียบกับ observations ในช่วงเวลาต่างๆ เพื่อศึกษา spectral evolution
                                        """)
                                        
                                        # Export results
                                        with st.expander("💾 Export Fitting Results"):
                                            # Prepare export data
                                            export_data = {
                                                'Energy_keV': energy,
                                                'Observed_Rate': observed_rate,
                                                'Observed_Error': observed_error,
                                                'Model_Rate': model_rate,
                                                'Residuals': residuals
                                            }
                                            df_export = pd.DataFrame(export_data)
                                            
                                            # Best-fit parameters text
                                            params_text = "# Best-fit Parameters\n"
                                            params_text += f"# Chi-squared: {fit_result['chi_squared']:.2f}\n"
                                            params_text += f"# DOF: {fit_result['dof']}\n"
                                            params_text += f"# Reduced chi-squared: {fit_result['reduced_chi_squared']:.3f}\n"
                                            params_text += "#\n"
                                            for param, value in fit_result['best_params'].items():
                                                error = fit_result['param_errors'].get(param)
                                                if error:
                                                    params_text += f"# {param} = {value:.4f} ± {error:.4f}\n"
                                                else:
                                                    params_text += f"# {param} = {value:.4f}\n"
                                            params_text += "#\n"
                                            
                                            csv_output = params_text + df_export.to_csv(index=False)
                                            
                                            st.download_button(
                                                label="📥 Download Fitting Results (CSV)",
                                                data=csv_output,
                                                file_name="fairall9_spectral_fitting_results.csv",
                                                mime="text/csv"
                                            )
                                            
                                            st.dataframe(df_export.head(20), width='stretch')
                                        
                                    else:
                                        st.error(f"❌ การฟิตล้มเหลว: {fit_result.get('message', 'Unknown error')}")
                                        st.info("ลองปรับค่าพารามิเตอร์เริ่มต้นหรือเปลี่ยน model components")
                            
                            except Exception as e:
                                st.error(f"❌ เกิดข้อผิดพลาด: {e}")
                                import traceback
                                st.code(traceback.format_exc())
            
            else:
                st.warning("⚠️ ต้องมีทั้ง Source Spectrum และ ARF File เพื่อทำการฟิต")
                if not source_files:
                    st.info("ไม่พบไฟล์ Source Spectrum")
                if not arf_files:
                    st.info("ไม่พบไฟล์ ARF")
    
    else:
        st.warning("ไม่พบไฟล์ที่แนบมา กรุณาอัพโหลดไฟล์ใหม่")

else:  # Upload new files
    st.sidebar.write("อัพโหลดไฟล์ของคุณ:")

    uploaded_source = st.sidebar.file_uploader("Source Spectrum (.fits)",
                                               type=['fits'])
    uploaded_bkg = st.sidebar.file_uploader("Background Spectrum (.fits)",
                                            type=['fits'])
    uploaded_arf = st.sidebar.file_uploader("ARF File (.arf)", type=['arf'])
    uploaded_rmf = st.sidebar.file_uploader("RMF File (.rmf)", type=['rmf'])

    # Create tabs for uploaded files
    tabs = st.tabs([
        "📊 Source Spectrum", "🌌 Background Spectrum",
        "🔬 Background Subtraction", "📈 ARF File", "🔲 RMF File"
    ])

    with tabs[0]:
        if uploaded_source:
            hdul = read_fits_file(uploaded_source)
            if hdul:
                plot_spectrum(hdul, "Source Spectrum")
                with st.expander("📋 ดู Header Information"):
                    display_fits_header(hdul, 1)
                hdul.close()

    with tabs[1]:
        if uploaded_bkg:
            hdul = read_fits_file(uploaded_bkg)
            if hdul:
                plot_spectrum(hdul, "Background Spectrum")
                with st.expander("📋 ดู Header Information"):
                    display_fits_header(hdul, 1)
                hdul.close()

    with tabs[2]:
        if uploaded_source and uploaded_bkg:
            st.subheader("🔬 Background Subtraction Analysis")
            try:
                source_hdul = read_fits_file(uploaded_source)
                bkg_hdul = read_fits_file(uploaded_bkg)

                if source_hdul and bkg_hdul and len(source_hdul) > 1 and len(
                        bkg_hdul) > 1:
                    source_data = source_hdul[1].data
                    bkg_data = bkg_hdul[1].data

                    if ('CHANNEL' in source_data.columns.names
                            and 'COUNTS' in source_data.columns.names
                            and 'CHANNEL' in bkg_data.columns.names
                            and 'COUNTS' in bkg_data.columns.names):

                        source_channels = source_data['CHANNEL']
                        source_counts = source_data['COUNTS']
                        bkg_counts = bkg_data['COUNTS']

                        min_len = min(len(source_counts), len(bkg_counts))
                        source_channels = source_channels[:min_len]
                        source_counts = source_counts[:min_len]
                        bkg_counts = bkg_counts[:min_len]

                        subtracted_counts = source_counts - bkg_counts

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(x=source_channels,
                                       y=source_counts,
                                       mode='lines',
                                       name='Source',
                                       line=dict(color='blue'),
                                       opacity=0.7))
                        fig.add_trace(
                            go.Scatter(x=source_channels,
                                       y=bkg_counts,
                                       mode='lines',
                                       name='Background',
                                       line=dict(color='red'),
                                       opacity=0.7))
                        fig.add_trace(
                            go.Scatter(x=source_channels,
                                       y=subtracted_counts,
                                       mode='lines',
                                       name='Subtracted',
                                       line=dict(color='green', width=2)))
                        fig.update_layout(title="Background Subtraction",
                                          xaxis_title="Channel",
                                          yaxis_title="Counts",
                                          template='plotly_white')
                        st.plotly_chart(fig, width='stretch')

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Source Total",
                                      f"{np.sum(source_counts):,.0f}")
                        with col2:
                            st.metric("Background Total",
                                      f"{np.sum(bkg_counts):,.0f}")
                        with col3:
                            st.metric("Net Total",
                                      f"{np.sum(subtracted_counts):,.0f}")

                    source_hdul.close()
                    bkg_hdul.close()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("โปรดอัพโหลดทั้ง Source และ Background Spectrum")

    with tabs[3]:
        if uploaded_arf:
            plot_arf(uploaded_arf)

    with tabs[4]:
        if uploaded_rmf:
            plot_rmf(uploaded_rmf)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**คำอธิบายไฟล์:**
- **FITS (Source)**: สเปกตรัม X-ray จากแหล่ง
- **FITS (Background)**: สเปกตรัมพื้นหลัง
- **ARF**: ประสิทธิภาพการรับแสง
- **RMF**: การกระจายพลังงาน
""")
