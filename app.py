# ============================
# ADS PMF SCALING WEB APP
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import os
from datetime import date
import re

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="ADS PMF Scaling Tool",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# 2. CUSTOM CSS: CENTERED JUGGLING LOADER
# ---------------------------------------------------------
st.markdown("""
<style>
    /* The Overlay (Background) */
    .bose-loader-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(255, 255, 255, 0.95);
        z-index: 999999;
        display: flex;
        justify-content: center; 
        align-items: center;
        flex-direction: column;
        backdrop-filter: blur(4px);
    }

    /* Container for the letters */
    .bose-juggler {
        display: flex;
        justify-content: center;
        align-items: flex-end;
        gap: 15px;
        height: 100px;
        margin-bottom: 20px;
    }

    /* Individual Letters */
    .bose-letter {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 900;
        font-size: 80px; 
        color: #000000;
        line-height: 1;
        animation: juggle 1.4s ease-in-out infinite;
    }

    /* Staggered Delay for the "Wave/Juggle" Effect */
    .bose-letter:nth-child(1) { animation-delay: 0.0s; }
    .bose-letter:nth-child(2) { animation-delay: 0.15s; }
    .bose-letter:nth-child(3) { animation-delay: 0.3s; }
    .bose-letter:nth-child(4) { animation-delay: 0.45s; }
    .bose-letter:nth-child(4) { animation-delay: 0.60s; }

    /* Status Text */
    .bose-status {
        color: #333; 
        font-family: sans-serif;
        font-size: 18px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        animation: fade 1.5s ease-in-out infinite alternate;
    }

    /* Keyframes: The Jump */
    @keyframes juggle {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-40px); }
    }
    
    /* Keyframes: Text pulsing */
    @keyframes fade {
        from { opacity: 0.6; }
        to { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def normalize_geo(name: str):
    """Normalizes geography names to handle mismatch (e.g. 'Bose UK' vs 'BOSE_UK')"""
    if pd.isna(name): return ""
    return str(name).strip().upper().replace(".", "").replace("_", "").replace(" ", "")

def generate_dynamic_pmf(last_df, curr_df, model_key, var_col, geo_col):
    """Calculates PMF multipliers by dividing Last Weekly by Current Weekly."""
    last_sub = last_df[last_df['ModelKey'] == model_key].copy()
    curr_sub = curr_df[curr_df['ModelKey'] == model_key].copy()
    
    meta_cols = ['ModelKey', var_col, geo_col]
    period_cols = [c for c in last_sub.columns if c not in meta_cols and not str(c).startswith("Unnamed:")]
    
    last_long = last_sub.melt(id_vars=[geo_col, var_col], value_vars=period_cols, 
                              var_name="SEASON", value_name="LAST_VAL")
    curr_long = curr_sub.melt(id_vars=[geo_col, var_col], value_vars=period_cols, 
                              var_name="SEASON", value_name="CURR_VAL")
    
    merged = pd.merge(last_long, curr_long, on=[geo_col, var_col, "SEASON"], how="inner")
    
    merged["LAST_VAL"] = pd.to_numeric(merged["LAST_VAL"], errors="coerce").fillna(0)
    merged["CURR_VAL"] = pd.to_numeric(merged["CURR_VAL"], errors="coerce").fillna(0)
    
    # Apply logic: if either is 0, multiplier is 1.0. Otherwise Last / Current
    merged["MULTIPLIER"] = np.where(
        (merged["CURR_VAL"] == 0) | (merged["LAST_VAL"] == 0), 
        1.0, 
        merged["LAST_VAL"] / merged["CURR_VAL"]
    )
    
    factors_df = merged[[geo_col, "SEASON", var_col, "LAST_VAL", "CURR_VAL", "MULTIPLIER"]].copy()
    factors_df.rename(columns={geo_col: "GEOGRAPHY", var_col: "VARIABLE"}, inplace=True)
    
    pmf_dict = {}
    for _, row in factors_df.iterrows():
        geo = normalize_geo(row["GEOGRAPHY"])
        season = str(row["SEASON"]).strip().upper()
        # Explicitly adding _PMF to match the ADS headers
        var = str(row["VARIABLE"]).strip().upper() + "_PMF" 
        pmf_dict[(geo, season, var)] = row["MULTIPLIER"]
        
    return pmf_dict, factors_df

def create_factors_excel(factors_df):
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        factors_df.to_excel(writer, sheet_name="Factors", index=False)
    return output_buffer.getvalue()

# ---------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processed_logs' not in st.session_state:
    st.session_state.processed_logs = None
if 'factors_file_bytes' not in st.session_state:
    st.session_state.factors_file_bytes = None
if 'file_signatures' not in st.session_state:
    st.session_state.file_signatures = None

# ---------------------------------------------------------
# 5. MAIN APP LAYOUT
# ---------------------------------------------------------
st.title("📊 ADS PMF Scaling Web Application")
st.write("Upload your ADS, Combined Weekly, Granular Spec, and Main Spec files to perform PMF scaling.")

# --- FILE UPLOAD SECTION ---
with st.expander("📁 Upload Required Files", expanded=True):
    ads_file = st.file_uploader("1. Upload ADS File (CSV or XLSX)", type=["csv", "xlsx"])
    weekly_file = st.file_uploader("2. Upload Combined Weekly File (XLSX)", type=["xlsx"])
    gran_file = st.file_uploader("3. Upload Granular Spec File (XLSX)", type=["xlsx"])
    main_spec_file = st.file_uploader("4. Upload Main Spec File (XLSX)", type=["xlsx"])

# Check if new files are uploaded to reset state
current_files = [ads_file, weekly_file, gran_file, main_spec_file]
if any(current_files) and current_files != st.session_state.file_signatures:
    st.session_state.processed_data = None
    st.session_state.file_signatures = current_files

# --- MAIN LOGIC BLOCK ---
if all(current_files):
    
    try:
        # 0. READ WEEKLY FILE FOR MAPPING
        weekly_xl = pd.ExcelFile(weekly_file)
        last_sheet = next((s for s in weekly_xl.sheet_names if "last" in s.lower()), None)
        curr_sheet = next((s for s in weekly_xl.sheet_names if "current" in s.lower()), None)
        
        if not last_sheet or not curr_sheet:
            st.error("❌ The Combined Weekly File must contain one sheet with 'last' and one with 'current' in the name.")
            st.stop()
            
        last_weekly_df = pd.read_excel(weekly_file, sheet_name=last_sheet)
        curr_weekly_df = pd.read_excel(weekly_file, sheet_name=curr_sheet)

        # 1. READ MAIN SPEC
        xl = pd.ExcelFile(main_spec_file)
        sheet_name = next((s for s in xl.sheet_names if "model" in s.lower()), xl.sheet_names[0])
        
        # Smart Header Detection
        preview = pd.read_excel(main_spec_file, sheet_name=sheet_name, nrows=50, header=None)
        header_row_idx = None
        for i, row in preview.iterrows():
            cells = [str(x).strip().lower() for x in row.fillna("")]
            if any("variable" in c for c in cells) and any("type" in c for c in cells):
                header_row_idx = i
                break
        
        if header_row_idx is None:
            st.error("❌ Could not auto-detect header row in Main Spec.")
            st.stop()

        ms_df = pd.read_excel(main_spec_file, sheet_name=sheet_name, header=header_row_idx, dtype=str)
        ms_df.columns = [str(c).strip().upper() for c in ms_df.columns]
        
        type_col = next((c for c in ms_df.columns if c in ["TYPE", "TYPES", "VARIABLE TYPE"]), None)
        var_col = next((c for c in ms_df.columns if c in ["VARIABLE", "VARIABLES", "VARIABLE NAME", "VAR NAME"]), None)
        
        if not type_col or not var_col:
            st.error("❌ Main Spec missing 'Type' or 'Variable' columns.")
            st.stop()
            
        available_types = sorted(ms_df[type_col].dropna().unique().tolist())
        var_to_type = dict(zip(ms_df[var_col].str.upper().str.strip(), ms_df[type_col]))

        # --- CONFIGURATION UI ---
        st.divider()
        st.subheader("⚙️ Configuration")
        
        # WEEKLY FILE MAPPING
        st.markdown("##### 📅 Weekly Data Mapping")
        cols_list = list(last_weekly_df.columns)
        available_models = last_weekly_df['ModelKey'].dropna().unique().tolist() if 'ModelKey' in cols_list else []
        selected_model = st.selectbox("Select ModelKey:", available_models)
        
        default_geo_idx = cols_list.index("DataBase") if "DataBase" in cols_list else 0
        default_var_idx = cols_list.index("3.3") if "3.3" in cols_list else 0

        c_map1, c_map2 = st.columns(2)
        weekly_geo_col = c_map1.selectbox("Geography Column (Weekly):", cols_list, index=default_geo_idx)
        weekly_var_col = c_map2.selectbox("Variable Column (Weekly):", cols_list, index=default_var_idx)
        
        st.divider()
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("##### 🗂️ Scaling Options")
            selected_type_category = st.radio("Select Variable Type to scale:", ["All"] + available_types)
        
        with c2:
            st.markdown("##### 🎯 Tolerance Settings")
            st.caption("Multiplication is **SKIPPED** if the calculated multiplier falls strictly between Min and Max.")
            
            tolerance_map = {}
            types_to_configure = available_types if selected_type_category == "All" else [selected_type_category]
            
            with st.container(height=200):
                for t in types_to_configure:
                    cc1, cc2 = st.columns(2)
                    t_min = cc1.number_input(f"Min ({t})", value=0.95, step=0.01, format="%.2f", key=f"min_{t}")
                    t_max = cc2.number_input(f"Max ({t})", value=1.05, step=0.01, format="%.2f", key=f"max_{t}")
                    tolerance_map[t] = (t_min, t_max)
            
            # --- UPPER LIMIT SETTING ---
            st.divider()
            st.markdown("##### 🛑 Upper Limit Threshold")
            st.caption("Multiplication is **SKIPPED** if the multiplier is $\ge$ this value.")
            max_multiplier_limit = st.number_input("Absolute Maximum Multiplier", value=100.0, step=1.0, format="%.2f")

        # --- PROCESS BUTTON ---
        st.divider()
        
        if st.session_state.processed_data is None:
            start_process = st.button("🚀 Start Dynamic PMF Scaling", type="primary", use_container_width=True)
            
            if start_process:
                # 🌀 SHOW CENTERED JUGGLING LOADER
                loader_placeholder = st.empty()
                loader_placeholder.markdown("""
                <div class="bose-loader-overlay">
                    <div class="bose-juggler">
                        <span class="bose-letter">A</span>
                        <span class="bose-letter">M</span>
                        <span class="bose-letter">I</span>
                        <span class="bose-letter">C</span>
                        <span class="bose-letter">A</span>
                    </div>
                    <div class="bose-status">Processing Data...</div>
                </div>
                """, unsafe_allow_html=True)

                # Give browser time to render animation
                time.sleep(0.8)

                try:
                    # ==========================================
                    # 🚀 CORE PROCESSING LOGIC
                    # ==========================================
                    today_str = date.today().isoformat()
                    
                    # 1. Generate Dynamic Factors Dictionary
                    pmf_dict, factors_df = generate_dynamic_pmf(
                        last_weekly_df, 
                        curr_weekly_df, 
                        selected_model, 
                        weekly_var_col, 
                        weekly_geo_col
                    )
                    
                    # 2. Load ADS
                    if ads_file.name.endswith(".csv"):
                        ads_df = pd.read_csv(ads_file, dtype=str)
                    else:
                        ads_df = pd.read_excel(ads_file, dtype=str)
                    
                    ads_filename = os.path.splitext(ads_file.name)[0]
                    ads_df.columns = [c.strip() for c in ads_df.columns]
                    geo_col = next((c for c in ads_df.columns if c.upper() == "GEOGRAPHY"), None)
                    season_col = next((c for c in ads_df.columns if c.upper() in ["SEASON", "PERIOD_DEFINITION", "TIME_PERIODS"]), None)

                    if not geo_col or not season_col:
                        loader_placeholder.empty()
                        st.error("❌ ADS missing Geography or Season column.")
                        st.stop()

                    ads_work = pd.DataFrame()
                    ads_work["_G"] = ads_df[geo_col].str.upper().str.strip()
                    ads_work["_S"] = ads_df[season_col].str.upper().str.strip()

                    # 3. Load Granular Skip Rules
                    map_df = pd.read_excel(gran_file, sheet_name="MAP", dtype=str)
                    map_df.columns = [c.upper() for c in map_df.columns]
                    geo2map = dict(zip(map_df["GEOGRAPHY"].str.upper().str.strip(), map_df["MAP"].str.upper().str.strip()))
                    ads_work["_MAP"] = ads_work["_G"].apply(normalize_geo).map({normalize_geo(k): v for k, v in geo2map.items()})

                    skip_triples = set()
                    gran_xl = pd.ExcelFile(gran_file)
                    season_pattern = re.compile(r"^S\d\s20\d{2}$")
                    
                    for code in map_df["MAP"].dropna().unique():
                        sheet_code = str(code)
                        if sheet_code in gran_xl.sheet_names:
                            gdf = pd.read_excel(gran_file, sheet_name=sheet_code, dtype=str).iloc[:, :4]
                            gdf.columns = [c.upper() for c in gdf.columns]
                            if "VARIABLE" in gdf.columns and "CONTRIBUTION" in gdf.columns:
                                gdf = gdf[gdf["CONTRIBUTION"].astype(str).str.match(season_pattern, na=False)]
                                for _, row in gdf.iterrows():
                                    skip_triples.add((f"{str(row['VARIABLE']).strip().upper()}_PMF", str(row["CONTRIBUTION"]).strip().upper(), sheet_code))

                    # 4. Apply Multipliers
                    result_ads = ads_df.copy()
                    skipped_rows = []
                    multiplied_rows = []
                    
                    if selected_type_category == "All":
                        allowed_vars = set(ms_df[var_col].str.upper().str.strip())
                    else:
                        allowed_vars = set(ms_df[ms_df[type_col] == selected_type_category][var_col].str.upper().str.strip())

                    common_vars = [c for c in ads_df.columns if "_PMF" in c.upper()]

                    G_arr = ads_work["_G"].values
                    S_arr = ads_work["_S"].values
                    M_arr = ads_work["_MAP"].values

                    for col in common_vars:
                        col_u = col.upper()
                        col_base = col_u.replace("_PMF", "")

                        if selected_type_category != "All" and col_base not in allowed_vars:
                            continue

                        v_type = var_to_type.get(col_base)
                        t_min, t_max = tolerance_map.get(v_type, (0.95, 1.05))

                        col_values = pd.to_numeric(result_ads[col], errors="coerce")
                        
                        for i in range(len(result_ads)):
                            season = S_arr[i]
                            map_code = M_arr[i]
                            geo = G_arr[i]

                            if (col_u, season, map_code) in skip_triples:
                                skipped_rows.append((i, geo, season, map_code, col, "Granular Spec"))
                                continue
                            
                            # Fetch multiplier mapping
                            mult = pmf_dict.get((normalize_geo(geo), season, col_u))

                            if mult is None or pd.isna(col_values.iat[i]):
                                continue

                            # --- UPPER LIMIT CHECK ---
                            if mult >= max_multiplier_limit:
                                skipped_rows.append((i, geo, season, map_code, col, f"Exceeds Limit ({mult:.3f})"))
                                continue

                            # --- TOLERANCE CHECK ---
                            if t_min < mult < t_max:
                                skipped_rows.append((i, geo, season, map_code, col, f"Tolerance ({mult:.3f})"))
                                continue

                            updated = col_values.iat[i] * mult
                            result_ads.at[i, col] = str(updated)
                            multiplied_rows.append((i, geo, season, map_code, col, col_values.iat[i], mult, updated))
                    
                    # 5. Save Results to Session State
                    log_output = io.BytesIO()
                    with pd.ExcelWriter(log_output, engine="openpyxl") as writer:
                        pd.DataFrame(skipped_rows, columns=["Row", "Geo", "Season", "MAP", "Variable", "Reason"]).to_excel(writer, sheet_name="Skipped", index=False)
                        pd.DataFrame(multiplied_rows, columns=["Row", "Geo", "Season", "MAP", "Var", "Orig", "Mult", "New"]).to_excel(writer, sheet_name="Multiplied", index=False)
                    
                    st.session_state.processed_data = result_ads.to_csv(index=False).encode()
                    st.session_state.processed_logs = log_output.getvalue()
                    st.session_state.factors_file_bytes = create_factors_excel(factors_df)
                    
                    st.session_state.scaled_filename = f"{ads_filename}_Scaled_{selected_type_category}_{today_str}.csv"
                    st.session_state.log_filename = f"{ads_filename}_Logs_{selected_type_category}_{today_str}.xlsx"
                    st.session_state.factors_filename = f"Calculated_Factors_{selected_type_category}_{today_str}.xlsx"

                    loader_placeholder.empty()
                    st.rerun()

                except Exception as e:
                    loader_placeholder.empty()
                    st.error(f"An error occurred: {e}")
                    st.stop()

    except Exception as e:
        st.error(f"Error initializing files: {e}")

# --- DOWNLOAD SECTION ---
if st.session_state.processed_data is not None:
    st.success("✅ PMF Scaling Complete!")
    
    with st.expander("📥 Download Outputs", expanded=True):
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            st.download_button(
                label="⬇️ Download Scaled ADS (CSV)",
                data=st.session_state.processed_data,
                file_name=st.session_state.scaled_filename,
                mime="text/csv",
                use_container_width=True
            )
        
        with col_d2:
            st.download_button(
                label="⬇️ Download Factors (Excel)",
                data=st.session_state.factors_file_bytes,
                file_name=st.session_state.factors_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        with col_d3:
            st.download_button(
                label="⬇️ Download Logs (Excel)",
                data=st.session_state.processed_logs,
                file_name=st.session_state.log_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    st.divider()
    if st.button("🔄 Reset and Start Over", type="secondary"):
        st.session_state.processed_data = None
        st.rerun()
