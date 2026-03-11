# ================================
# FULL STREAMLIT APP – A118T ULTIMATE STABLE VERSION
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================
# PAGE CONFIG & CSS
# ================================
st.set_page_config(page_title="SPC Hardness Dashboard - A118T", layout="wide")
st.title("📊 Hardness – Visual Analytics Dashboard (A118T)")

def add_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #ffffff; box-shadow: 2px 0 5px rgba(0,0,0,0.05); border-right: none; }
        h1, h2, h3 { color: #2c3e50 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 600; }
        [data-testid="stMetricValue"] { background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: #007bff; }
        thead tr th:first-child {display:none}
        tbody th {display:none}
        .stDataFrame { border: none !important; }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# ================================
# GLOBAL VARIABLES (TARGET 85-90)
# ================================
TARGET_MIN = 85.0
TARGET_MAX = 90.0

# ================================
# LOAD & CLEAN DATA
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1hC5nnxqDLjF8-wUm8gtj11_5HFMxBlogY84Z0cRCj2s/export?format=csv"

@st.cache_data
def load_main():
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    if "<!doctype html>" in r.text[:50].lower() or "<html" in r.text[:50].lower():
        st.error("🚨 LỖI BẢO MẬT: Link Google Sheet đang bị khóa. Vui lòng vào Google Sheet -> Share -> Chọn 'Anyone with the link' (Bất kỳ ai có liên kết).")
        st.stop()
    return pd.read_csv(StringIO(r.text))

raw = load_main()

# Xử lý ngày tháng
data_period_str = "N/A"
date_col = next((c for c in raw.columns if 'DATE' in str(c).upper()), None)
if date_col:
    raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
    min_date = raw[date_col].min()
    max_date = raw[date_col].max()
    if pd.notna(min_date) and pd.notna(max_date):
        data_period_str = f"{min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}"

current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
st.markdown(f"""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
    <strong>🕒 Report Generated:</strong> {current_time} &nbsp;&nbsp;|&nbsp;&nbsp; 
    <strong>📅 Data Period:</strong> {data_period_str} &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>🎯 Target Hardness:</strong> {TARGET_MIN} ~ {TARGET_MAX}
</div>
""", unsafe_allow_html=True)

# BỘ QUÉT TIÊU ĐỀ CỘT TUYỆT ĐỐI
col_mapping = {}
for col in raw.columns:
    clean_name = re.sub(r'[\s_]+', '', str(col)).upper()
    if 'PRODUCTSPEC' in clean_name: col_mapping[col] = 'Product_Spec'
    elif 'HRSTEELGRADE' in clean_name or 'STEELGRADE' in clean_name: col_mapping[col] = 'Material'
    elif 'CLASSIFY' in clean_name: col_mapping[col] = 'Rolling_Type'
    elif 'QUALITYCODE' in clean_name: col_mapping[col] = 'Quality_Code'
    elif 'ORDERGAUGE' in clean_name: col_mapping[col] = 'Order_Gauge'
    elif 'METALLICCOATING' in clean_name: col_mapping[col] = 'Metallic_Type'
    elif 'COILNO' in clean_name: col_mapping[col] = 'COIL_NO'
    elif 'STANDARDHARDNESS' in clean_name: col_mapping[col] = 'Std_Text'
    elif 'HARDNESS冶金' in clean_name or '冶金' in clean_name: col_mapping[col] = 'Hardness_LAB'
    elif '鍍鋅線C' in clean_name or ('HARDNESS' in clean_name and 'C' in clean_name and 'N' not in clean_name and 'S' not in clean_name): col_mapping[col] = 'Hardness_LINE'
    elif 'TENSILEYIELD' in clean_name: col_mapping[col] = 'YS'
    elif 'TENSILETENSILE' in clean_name: col_mapping[col] = 'TS'
    elif 'TENSILEELONG' in clean_name: col_mapping[col] = 'EL'
    elif 'STANDARDTSMIN' in clean_name: col_mapping[col] = 'Standard TS min'
    elif 'STANDARDTSMAX' in clean_name: col_mapping[col] = 'Standard TS max'
    elif 'STANDARDYSMIN' in clean_name: col_mapping[col] = 'Standard YS min'
    elif 'STANDARDYSMAX' in clean_name: col_mapping[col] = 'Standard YS max'
    elif 'STANDARDELMIN' in clean_name: col_mapping[col] = 'Standard EL min'
    elif 'STANDARDELMAX' in clean_name: col_mapping[col] = 'Standard EL max'

df = raw.rename(columns=col_mapping)

if "Hardness_LINE" not in df.columns:
    st.error(f"⚠️ Dữ liệu bị lỗi cột Độ cứng. Các cột hiện có: {list(raw.columns)}")
    st.stop()

if "COIL_NO" not in df.columns: df["COIL_NO"] = df.index 
if "Material" not in df.columns: df["Material"] = "A118T"
if "Product_Spec" not in df.columns: df["Product_Spec"] = "N/A"

# LỌC ĐỘC QUYỀN A118T & CÁC SPECS YÊU CẦU
allowed_keywords = "A118T|2657/G01T|N SZACC|NSZACC"
df = df[
    (df["Material"].astype(str).str.upper().str.contains(allowed_keywords, regex=True)) | 
    (df.get("Product_Spec", pd.Series(dtype=str)).astype(str).str.upper().str.contains(allowed_keywords, regex=True))
].copy()

if df.empty:
    st.error("⚠️ Không tìm thấy dữ liệu cho mã A118T, 2657/G01T hoặc N SZACC.")
    st.stop()

# XỬ LÝ SỐ LIỆU & LOẠI BỎ GIÁ TRỊ 0/NA
df["Limit_Min"] = 80.0
df["Limit_Max"] = 93.0

def split_std(x):
    if isinstance(x, str) and "~" in x:
        try:
            lo, hi = x.split("~")
            return float(lo.strip()), float(hi.strip())
        except: pass
    return 80.0, 93.0

if "Std_Text" in df.columns:
    limits = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))
    if not limits.empty and limits.shape[1] == 2:
        df[["Limit_Min", "Limit_Max"]] = limits

df["Lab_Min"] = df["Limit_Min"]
df["Lab_Max"] = df["Limit_Max"]
df["Rule_Name"] = "Direct Spec"

# Thay thế giá trị 0 thành NA để loại bỏ Outlier
test_cols = ["Hardness_LAB", "Hardness_LINE", "YS", "TS", "EL", 
             "Standard TS min", "Standard TS max", "Standard YS min", "Standard YS max", 
             "Standard EL min", "Standard EL max"]

for c in test_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace(0, np.nan)

# Ép định dạng 2 chữ số cho Gauge
if "Order_Gauge" in df.columns:
    df["Order_Gauge"] = pd.to_numeric(df["Order_Gauge"], errors="coerce")
    df["Order_Gauge"] = df["Order_Gauge"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

df = df.dropna(subset=["Hardness_LINE"])

# ================================
# SIDEBAR FILTER
# ================================
st.sidebar.header("🎛 FILTER (A118T & Specs)")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

all_specs = sorted(df["Product_Spec"].dropna().astype(str).unique()) if "Product_Spec" in df else []
all_rolling = sorted(df["Rolling_Type"].dropna().astype(str).unique()) if "Rolling_Type" in df else []
all_metal = sorted(df["Metallic_Type"].dropna().astype(str).unique()) if "Metallic_Type" in df else []
all_gauge = sorted(df["Order_Gauge"].dropna().unique()) if "Order_Gauge" in df else []
all_qgroup = sorted(df["Quality_Group"].dropna().astype(str).unique()) if "Quality_Group" in df else []

specs_filter = st.sidebar.selectbox("1. Product Specs", ["All"] + list(all_specs)) if all_specs else "All"
rolling = st.sidebar.selectbox("2. Rolling Type", ["All"] + list(all_rolling)) if all_rolling else "All"
metal = st.sidebar.selectbox("3. Metallic Type", ["All"] + list(all_metal)) if all_metal else "All"
gauge = st.sidebar.selectbox("4. Order Gauge (Thickness)", ["All"] + list(all_gauge)) if all_gauge else "All"
qgroup = st.sidebar.selectbox("5. Quality Group", ["All"] + list(all_qgroup)) if all_qgroup else "All"

df_master_full = df.copy() 

if specs_filter != "All" and "Product_Spec" in df: df = df[df["Product_Spec"].astype(str) == specs_filter]
if rolling != "All" and "Rolling_Type" in df: df = df[df["Rolling_Type"].astype(str) == rolling]
if metal != "All" and "Metallic_Type" in df: df = df[df["Metallic_Type"].astype(str) == metal]
if gauge != "All" and "Order_Gauge" in df: df = df[df["Order_Gauge"] == gauge]
if qgroup != "All" and "Quality_Group" in df: df = df[df["Quality_Group"].astype(str) == qgroup]

view_mode = st.sidebar.radio(
    "📊 View Mode",
    [
        "📋 Data Inspection",
        "📊 Executive KPI Dashboard",
        "🚀 Global Summary Dashboard",
        "📉 Hardness Analysis (Trend & Dist)",
        "🔗 Correlation: Hardness vs Mech Props",
        "⚙️ Mech Props Analysis",
        "🔍 Lookup: Hardness Range → Actual Mech Props",
        "🎯 Find Target Hardness (Reverse Lookup)",
        "🧮 Predict TS/YS/EL from Std Hardness",
        "🎛️ Control Limit Calculator (Compare 3 Methods)",
        "👑 Master Dictionary Export",
    ]
)

GROUP_COLS = [c for c in ["Product_Spec", "Rolling_Type", "Metallic_Type", "Quality_Group", "Material", "Order_Gauge"] if c in df.columns]
if not GROUP_COLS: GROUP_COLS = ["Material"]

cnt = df.groupby(GROUP_COLS).agg(N_Coils=("COIL_NO","nunique")).reset_index()
valid = cnt[cnt["N_Coils"] >= 1] 

if valid.empty:
    st.warning("⚠️ No valid coils found for the current filter. Please adjust the sidebar.")
    st.stop()

# ==============================================================================
# 0. EXECUTIVE KPI DASHBOARD (OVERVIEW)
# ==============================================================================
if view_mode == "📊 Executive KPI Dashboard":
    st.markdown("## 📊 Executive KPI Dashboard (Overall Quality Overview)")
    df_kpi = df.dropna(subset=['TS', 'YS', 'EL', 'Hardness_LINE']).copy()
    
    if df_kpi.empty:
        st.warning("⚠️ Không đủ dữ liệu Cơ tính (Mech Props) cho các cuộn thép trong bộ lọc này.")
    else:
        total_coils = len(df_kpi)
        def clean_num(val, is_pct=False):
            if pd.isna(val): return "0%" if is_pct else "0"
            v = round(float(val), 2)
            res = str(int(v)) if v.is_integer() else str(v)
            return f"{res}%" if is_pct else res

        def check_pass(val, min_col, max_col):
            s_min = df_kpi[min_col].fillna(0) if min_col in df_kpi.columns else 0
            s_max = df_kpi[max_col].fillna(9999).replace(0, 9999) if max_col in df_kpi.columns else 9999
            return (val >= s_min) & (val <= s_max)
        
        df_kpi['TS_Pass'] = check_pass(df_kpi['TS'], 'Standard TS min', 'Standard TS max')
        df_kpi['YS_Pass'] = check_pass(df_kpi['YS'], 'Standard YS min', 'Standard YS max')
        df_kpi['EL_Pass'] = df_kpi['EL'] >= (df_kpi['Standard EL min'].fillna(0) if 'Standard EL min' in df_kpi.columns else 0)
        df_kpi['All_Pass'] = df_kpi['TS_Pass'] & df_kpi['YS_Pass'] & df_kpi['EL_Pass']
        df_kpi['HRB_Pass'] = (df_kpi['Hardness_LINE'] >= df_kpi['Limit_Min']) & (df_kpi['Hardness_LINE'] <= df_kpi['Limit_Max'])
        df_kpi['Target_Pass'] = (df_kpi['Hardness_LINE'] >= TARGET_MIN) & (df_kpi['Hardness_LINE'] <= TARGET_MAX)
        
        yield_rate = df_kpi['All_Pass'].mean() * 100
        hrb_yield = df_kpi['HRB_Pass'].mean() * 100 
        target_yield = df_kpi['Target_Pass'].mean() * 100

        st.markdown("### 🏆 Overall Quality Metrics")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        col1.metric("📦 Total Coils Tested", f"{total_coils:,}")
        col2.metric("✅ Mech Yield Rate", clean_num(yield_rate, True), clean_num(yield_rate - 100, True) if yield_rate < 100 else "Perfect")
        col3.metric("🎯 Std HRB Yield", clean_num(hrb_yield, True), clean_num(hrb_yield - 100, True) if hrb_yield < 100 else "In Control")
        col4.metric(f"🌟 Target Yield ({TARGET_MIN}-{TARGET_MAX})", clean_num(target_yield, True))
        col5.metric("TS Pass", clean_num(df_kpi['TS_Pass'].mean() * 100, True))
        col6.metric("YS Pass", clean_num(df_kpi['YS_Pass'].mean() * 100, True))
        col7.metric("EL Pass", clean_num(df_kpi['EL_Pass'].mean() * 100, True))
        
        st.markdown("---")

        st.markdown("### ⚠️ High-Risk Specs Watchlist")
        group_cols = ["Product_Spec", "Quality_Group", "Material", "Order_Gauge"]
        valid_group_cols = [c for c in group_cols if c in df_kpi.columns]
        
        risk_summary = df_kpi.groupby(valid_group_cols).agg(
            Total_Coils=('COIL_NO', 'count'),
            Mech_Pass_Coils=('All_Pass', 'sum'),
            HRB_Pass_Coils=('HRB_Pass', 'sum'), 
            Target_Pass_Coils=('Target_Pass', 'sum'),
            Hardness_Mean=('Hardness_LINE', 'mean'),
            Hardness_Std=('Hardness_LINE', 'std'),
            LSL=('Limit_Min', 'first'),
            USL=('Limit_Max', 'first')
        ).reset_index()
        
        risk_summary['Mech Yield (%)'] = (risk_summary['Mech_Pass_Coils'] / risk_summary['Total_Coils'] * 100)
        risk_summary['HRB Yield (%)'] = (risk_summary['HRB_Pass_Coils'] / risk_summary['Total_Coils'] * 100)
        risk_summary['Target Yield (%)'] = (risk_summary['Target_Pass_Coils'] / risk_summary['Total_Coils'] * 100)
        
        risk_top = risk_summary[risk_summary['Total_Coils'] >= 3].sort_values(['Mech Yield (%)', 'HRB Yield (%)']).head(10)
        
        if not risk_top.empty:
            rename_dict = {"Product_Spec": "Specification", "Quality_Group": "Quality", "Material": "Material", "Order_Gauge": "Gauge", "Total_Coils": "Tested Coils", "Hardness_Mean": "Avg Hardness", "Hardness_Std": "Hardness Std Dev"}
            risk_top = risk_top.rename(columns=rename_dict)
            
            cols_order = ["Specification", "Quality", "Material", "Gauge", "Tested Coils", "Mech Yield (%)", "HRB Yield (%)", "Target Yield (%)", "Avg Hardness", "Hardness Std Dev"]
            risk_top_display = risk_top[[c for c in cols_order if c in risk_top.columns]].copy()
            
            for col in ['Mech Yield (%)', 'HRB Yield (%)', 'Target Yield (%)']:
                if col in risk_top_display.columns: risk_top_display[col] = risk_top_display[col].apply(lambda x: clean_num(x, True))
            risk_top_display['Avg Hardness'] = risk_top_display['Avg Hardness'].apply(lambda x: clean_num(x))
            risk_top_display['Hardness Std Dev'] = risk_top_display['Hardness Std Dev'].apply(lambda x: clean_num(x))
            
            def style_risk(val):
                try:
                    num = float(str(val).replace('%', '').strip())
                    if num < 100: return 'color: #d32f2f; font-weight: bold; background-color: #ffebee'
                    if num >= 100: return 'color: #388e3c; font-weight: bold'
                except: pass
                return ''

            styled_risk = risk_top_display.style
            if hasattr(styled_risk, "map"): styled_risk = styled_risk.map(style_risk, subset=['Mech Yield (%)', 'HRB Yield (%)'])
            else: styled_risk = styled_risk.applymap(style_risk, subset=['Mech Yield (%)', 'HRB Yield (%)'])
            
            st.dataframe(styled_risk, use_container_width=True, hide_index=True)
            
            st.markdown("#### 🔔 Visual Deep Dive: Top 5 Risk Distributions")
            top_5_risks = risk_top.head(5).to_dict('records')
            
            if len(top_5_risks) > 0:
                chart_cols = st.columns(min(len(top_5_risks), 3))
                for idx, item in enumerate(top_5_risks):
                    spec, mat, gauge_val = item.get("Specification", "N/A"), item.get("Material", "N/A"), item.get("Gauge", "N/A")
                    tdf = df_kpi[(df_kpi.get("Product_Spec", "") == spec) & (df_kpi.get("Material", "") == mat) & (df_kpi.get("Order_Gauge", "") == gauge_val)]
                    
                    if not tdf.empty:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        h_data = tdf["Hardness_LINE"].dropna()
                        ax.hist(h_data, bins=15, color="#ff9999", edgecolor="white", density=True, alpha=0.8)
                        
                        m_val, s_val = h_data.mean(), h_data.std()
                        if s_val > 0:
                            x_ax = np.linspace(h_data.min() - 2, h_data.max() + 2, 100)
                            ax.plot(x_ax, (1/(s_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_ax - m_val) / s_val)**2), color="#cc0000", lw=2)
                        
                        l_min, l_max = tdf["Limit_Min"].iloc[0], tdf["Limit_Max"].iloc[0]
                        ax.axvline(l_min, color="black", linestyle="--", lw=1.5, label=f"Std LSL ({l_min:.0f})")
                        if l_max < 9000: ax.axvline(l_max, color="black", linestyle="--", lw=1.5, label=f"Std USL ({l_max:.0f})")
                        
                        ax.axvline(TARGET_MIN, color="green", linestyle=":", lw=2, label=f"Target ({TARGET_MIN})")
                        ax.axvline(TARGET_MAX, color="green", linestyle=":", lw=2, label=f"Target ({TARGET_MAX})")
                        ax.axvspan(TARGET_MIN, TARGET_MAX, color="green", alpha=0.1)

                        ax.set_title(f"TOP {idx+1}: {spec}\nMat: {mat} | Gauge: {gauge_val} | N={len(h_data)}", fontsize=10, fontweight="bold")
                        ax.legend(fontsize=8, loc="upper right")
                        ax.grid(alpha=0.3, linestyle=":")
                        chart_cols[idx % 3].pyplot(fig)
            
            st.markdown("#### 📑 Export Actionable Report")
            import streamlit.components.v1 as components
            col_csv, col_pdf, _ = st.columns([2, 2, 6])
            with col_csv: st.download_button("📥 Download Watchlist (CSV)", data=risk_top.to_csv(index=False).encode('utf-8-sig'), file_name="High_Risk.csv", mime="text/csv", use_container_width=True)
            with col_pdf: 
                if st.button("🖨️ Save as PDF", use_container_width=True): components.html("<script>window.parent.print();</script>", height=0)
    st.stop()

# ==============================================================================
# 🚀 GLOBAL SUMMARY DASHBOARD
# ==============================================================================
if view_mode == "🚀 Global Summary Dashboard":
    st.markdown("## 🚀 Global Process Dashboard (A118T)")
    tab1, tab2 = st.tabs(["📊 1. Performance Overview", "🧠 2. Decision Support (Risk AI)"])

    with tab1:
        stats_rows = []
        for _, g in valid.iterrows():
            conditions = [df[col] == g[col] for col in GROUP_COLS]
            sub_grp = df[np.logical_and.reduce(conditions)].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 1: continue

            l_min_val, l_max_val = sub_grp['Limit_Min'].min(), sub_grp['Limit_Max'].max()
            
            def get_l_str(s_min, s_max):
                v_min = sub_grp[s_min].max() if s_min in sub_grp else 0 
                v_max = sub_grp[s_max].min() if s_max in sub_grp else 0 
                if pd.isna(v_min): v_min = 0
                if pd.isna(v_max): v_max = 0
                if v_min > 0 and v_max > 0 and v_max < 9000: return f"{v_min:.0f}~{v_max:.0f}"
                elif v_min > 0: return f"≥ {v_min:.0f}"
                elif v_max > 0 and v_max < 9000: return f"≤ {v_max:.0f}"
                else: return "-"

            n_total = len(sub_grp)
            n_ng = sub_grp[(sub_grp["Hardness_LINE"] < l_min_val) | (sub_grp["Hardness_LINE"] > l_max_val)].shape[0]

            row_data = {col: g[col] for col in GROUP_COLS}
            row_data.update({
                "HRB Limit": f"{l_min_val:.0f}~{l_max_val:.0f}", "N": n_total, "Pass Rate": ((n_total - n_ng) / n_total) * 100,
                "HRB (Avg)": sub_grp["Hardness_LINE"].mean(), "TS (Avg)": sub_grp["TS"].mean(),
                "YS (Avg)": sub_grp["YS"].mean(), "EL (Avg)": sub_grp["EL"].mean(),
                "TS Limit": get_l_str("Standard TS min", "Standard TS max"), 
                "YS Limit": get_l_str("Standard YS min", "Standard YS max"), 
                "EL Limit": get_l_str("Standard EL min", "Standard EL max")
            })
            stats_rows.append(row_data)

        if stats_rows:
            def c_pass(val): return f"background-color: {'#d4edda' if val >= 98 else ('#fff3cd' if val >= 90 else '#f8d7da')}; color: {'#155724' if val >= 98 else ('#856404' if val >= 90 else '#721c24')}; font-weight: bold"
            st.dataframe(pd.DataFrame(stats_rows).style.format("{:.1f}", subset=[c for c in pd.DataFrame(stats_rows).columns if "(Avg)" in c or "Pass" in c]).applymap(c_pass, subset=["Pass Rate"]).background_gradient(subset=["HRB (Avg)"], cmap="Blues"), use_container_width=True)
        else: st.warning("Insufficient data.")

    with tab2:
        col_in1, col_in2 = st.columns([1, 1])
        with col_in1: user_hrb = st.number_input("1️⃣ Target HRB", value=85.0, step=0.5, format="%.1f")
        with col_in2: safety_k = st.selectbox("2️⃣ Select Safety Factor:", [1.0, 2.0, 3.0], index=1)

        rows_ts, rows_ys, rows_el = [], [], []
        for _, g in valid.iterrows():
            conditions = [df[col] == g[col] for col in GROUP_COLS]
            sub_grp = df[np.logical_and.reduce(conditions)].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 3: continue 

            X = sub_grp[["Hardness_LINE"]].values
            def g_risk(col, sp_min):
                m = LinearRegression().fit(X, sub_grp[col].values)
                pred = m.predict([[user_hrb]])[0]
                safe = pred - (safety_k * np.sqrt(mean_squared_error(sub_grp[col], m.predict(X))))
                return pred, safe, "🔴 High Risk" if (sp_min > 0 and safe < sp_min) else "🟢 Safe"

            try:
                b_dict = {col: g[col] for col in GROUP_COLS}
                
                ts_m = sub_grp["Standard TS min"].max() if "Standard TS min" in sub_grp else 0
                p_ts, s_ts, r_ts = g_risk("TS", ts_m)
                dt = b_dict.copy(); dt.update({"Pred TS": f"{p_ts:.0f}", "Worst Case": f"{s_ts:.0f}", "Limit": f"≥ {ts_m:.0f}" if ts_m > 0 else "-", "Status": r_ts}); rows_ts.append(dt)
                
                ys_m = sub_grp["Standard YS min"].max() if "Standard YS min" in sub_grp else 0
                p_ys, s_ys, r_ys = g_risk("YS", ys_m)
                dy = b_dict.copy(); dy.update({"Pred YS": f"{p_ys:.0f}", "Worst Case": f"{s_ys:.0f}", "Limit": f"≥ {ys_m:.0f}" if ys_m > 0 else "-", "Status": r_ys}); rows_ys.append(dy)

                el_m = sub_grp["Standard EL min"].max() if "Standard EL min" in sub_grp else 0
                p_el, s_el, r_el = g_risk("EL", el_m)
                de = b_dict.copy(); de.update({"Pred EL": f"{p_el:.1f}", "Worst Case": f"{s_el:.1f}", "Limit": f"≥ {el_m:.1f}" if el_m > 0 else "-", "Status": r_el}); rows_el.append(de)
            except: pass

        if rows_ts:
            def sr(val): return 'color: red; font-weight: bold' if "🔴" in val else 'color: green; font-weight: bold'
            c_top1, c_top2 = st.columns(2)
            with c_top1: st.markdown("##### 🔹 Tensile Strength (TS)"); st.dataframe(pd.DataFrame(rows_ts).style.applymap(sr, subset=["Status"]), use_container_width=True, hide_index=True)
            with c_top2: st.markdown("##### 🔸 Yield Strength (YS)"); st.dataframe(pd.DataFrame(rows_ys).style.applymap(sr, subset=["Status"]), use_container_width=True, hide_index=True)
            st.markdown("##### 🔻 Elongation (EL)"); st.dataframe(pd.DataFrame(rows_el).style.applymap(sr, subset=["Status"]), use_container_width=True, hide_index=True)
    st.stop()

# ==============================================================================
# 👑 MASTER DICTIONARY EXPORT (FULL VIEW)
# ==============================================================================
if view_mode == "👑 Master Dictionary Export":
    st.markdown("---")
    st.header("👑 Master Mechanical Properties Dictionary (A118T)")
    
    col_sig1, col_sig2 = st.columns(2)
    with col_sig1: target_k = st.number_input("🎯 Target Zone Multiplier (Default: 1.0 σ)", value=1.0, step=0.1, key="k_target")
    with col_sig2: control_k = st.number_input("🚧 Control Limit Multiplier (Default: 3.0 σ)", value=3.0, step=0.5, key="k_control")
    
    if st.button("🚀 Generate & Download Master Dictionary", type="primary"):
        master_data = []
        clean_master_df = df_master_full.dropna(subset=['Hardness_LINE', 'TS', 'YS', 'EL'])
        
        for keys, group in clean_master_df.groupby(GROUP_COLS):
            valid_coils_count = len(group)
            if valid_coils_count < 3: continue 
            
            mean_hrb = group['Hardness_LINE'].mean()
            std_hrb = group['Hardness_LINE'].std() if len(group) > 1 else 0
            mrs = np.abs(np.diff(group['Hardness_LINE'].values)) 
            sigma_imr = np.mean(mrs) / 1.128 if len(mrs) > 0 else std_hrb 
            
            t_min, t_max = mean_hrb - (target_k * std_hrb), mean_hrb + (target_k * std_hrb)
            c_min, c_max = mean_hrb - (control_k * std_hrb), mean_hrb + (control_k * std_hrb)
            imr_min, imr_max = mean_hrb - (control_k * sigma_imr), mean_hrb + (control_k * sigma_imr)
            
            ts_mu, ts_sig = group['TS'].mean(), group['TS'].std() if valid_coils_count > 1 else 0
            ys_mu, ys_sig = group['YS'].mean(), group['YS'].std() if valid_coils_count > 1 else 0
            el_mu, el_sig = group['EL'].mean(), group['EL'].std() if valid_coils_count > 1 else 0
            
            target_group = group[(group['Hardness_LINE'] >= t_min) & (group['Hardness_LINE'] <= t_max)]
            if len(target_group) > 0:
                curr_min = group['Limit_Min'].max() if 'Limit_Min' in group.columns else 0
                curr_max = group['Limit_Max'].min() if 'Limit_Max' in group.columns else 0
                curr_limit_str = f"{curr_min:.0f} ~ {curr_max:.0f}" if (0 < curr_max < 9000) else (f"≥ {curr_min:.0f}" if curr_min > 0 else "N/A")
                
                exp_ts_min, exp_ts_max = target_group['TS'].mean() - (control_k * target_group['TS'].std()), target_group['TS'].mean() + (control_k * target_group['TS'].std())
                exp_ys_min, exp_ys_max = target_group['YS'].mean() - (control_k * target_group['YS'].std()), target_group['YS'].mean() + (control_k * target_group['YS'].std())
                exp_el_min, exp_el_max = max(0, target_group['EL'].mean() - (control_k * target_group['EL'].std())), target_group['EL'].mean() + (control_k * target_group['EL'].std())

                master_dict = {col: (keys[idx] if isinstance(keys, tuple) else keys) for idx, col in enumerate(GROUP_COLS)}
                master_dict.update({
                    "Current HRB Limit": curr_limit_str, "Valid Coils (N)": valid_coils_count,
                    "Std Control Limit (HRB)": f"{c_min:.1f} ~ {c_max:.1f}", "I-MR Limit (HRB)": f"{imr_min:.1f} ~ {imr_max:.1f}",
                    "🎯 TARGET LIMIT (HRB)": f"{t_min:.1f} ~ {t_max:.1f}",
                    "TS Control": f"{ts_mu - control_k*ts_sig:.0f} ~ {ts_mu + control_k*ts_sig:.0f}", "Expected TS": f"{exp_ts_min:.0f} ~ {exp_ts_max:.0f}",
                    "YS Control": f"{ys_mu - control_k*ys_sig:.0f} ~ {ys_mu + control_k*ys_sig:.0f}", "Expected YS": f"{exp_ys_min:.0f} ~ {exp_ys_max:.0f}",
                    "EL Control": f"{max(0, el_mu - control_k*el_sig):.1f} ~ {el_mu + control_k*el_sig:.1f}", "Expected EL": f"{exp_el_min:.1f} ~ {exp_el_max:.1f}"
                })
                master_data.append(master_dict)
        
        if len(master_data) > 0:
            output_buffer = BytesIO()
            pd.DataFrame(master_data).to_excel(output_buffer, index=False)
            st.success(f"✅ Dictionary successfully generated for **{len(master_data)} product groups**.")
            st.download_button("📥 Download Master Report (Excel)", data=output_buffer.getvalue(), file_name=f"Master_Dictionary_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.ms-excel")
        else: st.warning("Not enough data to generate dictionary.")
    st.stop()

# ==============================================================================
# MAIN LOOP FOR ALL OTHER VIEWS 
# ==============================================================================
for i, (_, g) in enumerate(valid.iterrows()):
    conditions = [df[col] == g[col] for col in GROUP_COLS]
    sub = df[np.logical_and.reduce(conditions)].sort_values("COIL_NO")

    lo, hi = sub["Limit_Min"].iloc[0], sub["Limit_Max"].iloc[0]

    sub["NG_LAB"] = (sub.get("Hardness_LAB", sub["Hardness_LINE"]) < lo) | (sub.get("Hardness_LAB", sub["Hardness_LINE"]) > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["NG"] = sub["NG_LAB"] | sub["NG_LINE"] 

    group_title = " | ".join([f"{str(g[c])}" for c in GROUP_COLS])
    st.markdown("---")
    st.markdown(f"### 🧱 {group_title}")
    st.markdown(f"**Coils:** {sub['COIL_NO'].nunique()} | **Std Limit:** {lo:.1f} ~ {hi:.1f}")
        
    if view_mode == "📋 Data Inspection":
        def highlight_ng_rows(row): return ['background-color: #ffe6e6'] * len(row) if row['NG'] else [''] * len(row)
        num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(sub.style.format("{:.0f}", subset=[c for c in num_cols if c not in ['Limit_Min', 'Limit_Max', 'Order_Gauge']]).apply(highlight_ng_rows, axis=1), use_container_width=True)

    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        tab_trend, tab_dist = st.tabs(["📈 Trend Analysis", "📊 Distribution & SPC"])
        
        with tab_trend:
            x = np.arange(1, len(sub)+1)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            if "Hardness_LAB" in sub.columns and not sub["Hardness_LAB"].isna().all(): ax.plot(x, sub["Hardness_LAB"], marker="o", linewidth=2, label="LAB", alpha=0.5)
            ax.plot(x, sub["Hardness_LINE"], marker="s", linewidth=2, label="LINE", alpha=0.9) 
            
            ax.axhline(lo, linestyle="--", linewidth=2, color="red", label=f"Std LSL={lo}")
            ax.axhline(hi, linestyle="--", linewidth=2, color="red", label=f"Std USL={hi}")
            ax.axhline(TARGET_MIN, linestyle="--", linewidth=2, color="green", label=f"Target LSL={TARGET_MIN}")
            ax.axhline(TARGET_MAX, linestyle="--", linewidth=2, color="green", label=f"Target USL={TARGET_MAX}")
            ax.fill_between(x, TARGET_MIN, TARGET_MAX, color="green", alpha=0.1, label="Target Zone")
            
            ax.set_title("Hardness Trend by Coil Sequence", weight="bold")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=5)
            plt.tight_layout(); st.pyplot(fig)
            
        with tab_dist:
            line = sub["Hardness_LINE"].dropna()
            lab = sub["Hardness_LAB"].dropna() if "Hardness_LAB" in sub.columns else pd.Series(dtype=float)
            
            if len(line) < 5: st.warning("⚠️ Cần ít nhất 5 cuộn thép để phân tích phân phối.")
            else:
                spc_line = None
                if len(line) >= 2 and line.std(ddof=1) > 0:
                    mean, std = line.mean(), line.std(ddof=1)
                    cp = (hi - lo) / (6 * std)
                    ca = ((mean - (hi + lo) / 2) / ((hi - lo) / 2)) * 100
                    cpu, cpl = (hi - mean) / (3 * std), (mean - lo) / (3 * std)
                    spc_line = (mean, std, cp, ca, min(cpu, cpl))

                mean_line, std_line = line.mean(), line.std(ddof=1)
                
                vals = [line.min(), line.max(), lo, hi, TARGET_MIN, TARGET_MAX]
                if not lab.empty: vals.extend([lab.min(), lab.max()])
                x_min, x_max = min(vals) - 2, max(vals) + 2
                
                fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
                ax_dist.hist(line, bins=np.linspace(x_min, x_max, 30), density=True, alpha=0.6, color="#ff7f0e", edgecolor="white", label="LINE Hist")
                if not lab.empty: ax_dist.hist(lab, bins=np.linspace(x_min, x_max, 30), density=True, alpha=0.3, color="#1f77b4", edgecolor="None", label="LAB Hist")
                
                if std_line > 0:
                    xs = np.linspace(x_min, x_max, 400)
                    ax_dist.plot(xs, (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2), linewidth=2.5, color="#b25e00", label="LINE Fit")
                
                ax_dist.axvline(lo, linestyle="--", linewidth=2, color="red", label="Std LSL")
                ax_dist.axvline(hi, linestyle="--", linewidth=2, color="red", label="Std USL")
                ax_dist.axvline(TARGET_MIN, linestyle=":", linewidth=2, color="green", label="Target LSL")
                ax_dist.axvline(TARGET_MAX, linestyle=":", linewidth=2, color="green", label="Target USL")
                ax_dist.axvspan(TARGET_MIN, TARGET_MAX, color="green", alpha=0.1)

                ax_dist.set_xlim(x_min, x_max)
                ax_dist.set_title("Hardness Distribution (LINE vs LAB)", weight="bold")
                ax_dist.legend(); ax_dist.grid(alpha=0.3)
                st.pyplot(fig_dist)

                if spc_line:
                    mean_val, std_val, cp_val, ca_val, cpk_val = spc_line
                    eval_msg = "Excellent" if cpk_val >= 1.33 else ("Good" if cpk_val >= 1.0 else "Poor")
                    color_code = "green" if cpk_val >= 1.33 else ("orange" if cpk_val >= 1.0 else "red")
                    df_spc = pd.DataFrame([{"N": len(line), "Mean": mean_val, "Std": std_val, "Cp": cp_val, "Ca (%)": ca_val, "Cpk": cpk_val, "Rating": eval_msg}])
                    st.dataframe(df_spc.style.format("{:.2f}", subset=["Mean", "Std", "Cp", "Ca (%)", "Cpk"]).applymap(lambda v: f'color: {color_code}; font-weight: bold', subset=['Rating']), hide_index=True)

    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":
        if i == 0: corr_bin_summary = []
        sub_corr = sub.dropna(subset=["Hardness_LINE","TS","YS","EL"]).copy()
        
        bins = [0,56,58,60,62,65,70,75,80,85,88,92,97,100]
        labels = ["<56","56-58","58-60","60-62","62-65","65-70","70-75","75-80","80-85","85-88","88-92","92-97","≥97"]
        sub_corr["HRB_bin"] = pd.cut(sub_corr["Hardness_LINE"], bins=bins, labels=labels, right=False)
        
        summary = (sub_corr.groupby("HRB_bin", observed=True).agg(
            N_coils=("COIL_NO","count"),
            TS_mean=("TS","mean"), TS_min=("TS","min"), TS_max=("TS","max"),
            YS_mean=("YS","mean"), YS_min=("YS","min"), YS_max=("YS","max"),
            EL_mean=("EL","mean"), EL_min=("EL","min"), EL_max=("EL","max"),
            Std_TS_min=("Standard TS min", "max"), Std_TS_max=("Standard TS max", "max"),
            Std_YS_min=("Standard YS min", "max"), Std_YS_max=("Standard YS max", "max"),
            Std_EL_min=("Standard EL min", "max"), Std_EL_max=("Standard EL max", "max"),
        ).reset_index())
        summary = summary[summary["N_coils"]>0]

        if not summary.empty:
            x = np.arange(len(summary))
            fig, ax = plt.subplots(figsize=(15,6))
            
            def p_prop(x, y, ymin, ymax, c, lbl, m):
                ax.plot(x, y, marker=m, color=c, label=lbl, lw=2)
                ax.fill_between(x, ymin, ymax, color=c, alpha=0.1)
            
            p_prop(x, summary["TS_mean"], summary["TS_min"], summary["TS_max"], "#1f77b4", "TS", "o")
            p_prop(x, summary["YS_mean"], summary["YS_min"], summary["YS_max"], "#2ca02c", "YS", "s")
            p_prop(x, summary["EL_mean"], summary["EL_min"], summary["EL_max"], "#ff7f0e", "EL", "^")

            for j, row in enumerate(summary.itertuples()):
                ax.annotate(f"{row.TS_mean:.0f}", (x[j], row.TS_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="#1f77b4")
                ax.annotate(f"{row.YS_mean:.0f}", (x[j], row.YS_mean), xytext=(0,-15), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="#2ca02c")
                
                el_spec = row.Std_EL_min if pd.notna(row.Std_EL_min) else 0
                is_fail = (el_spec > 0) and (row.EL_mean < el_spec)
                ax.annotate(f"{row.EL_mean:.1f}%" + ("❌" if is_fail else ""), (x[j], row.EL_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, color="red" if is_fail else "#ff7f0e", fontweight=("bold" if is_fail else "normal"))

            ax.set_xticks(x); ax.set_xticklabels(summary["HRB_bin"])
            ax.set_title("Hardness vs Mechanical Properties", fontweight="bold"); ax.grid(True, ls="--", alpha=0.5); ax.legend(); st.pyplot(fig)

            specs_str = f"Specs: {', '.join(str(x) for x in sub['Product_Spec'].dropna().unique())}" if 'Product_Spec' in sub.columns else "Specs: N/A"

            for row in summary.itertuples():
                bin_data = sub_corr[sub_corr["HRB_bin"] == row.HRB_bin]
                corr_bin_summary.append({
                    "Specification List": specs_str, "Material": g.get("Material", "N/A"), "Gauge": g.get("Order_Gauge", "N/A"),
                    "Hardness Bin": row.HRB_bin, "N": row.N_coils,
                    "TS Spec": f"{row.Std_TS_min:.0f}~{row.Std_TS_max:.0f}" if pd.notna(row.Std_TS_max) and row.Std_TS_max < 9000 else (f"≥{row.Std_TS_min:.0f}" if pd.notna(row.Std_TS_min) else "-"),
                    "TS Actual": f"{row.TS_min:.0f}~{row.TS_max:.0f}", "TS Mean": f"{row.TS_mean:.1f}", "TS Std": f"{bin_data['TS'].std():.1f}",
                    "YS Spec": f"{row.Std_YS_min:.0f}~{row.Std_YS_max:.0f}" if pd.notna(row.Std_YS_max) and row.Std_YS_max < 9000 else (f"≥{row.Std_YS_min:.0f}" if pd.notna(row.Std_YS_min) else "-"),
                    "YS Actual": f"{row.YS_min:.0f}~{row.YS_max:.0f}", "YS Mean": f"{row.YS_mean:.1f}", "YS Std": f"{bin_data['YS'].std():.1f}",
                    "EL Spec": f"≥{row.Std_EL_min:.0f}" if pd.notna(row.Std_EL_min) else "-",
                    "EL Actual": f"{row.EL_min:.1f}~{row.EL_max:.1f}", "EL Mean": f"{row.EL_mean:.1f}", "EL Std": f"{bin_data['EL'].std():.1f}"
                })

        if i == len(valid) - 1 and 'corr_bin_summary' in locals() and len(corr_bin_summary) > 0:
            st.markdown("---")
            st.markdown(f"## 📊 Hardness Binning Comprehensive Report")
            df_full = pd.DataFrame(corr_bin_summary)
            
            def d_bin(title, cols, c_code):
                st.markdown(f"#### {title}")
                styled = df_full[["Specification List", "Material", "Gauge", "Hardness Bin", "N"] + cols].style.set_properties(**{'background-color': c_code, 'font-weight': 'bold'}, subset=[c for c in cols if "Std" in c])
                st.dataframe(styled, use_container_width=True, hide_index=True)

            d_bin("📉 TS Analysis by Hardness Bin", ["TS Spec", "TS Actual", "TS Mean", "TS Std"], "#e6f2ff")
            d_bin("📉 YS Analysis by Hardness Bin", ["YS Spec", "YS Actual", "YS Mean", "YS Std"], "#f2fff2")
            d_bin("📉 EL Analysis by Hardness Bin", ["EL Spec", "EL Actual", "EL Mean", "EL Std"], "#fff5e6")
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_full.to_excel(writer, sheet_name='All_Data', index=False)
                df_full[["Specification List", "Material", "Gauge", "Hardness Bin", "N", "TS Spec", "TS Actual", "TS Mean", "TS Std"]].to_excel(writer, sheet_name='TS_Only', index=False)
                df_full[["Specification List", "Material", "Gauge", "Hardness Bin", "N", "YS Spec", "YS Actual", "YS Mean", "YS Std"]].to_excel(writer, sheet_name='YS_Only', index=False)
                df_full[["Specification List", "Material", "Gauge", "Hardness Bin", "N", "EL Spec", "EL Actual", "EL Mean", "EL Std"]].to_excel(writer, sheet_name='EL_Only', index=False)
                for s in writer.sheets:
                    writer.sheets[s].set_column('A:A', 25); writer.sheets[s].set_column('B:C', 15); writer.sheets[s].set_column('D:Z', 12) 
            
            st.download_button("📥 Export Binning Report (Excel)", data=output.getvalue(), file_name=f"Hardness_Bin_Report_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    elif view_mode == "⚙️ Mech Props Analysis":
        if i == 0: ts_summary, ys_summary, el_summary = [], [], []

        props_config = [
            {"col": "TS", "name": "Tensile Strength (TS)", "color": "#1f77b4", "min_c": "Standard TS min", "max_c": "Standard TS max"},
            {"col": "YS", "name": "Yield Strength (YS)", "color": "#2ca02c", "min_c": "Standard YS min", "max_c": "Standard YS max"},
            {"col": "EL", "name": "Elongation (EL)", "color": "#ff7f0e", "min_c": "Standard EL min", "max_c": "Standard EL max"}
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        has_data = False
        
        h_data = sub["Hardness_LINE"].dropna()
        hrb_rng = f"{h_data.min():.1f} ~ {h_data.max():.1f}" if not h_data.empty else "N/A"
        
        for j, cfg in enumerate(props_config):
            col = cfg["col"]
            data = sub[col].dropna()
            
            if not data.empty:
                has_data = True
                mean, std = data.mean(), data.std() if len(data) > 1 else 0
                
                spec_min = sub[cfg["min_c"]].max() if cfg["min_c"] in sub.columns else 0
                spec_max = sub[cfg["max_c"]].min() if cfg["max_c"] in sub.columns else 0
                if pd.isna(spec_min): spec_min = 0
                if pd.isna(spec_max): spec_max = 0
                
                lcl_3s, ucl_3s = mean - 3 * std, mean + 3 * std
                
                axes[j].hist(data, bins=15, color=cfg["color"], alpha=0.5, density=True)
                if std > 0:
                    x_p = np.linspace(data.min() - 3*std, data.max() + 3*std, 200)
                    axes[j].plot(x_p, (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_p-mean)/std)**2), color=cfg["color"], lw=2)
                
                if spec_min > 0: axes[j].axvline(spec_min, color="red", linestyle="--", linewidth=2)
                if spec_max > 0 and spec_max < 9000: axes[j].axvline(spec_max, color="red", linestyle="--", linewidth=2)
                axes[j].axvline(lcl_3s, color="blue", linestyle=":", linewidth=1.5)
                axes[j].axvline(ucl_3s, color="blue", linestyle=":", linewidth=1.5)
                
                axes[j].set_title(f"{cfg['name']}\n(Mean={mean:.1f}, Std={std:.1f})", fontweight="bold")
                
                row_data = {
                    "Group": group_title, "N": len(data), "Hardness Range (HRB)": hrb_rng,
                    "Limit (Spec)": f"{spec_min:.0f}~{spec_max:.0f}" if (spec_max > 0 and spec_max < 9000) else f"≥ {spec_min:.0f}",
                    "Actual Range": f"{data.min():.1f}~{data.max():.1f}",
                    "Mean": f"{mean:.1f}", "Std Dev": f"{std:.1f}", "LCL (-3σ)": f"{lcl_3s:.1f}", "UCL (+3σ)": f"{ucl_3s:.1f}"  
                }
                if col == "TS": ts_summary.append(row_data)
                elif col == "YS": ys_summary.append(row_data)
                elif col == "EL": el_summary.append(row_data)
            else: axes[j].set_title(f"{cfg['name']}\n(No Data)")
            axes[j].grid(alpha=0.3, linestyle="--")

        if has_data: st.pyplot(fig)
        else: st.warning("⚠️ Không có dữ liệu Cơ tính (TS/YS/EL) cho nhóm này.")

        if i == len(valid) - 1:
            st.markdown("---")
            st.markdown("## 📊 Mechanical Properties Comprehensive Report")
            
            def d_sum(title, data_list, c_code):
                if data_list:
                    st.markdown(f"#### {title}")
                    styled_df = pd.DataFrame(data_list).style.set_properties(**{'font-weight': 'bold'}, subset=['Mean']) \
                                        .set_properties(**{'background-color': '#f0f8ff', 'font-weight': 'bold', 'color': '#0056b3'}, subset=['Hardness Range (HRB)']) \
                                        .set_properties(**{'background-color': c_code, 'color': '#004085'}, subset=['LCL (-3σ)', 'UCL (+3σ)'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

            d_sum("1️⃣ Tensile Strength (TS)", ts_summary, "#e6f2ff") 
            d_sum("2️⃣ Yield Strength (YS)", ys_summary, "#f2fff2")   
            d_sum("3️⃣ Elongation (EL)", el_summary, "#fff5e6")        

            if ts_summary or ys_summary or el_summary:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    if ts_summary: pd.DataFrame(ts_summary).to_excel(writer, sheet_name='TS_Summary', index=False)
                    if ys_summary: pd.DataFrame(ys_summary).to_excel(writer, sheet_name='YS_Summary', index=False)
                    if el_summary: pd.DataFrame(el_summary).to_excel(writer, sheet_name='EL_Summary', index=False)
                st.download_button("📥 Export Full Mech Report (Excel)", data=output.getvalue(), file_name=f"Mech_Report_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    elif view_mode == "🔍 Lookup: Hardness Range → Actual Mech Props":
        c1, c2 = st.columns(2)
        actual_min = float(sub["Hardness_LINE"].min()) if not sub["Hardness_LINE"].empty else 0.0
        actual_max = float(sub["Hardness_LINE"].max()) if not sub["Hardness_LINE"].empty else 100.0
        mn = c1.number_input("Min HRB", value=actual_min, step=0.5, key=f"lk1_{i}")
        mx = c2.number_input("Max HRB", value=actual_max, step=0.5, key=f"lk2_{i}")
        filt = sub[(sub["Hardness_LINE"] >= mn) & (sub["Hardness_LINE"] <= mx)].dropna(subset=["TS", "YS", "EL"])
        if not filt.empty: 
            st.success(f"✅ Found {len(filt)} coils.")
            st.dataframe(filt[["TS", "YS", "EL"]].describe().T.style.format("{:.1f}"), use_container_width=True)
        else: st.error("No coils found.")

    elif view_mode == "🎯 Find Target Hardness (Reverse Lookup)":
        c1, c2, c3 = st.columns(3)
        r_ys_min = c1.number_input("Min YS", value=float(sub['YS'].min()) if not sub['YS'].isna().all() else 0.0, step=5.0, key=f"ymin_{i}")
        r_ys_max = c1.number_input("Max YS", value=float(sub['YS'].max()) if not sub['YS'].isna().all() else 1000.0, step=5.0, key=f"ymax_{i}")
        r_ts_min = c2.number_input("Min TS", value=float(sub['TS'].min()) if not sub['TS'].isna().all() else 0.0, step=5.0, key=f"tmin_{i}")
        r_ts_max = c2.number_input("Max TS", value=float(sub['TS'].max()) if not sub['TS'].isna().all() else 1000.0, step=5.0, key=f"tmax_{i}")
        r_el_min = c3.number_input("Min EL", value=float(sub['EL'].min()) if not sub['EL'].isna().all() else 0.0, step=1.0, key=f"emin_{i}")
        r_el_max = c3.number_input("Max EL", value=float(sub['EL'].max()) if not sub['EL'].isna().all() else 100.0, step=1.0, key=f"emax_{i}")

        filtered = sub[(sub['YS'] >= r_ys_min) & (sub['YS'] <= r_ys_max) & (sub['TS'] >= r_ts_min) & (sub['TS'] <= r_ts_max) & (sub['EL'] >= r_el_min) & (sub['EL'] <= r_el_max)]
        if not filtered.empty:
            st.success(f"✅ Target Hardness: **{filtered['Hardness_LINE'].min():.1f} ~ {filtered['Hardness_LINE'].max():.1f} HRB**")
            st.dataframe(filtered[['COIL_NO','Hardness_LINE','YS','TS','EL']], height=300)
        else: st.error("❌ No coils found matching these specs.")

    elif view_mode == "🧮 Predict TS/YS/EL from Std Hardness":
        st.markdown(f"#### 🧮 AI Prediction Engine: {group_title}")
        train_df = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])
        
        if len(train_df) < 3:
            st.warning("⚠️ Cần ít nhất 3 cuộn thép có đủ số liệu cơ tính để kích hoạt AI Prediction.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                mean_h = train_df["Hardness_LINE"].mean()
                target_h = st.number_input("🎯 Target Hardness", value=float(round(mean_h, 1)), step=0.1, key=f"ai_{i}")
            
            X_train = train_df[["Hardness_LINE"]].values
            preds = {}
            model_metrics = {}
            
            for col in ["TS", "YS", "EL"]:
                model = LinearRegression().fit(X_train, train_df[col].values)
                val = model.predict([[target_h]])[0]
                preds[col] = val 
                y_true = train_df[col].values
                y_pred = model.predict(X_train)
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                model_metrics[col] = {"r2": r2, "rmse": rmse}

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            colors = {"TS": "#2980b9", "YS": "#27ae60", "EL": "#c0392b"} 
            idx = list(range(len(train_df)))
            nxt = len(train_df)

            for col in ["TS", "YS", "EL"]:
                sec = (col == "EL")
                fig.add_trace(go.Scatter(
                    x=idx, y=train_df[col], mode='lines', line=dict(color=colors[col], width=2, shape='spline'), 
                    name=f"{col} (History)", opacity=0.6, hoverinfo='y' 
                ), secondary_y=sec)
                
                last_val_raw = train_df[col].iloc[-1]
                pred_clean = round(preds[col], 1) if col == "EL" else int(round(preds[col]))
                last_clean = round(last_val_raw, 1) if col == "EL" else int(round(last_val_raw))
                
                fig.add_trace(go.Scatter(
                    x=[idx[-1], nxt], y=[last_val_raw, preds[col]], mode='lines',
                    line=dict(color=colors[col], width=2, dash='dot'), showlegend=False, hoverinfo='skip'
                ), secondary_y=sec)

                fig.add_trace(go.Scatter(
                    x=[nxt], y=[preds[col]], mode='markers+text', text=[f"<b>{pred_clean}</b>"], 
                    textposition="middle right" if nxt < 10 else "top center",
                    marker=dict(color=colors[col], size=14, symbol='diamond', line=dict(width=2, color='white')), 
                    name=f"Pred {col}",
                    hovertemplate=(f"<b>🎯 Pred {col}: {pred_clean}</b><br>🔙 Last {col}: {last_clean}<br>📈 Change: {pred_clean - last_clean:.1f}<extra></extra>")
                ), secondary_y=sec)

            fig.add_vline(x=nxt - 0.5, line_width=1, line_dash="dash", line_color="gray")
            fig.add_annotation(x=nxt - 0.5, y=1.05, yref="paper", text="Forecast Zone ➔", showarrow=False, font=dict(color="gray"))

            fig.update_layout(
                height=500, title=dict(text=f"📈 AI Prediction at Target Hardness = {target_h}", font=dict(size=18)),
                plot_bgcolor="white", hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=80, b=20)
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', title="Coil Sequence")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', secondary_y=False, title="Strength (MPa)")
            fig.update_yaxes(showgrid=False, secondary_y=True, title="Elongation (%)")

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### 🏁 Forecast Summary & Confidence Score")
            c1, c2, c3 = st.columns(3)
            def get_delta(p, l): return round(p - l, 1)
            last_ts = train_df["TS"].iloc[-1]; last_ys = train_df["YS"].iloc[-1]; last_el = train_df["EL"].iloc[-1]

            c1.metric("Tensile Strength (TS)", f"{int(round(preds['TS']))} MPa", f"{get_delta(preds['TS'], last_ts)} vs Last")
            c1.caption(f"🎯 **R² Score:** {model_metrics['TS']['r2']:.2f} | **Sai số (RMSE):** ±{model_metrics['TS']['rmse']:.1f}")

            c2.metric("Yield Strength (YS)", f"{int(round(preds['YS']))} MPa", f"{get_delta(preds['YS'], last_ys)} vs Last")
            c2.caption(f"🎯 **R² Score:** {model_metrics['YS']['r2']:.2f} | **Sai số (RMSE):** ±{model_metrics['YS']['rmse']:.1f}")

            c3.metric("Elongation (EL)", f"{round(preds['EL'], 1)} %", f"{get_delta(preds['EL'], last_el)} vs Last")
            c3.caption(f"🎯 **R² Score:** {model_metrics['EL']['r2']:.2f} | **Sai số (RMSE):** ±{model_metrics['EL']['rmse']:.1f}")

    elif view_mode == "🎛️ Control Limit Calculator (Compare 3 Methods)":
        
        # --- 1. HIỂN THỊ GIẢI THÍCH DUY NHẤT MỘT LẦN Ở ĐẦU VIEW ---
        if i == 0:
            all_groups_summary = []
            st.markdown("### 📘 管制界限計算方法說明 (Method Explanation)")
            with st.expander("🔍 點擊查看方法差異 (Click to view method details)", expanded=True):
                st.markdown("""
                | 方法 (Method) | 名稱 (Name) | 運作原理 (Description) |
                | :--- | :--- | :--- |
                | **M1: Standard** | **標準統計法** | 基於全體數據計算。若存在極端異常值，界限容易被過度拉伸。 |
                | **M2: IQR Robust** | **四分位距穩健統計法** | 自動剔除因操作失誤產生的「極端值」，使管制界限更符合實際規律。 |
                | **M3: Smart Hybrid** | **智能混合法** | 結合統計趨勢與客戶規範 (Spec)，確保管制區間始終在安全範圍內。 |
                | **M4: I-MR (SPC)** | **專業製程管制** | **最佳化方案：** 觀測相鄰鋼捲間的波動，是判斷製程是否「穩定」最科學的方法。 |
                """)

        st.markdown(f"### 🎛️ Control Limits Analysis: {group_title}")
        data = sub["Hardness_LINE"].dropna()
        data_lab = sub["Hardness_LAB"].dropna() if "Hardness_LAB" in sub.columns else pd.Series(dtype=float)
        
        if len(data) < 5: 
            st.warning(f"⚠️ Dữ liệu không đủ để phân tích (N={len(data)})")
        else:
            with st.expander("⚙️ 設定參數 (Settings)", expanded=False):
                c1, c2 = st.columns(2)
                sigma_n = c1.number_input("1. Sigma Multiplier (K)", 1.0, 6.0, 3.0, 0.5, key=f"sig_{i}")
                iqr_k = c2.number_input("2. IQR Sensitivity", 0.5, 3.0, 0.7, 0.1, key=f"iqr_{i}")

            spec_min = lo
            spec_max = hi
            display_max = spec_max if (spec_max > 0 and spec_max < 9000) else 0
            
            mu = data.mean()
            std_dev = data.std()
            
            # M1: Standard
            m1_min, m1_max = mu - sigma_n*std_dev, mu + sigma_n*std_dev
            
            # M2: IQR Robust
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            clean_data = data[~((data < (Q1 - iqr_k * IQR)) | (data > (Q3 + iqr_k * IQR)))]
            if clean_data.empty: clean_data = data
            mu_clean, sigma_clean = clean_data.mean(), clean_data.std()
            if pd.isna(sigma_clean) or sigma_clean == 0: sigma_clean = std_dev
            m2_min, m2_max = mu_clean - sigma_n*sigma_clean, mu_clean + sigma_n*sigma_clean
            
            # M3: Smart Hybrid
            m3_min = max(m2_min, spec_min)
            m3_max = min(m2_max, spec_max) if (spec_max > 0 and spec_max < 9000) else m2_max
            if m3_min >= m3_max: m3_min, m3_max = m2_min, m2_max
            
            # M4: I-MR (SPC) - Tối ưu cho thép cuộn
            mrs = np.abs(np.diff(data))
            mr_bar = np.mean(mrs) if len(mrs) > 0 else 0
            sigma_imr = mr_bar / 1.128 if mr_bar > 0 else std_dev
            m4_min, m4_max = mu - sigma_n * sigma_imr, mu + sigma_n * sigma_imr

            spec_str = f"Ctrl: {spec_min:.0f}~{display_max:.0f}"

            all_groups_summary.append({
                "Group": group_title,
                "N": len(data),
                "Current Spec": spec_str,
                "M1: Standard": f"{m1_min:.1f} ~ {m1_max:.1f}",
                "M2: IQR (Robust)": f"{m2_min:.1f} ~ {m2_max:.1f}",
                "M3: Smart Hybrid": f"{m3_min:.1f} ~ {m3_max:.1f}", 
                "M4: I-MR (Optimal)": f"{m4_min:.1f} ~ {m4_max:.1f}",
                "Status": "✅ Stable" if (display_max > 0 and m4_max <= display_max) else "⚠️ Narrow Spec"
            })
            
            # ==================================================================
            # BIỂU ĐỒ 1: LIMITS COMPARISON (FULL WIDTH)
            # ==================================================================
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.hist(data, bins=15, density=True, alpha=0.6, color="#1f77b4", label="LINE (Production)")
            if not data_lab.empty: ax.hist(data_lab, bins=15, density=True, alpha=0.4, color="#ff7f0e", label="LAB (Ref)")
            
            ax.axvline(m1_min, c="red", ls=":", alpha=0.4, label="M1: Standard")
            ax.axvline(m1_max, c="red", ls=":", alpha=0.4)
            ax.axvline(m2_min, c="blue", ls="--", alpha=0.5, label="M2: IQR")
            ax.axvline(m2_max, c="blue", ls="--", alpha=0.5)
            ax.axvline(m4_min, c="purple", ls="-.", lw=2, label="M4: I-MR (SPC)")
            ax.axvline(m4_max, c="purple", ls="-.", lw=2)
            ax.axvspan(m3_min, m3_max, color="green", alpha=0.15, label="M3: Hybrid Zone")
            
            if spec_min > 0: ax.axvline(spec_min, c="black", lw=2)
            if display_max > 0: ax.axvline(display_max, c="black", lw=2)
            
            ax.set_title(f"Limits Comparison (σ={sigma_n})", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize="small")
            st.pyplot(fig)

            # ==================================================================
            # BIỂU ĐỒ 2: CHI TIẾT M1 VS M4 VS SPECS
            # ==================================================================
            st.write("---") 
            st.markdown(f"#### 📊 Detailed Distribution Analysis")
            
            from scipy.stats import norm
            n_samples = len(data)
            bins_sturges = int(round(1 + 3.322 * np.log10(n_samples))) if n_samples > 0 else 10
            
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.hist(data, bins=bins_sturges, density=True, alpha=0.2, color="#1f77b4", label="LINE Actual")
            
            x_min_val = min([m1_min, m4_min, spec_min, data.min()]) - 5
            x_max_val = max([m1_max, m4_max, display_max, data.max()]) + 5
            x_axis = np.linspace(x_min_val, x_max_val, 500)
            
            ax2.plot(x_axis, norm.pdf(x_axis, mu, std_dev), color="red", lw=2, label=f"M1 Curve (σ={std_dev:.2f})")
            ax2.plot(x_axis, norm.pdf(x_axis, mu, sigma_imr), color="purple", lw=2, ls="--", label=f"M4 Curve (σ={sigma_imr:.2f})")

            ax2.axvline(m1_min, color="red", ls=":", lw=1.5); ax2.axvline(m1_max, color="red", ls=":", lw=1.5)
            ax2.axvline(m4_min, color="purple", ls="-.", lw=2); ax2.axvline(m4_max, color="purple", ls="-.", lw=2)
            
            if spec_min > 0: ax2.axvline(spec_min, color="black", lw=2.5, label="Control Spec")
            if display_max > 0: ax2.axvline(display_max, color="black", lw=2.5)

            ax2.xaxis.set_major_locator(plt.MultipleLocator(5))
            ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
            ax2.grid(which='both', axis='x', linestyle='--', alpha=0.3)
            ax2.set_title(f"Detailed Analysis (Sturges k={bins_sturges})", fontsize=11, fontweight="bold")
            ax2.legend(loc="upper right", fontsize="small")
            st.pyplot(fig2)

            # ==================================================================
            # 3. SUMMARY TABLE & EXCEL EXPORT (DỰ PHÓNG CƠ TÍNH)
            # ==================================================================
            st.write("---") 
            st.markdown(f"#### 📌 Limit Summary & Mechanical Estimation")
            
            # Hàm nội suy cơ tính từ độ cứng A118T
            def get_mech(h_val):
                try:
                    h = float(h_val)
                    if h <= 0 or pd.isna(h): return 0, 0, 0
                    ts = 5.5 * h + 75
                    ys = ts * 0.75
                    el = 100 - (1.1 * h)
                    return ts, ys, el
                except: return 0, 0, 0

            target_k = 1.0 
            new_target_min = mu - target_k * sigma_imr
            new_target_max = mu + target_k * sigma_imr

            rows = []
            configs = [
                ("🎯 Old Target Goal", spec_min, display_max, "-"),
                ("🔴 M1: Standard (Historical)", m1_min, m1_max, std_dev),
                ("🔵 M2: IQR (Robust)", m2_min, m2_max, sigma_clean),
                ("🟣 M4: I-MR (Control Limits)", m4_min, m4_max, sigma_imr),
                (f"🌟 New Core Target (±{target_k}σ)", new_target_min, new_target_max, "-")
            ]

            for cat, l_min, l_max, sig in configs:
                ts_lmin, ys_lmin, el_lmax = get_mech(l_min)
                ts_lmax, ys_lmax, el_lmin = get_mech(l_max)
                
                valid_data = data[(data >= l_min) & (data <= l_max)] if l_max > 0 else []
                
                if len(valid_data) > 0:
                    act_min, act_max = valid_data.min(), valid_data.max()
                    ts_amin, ys_amin, el_amax = get_mech(act_min)
                    ts_amax, ys_amax, el_amin = get_mech(act_max)
                    
                    act_ts = f"{ts_amin:.0f} ~ {ts_amax:.0f}"
                    act_ys = f"{ys_amin:.0f} ~ {ys_amax:.0f}"
                    act_el = f"{el_amax:.1f} ~ {el_amin:.1f}"
                else:
                    act_ts = act_ys = act_el = "N/A"

                rows.append({
                    "Limit Type": cat,
                    "Hardness Limits": f"{l_min:.1f} ~ {l_max:.1f}",
                    "Variation": f"σ={sig:.2f}" if isinstance(sig, float) else sig,
                    "Theoretical TS": f"{ts_lmin:.0f} ~ {ts_lmax:.0f}",
                    "Actual TS": act_ts,
                    "Theoretical YS": f"{ys_lmin:.0f} ~ {ys_lmax:.0f}",
                    "Actual YS": act_ys,
                    "Theoretical EL (%)": f"{el_lmax:.1f} ~ {el_lmin:.1f}",
                    "Actual EL (%)": act_el
                })

            df_summary = pd.DataFrame(rows)
            
            def highlight_new_target(s):
                if "🌟 New Core Target" in s['Limit Type']:
                    return ['background-color: #d4edda; font-weight: bold; color: #155724'] * len(s)
                return [''] * len(s)

            st.dataframe(
                df_summary.style.apply(highlight_new_target, axis=1), 
                use_container_width=True, 
                hide_index=True
            )
            st.caption("*(**) TS: Tensile Strength (MPa) | YS: Yield Strength (MPa) | EL: Elongation (%)*")

            # EXCEL EXPORT BUTTON
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                worksheet = writer.sheets['Summary']
                for idx, col_name in enumerate(df_summary.columns):
                    max_len = max(df_summary[col_name].astype(str).map(len).max(), len(col_name)) + 2
                    worksheet.set_column(idx, idx, max_len)

            safe_group_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', group_title)
            st.download_button(
                label="📥 Download Summary as Excel",
                data=buffer.getvalue(),
                file_name=f"Mech_Estimation_{safe_group_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_sum_{i}"
            )

        # --- HIỂN THỊ BẢNG TỔNG HỢP TOÀN BỘ Ở CUỐI TRANG ---
        if i == len(valid) - 1 and 'all_groups_summary' in locals() and len(all_groups_summary) > 0:
            st.markdown("---")
            st.markdown("## 📊 Summary of Control Limits")
            df_total = pd.DataFrame(all_groups_summary)
            
            def style_status(val):
                return 'color: red; font-weight: bold' if 'Narrow' in val else 'color: green; font-weight: bold'

            styled_df = (
                df_total.style
                .applymap(style_status, subset=['Status'])
                .set_properties(**{'background-color': '#e6f2ff', 'color': '#004085', 'font-weight': 'bold', 'border': '2px solid #0056b3'}, subset=['M4: I-MR (Optimal)'])
            )
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            st.download_button("📥 Export Complete SPC Summary CSV", df_total.to_csv(index=False).encode('utf-8-sig'), f"SPC_Summary_A118T.csv")
