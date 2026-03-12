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
        
        /* 調整指標卡片字體大小與換行以避免截斷 (Adjust metric card font size and wrap to prevent truncation) */
        [data-testid="stMetricValue"] { 
            background-color: white; 
            padding: 10px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
            color: #007bff; 
            font-size: 1.6rem !important; 
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            white-space: normal !important; 
        }
        
        thead tr th:first-child {display:none}
        tbody th {display:none}
        .stDataFrame { border: none !important; }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# ================================
# ================================
# DYNAMIC TARGET SETTINGS (SIDEBAR)
# ================================
st.sidebar.header("⚙️ GLOBAL SETTINGS")
st.sidebar.markdown("**🎯 Target Hardness (HRB)**")
c_t1, c_t2 = st.sidebar.columns(2)

# 加入 unique key 避免 DuplicateElementId 錯誤 (Add unique keys to avoid duplicate ID errors)
TARGET_MIN = c_t1.number_input("Target Min", value=85.0, step=0.5, format="%.1f", key="global_target_min")
TARGET_MAX = c_t2.number_input("Target Max", value=90.0, step=0.5, format="%.1f", key="global_target_max")
st.sidebar.markdown("---")

# ================================
# LOAD & CLEAN DATA
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1hC5nnxqDLjF8-wUm8gtj11_5HFMxBlogY84Z0cRCj2s/export?format=csv"

@st.cache_data
def load_main():
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    if "<!doctype html>" in r.text[:50].lower() or "<html" in r.text[:50].lower():
        st.error("🚨 SECURITY ERROR: Restricted Google Sheet link. Please go to Google Sheet -> Share -> Select 'Anyone with the link'.")
        st.stop()
    return pd.read_csv(StringIO(r.text))

raw = load_main()

# 處理日期時間 (Date/Time Processing)
data_period_str = "N/A"
date_col = next((c for c in raw.columns if 'DATE' in str(c).upper()), None)
if date_col:
    raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
    min_date = raw[date_col].min()
    max_date = raw[date_col].max()
    if pd.notna(min_date) and pd.notna(max_date):
        data_period_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"

# 絕對安全的台灣時區設定法 (Absolutely safe Taiwan timezone setting)
import datetime as dt
tz_tw = dt.timezone(dt.timedelta(hours=8))
current_time = dt.datetime.now(tz_tw).strftime("%Y-%m-%d %H:%M")

st.markdown(f"""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
    <strong>🕒 Report Generated:</strong> {current_time} &nbsp;&nbsp;|&nbsp;&nbsp; 
    <strong>📅 Data Period:</strong> {data_period_str} &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>🎯 Target Hardness:</strong> {TARGET_MIN} ~ {TARGET_MAX}
</div>
""", unsafe_allow_html=True)

# 絕對欄位標題掃描器 (Absolute Column Header Scanner)
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
    st.error(f"⚠️ Column error: Hardness_LINE missing. Available columns: {list(raw.columns)}")
    st.stop()

if "COIL_NO" not in df.columns: df["COIL_NO"] = df.index 
if "Material" not in df.columns: df["Material"] = "A118T"
if "Product_Spec" not in df.columns: df["Product_Spec"] = "N/A"

# 移除硬編碼限制，允許讀取 Google Sheet 中的所有鋼種與規格 
# (Remove hardcoded limits to allow loading all materials/specs from Google Sheet)

if df.empty:
    st.error("⚠️ No valid data found in the Google Sheet.")
    st.stop()

if df.empty:
    st.error("⚠️ No data found for A118T, 2657/G01T, or N SZACC.")
    st.stop()

# 數據處理與移除 0/NA 值 (Data processing & remove 0/NA)
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

# 將 0 替換為 NA 以移除異常值 (Replace 0 with NA to remove outliers completely)
test_cols = ["Hardness_LAB", "Hardness_LINE", "YS", "TS", "EL", 
             "Standard TS min", "Standard TS max", "Standard YS min", "Standard YS max", 
             "Standard EL min", "Standard EL max"]

for c in test_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == 0, c] = np.nan 

# 強制 Gauge 顯示兩位小數 (Force 2 decimal format for Gauge)
if "Order_Gauge" in df.columns:
    df["Order_Gauge"] = pd.to_numeric(df["Order_Gauge"], errors="coerce")
    df["Order_Gauge"] = df["Order_Gauge"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

df = df.dropna(subset=["Hardness_LINE"])

# ================================
# ================================
# ================================
# SIDEBAR FILTER & SETTINGS
# ================================
st.sidebar.header("🎛 GLOBAL SETTINGS")
st.sidebar.markdown("**🎯 Target Hardness (HRB)**")
c_t1, c_t2 = st.sidebar.columns(2)
TARGET_MIN = c_t1.number_input("Target Min", value=85.0, step=0.5, format="%.1f")
TARGET_MAX = c_t2.number_input("Target Max", value=90.0, step=0.5, format="%.1f")

st.sidebar.markdown("---")
st.sidebar.header("🔍 DATA FILTER")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

all_specs = sorted(df["Product_Spec"].dropna().astype(str).unique()) if "Product_Spec" in df else []
all_rolling = sorted(df["Rolling_Type"].dropna().astype(str).unique()) if "Rolling_Type" in df else []
all_metal = sorted(df["Metallic_Type"].dropna().astype(str).unique()) if "Metallic_Type" in df else []
all_qgroup = sorted(df["Quality_Group"].dropna().astype(str).unique()) if "Quality_Group" in df else []

all_gauge = []
if "Order_Gauge" in df.columns:
    valid_gauges = pd.to_numeric(df["Order_Gauge"], errors="coerce").dropna()
    if not valid_gauges.empty:
        all_gauge = sorted(valid_gauges.unique())

specs_filter = st.sidebar.selectbox("1. Product Specs", ["All"] + list(all_specs))
rolling = st.sidebar.selectbox("2. Rolling Type", ["All"] + list(all_rolling))
metal = st.sidebar.selectbox("3. Metallic Type", ["All"] + list(all_metal))

gauge_input = st.sidebar.text_input("4. Order Gauge (ex: 1.5 or 1.5~1.8)", value="")
if all_gauge:
    gauge_hint = ", ".join([f"{g:.2f}" for g in all_gauge[:10]]) + ("..." if len(all_gauge) > 10 else "")
    st.sidebar.caption(f"💡 **Available:** {gauge_hint}")

qgroup = st.sidebar.selectbox("5. Quality Group", ["All"] + list(all_qgroup))

# 應用過濾邏輯 (Apply filtering logic)
df_master_full = df.copy() 

if specs_filter != "All": df = df[df["Product_Spec"].astype(str) == specs_filter]
if rolling != "All": df = df[df["Rolling_Type"].astype(str) == rolling]
if metal != "All": df = df[df["Metallic_Type"].astype(str) == metal]

if gauge_input.strip() != "" and "Order_Gauge" in df:
    df["temp_gauge_num"] = pd.to_numeric(df["Order_Gauge"], errors="coerce")
    if "~" in gauge_input or "-" in gauge_input:
        sep = "~" if "~" in gauge_input else "-"
        try:
            parts = gauge_input.split(sep)
            df = df[(df["temp_gauge_num"] >= float(parts[0])) & (df["temp_gauge_num"] <= float(parts[1]))]
        except: st.sidebar.error("⚠️ Invalid gauge format")
    else:
        try: df = df[(df["temp_gauge_num"] >= float(gauge_input) - 0.001) & (df["temp_gauge_num"] <= float(gauge_input) + 0.001)]
        except: st.sidebar.error("⚠️ Invalid gauge format")
    df = df.drop(columns=["temp_gauge_num"])
    
if qgroup != "All": df = df[df["Quality_Group"].astype(str) == qgroup]

# 重要：在導航前先定義 valid 變數 (CRITICAL: Define 'valid' variable before Navigation)
GROUP_COLS = [c for c in ["Product_Spec", "Rolling_Type", "Metallic_Type", "Quality_Group", "Material", "Order_Gauge"] if c in df.columns]
if not GROUP_COLS: GROUP_COLS = ["Material"]
cnt = df.groupby(GROUP_COLS).agg(N_Coils=("COIL_NO","nunique")).reset_index()
valid = cnt[cnt["N_Coils"] >= 1]

# ================================
# NAVIGATION MENU (GROUPED)
# ================================
st.sidebar.markdown("---")
st.sidebar.header("🧭 NAVIGATION")

menu_category = st.sidebar.selectbox("📂 Select Category", ["📊 Dashboards & KPIs", "🔬 Deep Analytics", "🛠️ Tools & AI Models"])

if menu_category == "📊 Dashboards & KPIs":
    view_mode = st.sidebar.radio("📍 Select View", ["📊 Executive KPI Dashboard", "🚀 Global Summary Dashboard", "📋 Data Inspection"])
elif menu_category == "🔬 Deep Analytics":
    view_mode = st.sidebar.radio("📍 Select View", ["📉 Hardness Analysis (Trend & Dist)", "🔗 Correlation: Hardness vs Mech Props", "⚙️ Mech Props Analysis"])
else:
    view_mode = st.sidebar.radio("📍 Select View", ["🔍 Lookup: Hardness Range → Actual Mech Props", "🎯 Find Target Hardness (Reverse Lookup)", "🧮 Predict TS/YS/EL from Std Hardness", "🎛️ Control Limit Calculator (Compare 3 Methods)", "👑 Master Dictionary Export"])

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
        st.warning("⚠️ Insufficient Mechanical Properties data for the selected filters.")
    else:
        total_coils = len(df_kpi)
        
        # 縮減為 1 位小數以節省空間 (Reduce to 1 decimal place to save space)
        def clean_num(val, is_pct=False):
            if pd.isna(val): return "0%" if is_pct else "0"
            v = round(float(val), 1) 
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
        
        # 縮短標題名稱避免溢出 (Shorten labels to prevent overflow)
        col1.metric("📦 Total Coils", f"{total_coils:,}")
        col2.metric("✅ Mech Yield", clean_num(yield_rate, True), clean_num(yield_rate - 100, True) if yield_rate < 100 else "Perfect")
        col3.metric("🎯 HRB Yield", clean_num(hrb_yield, True), clean_num(hrb_yield - 100, True) if hrb_yield < 100 else "Control")
        col4.metric(f"🌟 Target Yield", clean_num(target_yield, True))
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
            
            # 包含所有機械性能規格的直方圖矩陣 (Histogram matrix including all mech specs)
            st.markdown("#### 🔔 Visual Deep Dive: Top 10 Risk Distributions (Hardness & Mechanical)")
            top_10_risks = risk_top.head(10).to_dict('records')
            
            if len(top_10_risks) > 0:
                chart_cols = st.columns(2) # 每行 2 個以提供更多空間 (2 per row for more space)
                for idx, item in enumerate(top_10_risks):
                    spec, mat, gauge_val = item.get("Specification", "N/A"), item.get("Material", "N/A"), item.get("Gauge", "N/A")
                    tdf = df_kpi[(df_kpi.get("Product_Spec", "") == spec) & (df_kpi.get("Material", "") == mat) & (df_kpi.get("Order_Gauge", "") == gauge_val)]
                    
                    if not tdf.empty:
                        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
                        fig.suptitle(f"TOP {idx+1}: {spec} | Mat: {mat} | Gauge: {gauge_val}", fontsize=12, fontweight="bold")
                        
                        # 準備 4 個圖表的設定 (Prepare configurations for 4 charts)
                        metrics = [
                            {"col": "Hardness_LINE", "name": "Hardness (HRB)", "min": tdf["Limit_Min"].iloc[0] if "Limit_Min" in tdf.columns else 0, "max": tdf["Limit_Max"].iloc[0] if "Limit_Max" in tdf.columns else 0, "ax": axes[0, 0], "color": "#1f77b4"},
                            {"col": "TS", "name": "Tensile Strength (TS)", "min": tdf["Standard TS min"].max() if "Standard TS min" in tdf.columns else 0, "max": tdf["Standard TS max"].min() if "Standard TS max" in tdf.columns else 0, "ax": axes[0, 1], "color": "#2ca02c"},
                            {"col": "YS", "name": "Yield Strength (YS)", "min": tdf["Standard YS min"].max() if "Standard YS min" in tdf.columns else 0, "max": tdf["Standard YS max"].min() if "Standard YS max" in tdf.columns else 0, "ax": axes[1, 0], "color": "#ff7f0e"},
                            {"col": "EL", "name": "Elongation (EL)", "min": tdf["Standard EL min"].max() if "Standard EL min" in tdf.columns else 0, "max": 0, "ax": axes[1, 1], "color": "#9467bd"}
                        ]
                        
                        for m in metrics:
                            ax = m["ax"]
                            d = tdf[m["col"]].dropna()
                            if not d.empty:
                                ax.hist(d, bins=15, color=m["color"], edgecolor="white", density=True, alpha=0.6)
                                m_val, s_val = d.mean(), d.std()
                                if s_val > 0:
                                    x_ax = np.linspace(d.min() - 2*s_val, d.max() + 2*s_val, 100)
                                    ax.plot(x_ax, (1/(s_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_ax - m_val) / s_val)**2), color="#cc0000", lw=2)
                                
                                # 繪製規格界限 (Plot specification limits)
                                if pd.notna(m["min"]) and m["min"] > 0:
                                    ax.axvline(m["min"], color="black", linestyle="--", lw=1.5, label=f"LSL ({m['min']:.0f})")
                                if pd.notna(m["max"]) and 0 < m["max"] < 9000:
                                    ax.axvline(m["max"], color="black", linestyle="--", lw=1.5, label=f"USL ({m['max']:.0f})")
                                
                                # 為硬度圖表添加目標區域 (Add target zone for Hardness chart)
                                if m["col"] == "Hardness_LINE":
                                    ax.axvline(TARGET_MIN, color="green", linestyle=":", lw=2, label="Target")
                                    ax.axvline(TARGET_MAX, color="green", linestyle=":", lw=2)
                                    ax.axvspan(TARGET_MIN, TARGET_MAX, color="green", alpha=0.1)

                                ax.set_title(f"{m['name']} (N={len(d)})", fontsize=10, fontweight="bold")
                                ax.legend(fontsize=7, loc="upper right")
                                ax.grid(alpha=0.3, linestyle=":")
                            else:
                                ax.set_title(f"{m['name']} (No Data)", fontsize=10)
                                ax.axis('off')
                                
                        plt.tight_layout()
                        chart_cols[idx % 2].pyplot(fig)
            
            st.markdown("#### 📑 Export Actionable Report")
            import streamlit.components.v1 as components
            col_csv, col_pdf, _ = st.columns([2, 2, 6])
            with col_csv: st.download_button("📥 Download Watchlist (CSV)", data=risk_top.to_csv(index=False).encode('utf-8-sig'), file_name="High_Risk_Watchlist.csv", mime="text/csv", use_container_width=True)
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
        else: st.warning("⚠️ Insufficient data.")

    with tab2:
        col_in1, col_in2 = st.columns([1, 1])
        with col_in1: user_hrb = st.number_input("1️⃣ Target HRB (Simulated)", value=85.0, step=0.5, format="%.1f")
        with col_in2: safety_k = st.selectbox("2️⃣ Select Safety Factor (σ):", [1.0, 2.0, 3.0], index=1)

        rows_ts, rows_ys, rows_el = [], [], []
        for _, g in valid.iterrows():
            conditions = [df[col] == g[col] for col in GROUP_COLS]
            sub_grp = df[np.logical_and.reduce(conditions)].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 3: continue 

            l_min_val = sub_grp['Limit_Min'].min() if 'Limit_Min' in sub_grp.columns else 0
            l_max_val = sub_grp['Limit_Max'].max() if 'Limit_Max' in sub_grp.columns else 0
            hrb_spec_str = f"{l_min_val:.0f}~{l_max_val:.0f}" if l_max_val > 0 else "-"

            X = sub_grp[["Hardness_LINE"]].values
            
            # 預測並掃描雙向界限 (Predict and scan dual limits)
            def eval_risk(col, sp_min, sp_max, is_el=False):
                m = LinearRegression().fit(X, sub_grp[col].values)
                pred = m.predict([[user_hrb]])[0]
                rmse = np.sqrt(mean_squared_error(sub_grp[col], m.predict(X)))
                worst = pred - (safety_k * rmse) 
                best = pred + (safety_k * rmse)  
                
                sp_min = sp_min if pd.notna(sp_min) else 0
                sp_max = sp_max if pd.notna(sp_max) else 0
                
                # 掃描風險警告 (Scan for risk warnings)
                status = "🟢 Safe"
                if sp_min > 0 and worst < sp_min: status = "🔴 Risk (Low)"
                if not is_el and 0 < sp_max < 9000 and best > sp_max: status = "🔴 Risk (High)"
                
                # 顯示機械性能規格字串 (Display Mech Spec string)
                if is_el:
                    lim_str = f"≥ {sp_min:.1f}" if sp_min > 0 else "-"
                else:
                    if sp_min > 0 and 0 < sp_max < 9000: lim_str = f"{sp_min:.0f}~{sp_max:.0f}"
                    elif sp_min > 0: lim_str = f"≥ {sp_min:.0f}"
                    elif 0 < sp_max < 9000: lim_str = f"≤ {sp_max:.0f}"
                    else: lim_str = "-"
                    
                return pred, worst, best, lim_str, status

            try:
                # 隱藏不必要的欄位，新增 HRB Spec (Hide unnecessary columns, add HRB Spec)
                b_dict = {}
                for col in GROUP_COLS:
                    if col not in ["Rolling_Type", "Metallic_Type", "Material", "Quality_Group"]:
                        b_dict[col] = g[col]
                b_dict["HRB Spec"] = hrb_spec_str
                
                # 計算 TS (Calculate TS)
                ts_m_min = sub_grp["Standard TS min"].max() if "Standard TS min" in sub_grp else 0
                ts_m_max = sub_grp["Standard TS max"].min() if "Standard TS max" in sub_grp else 0
                p_ts, w_ts, b_ts, l_ts, st_ts = eval_risk("TS", ts_m_min, ts_m_max)
                dt = b_dict.copy()
                dt.update({"Pred TS": f"{p_ts:.0f}", "Est. Range": f"{w_ts:.0f}~{b_ts:.0f}", "Mech Spec": l_ts, "Status": st_ts})
                rows_ts.append(dt)
                
                ys_m_min = sub_grp["Standard YS min"].max() if "Standard YS min" in sub_grp else 0
                ys_m_max = sub_grp["Standard YS max"].min() if "Standard YS max" in sub_grp else 0
                p_ys, w_ys, b_ys, l_ys, st_ys = eval_risk("YS", ys_m_min, ys_m_max)
                dy = b_dict.copy()
                dy.update({"Pred YS": f"{p_ys:.0f}", "Est. Range": f"{w_ys:.0f}~{b_ys:.0f}", "Mech Spec": l_ys, "Status": st_ys})
                rows_ys.append(dy)

                el_m_min = sub_grp["Standard EL min"].max() if "Standard EL min" in sub_grp else 0
                p_el, w_el, b_el, l_el, st_el = eval_risk("EL", el_m_min, 0, is_el=True)
                de = b_dict.copy()
                de.update({"Pred EL": f"{p_el:.1f}", "Est. Range": f"{w_el:.1f}~{b_el:.1f}", "Mech Spec": l_el, "Status": st_el})
                rows_el.append(de)
            except: pass

        if rows_ts:
            def sr(val): 
                if isinstance(val, str):
                    if "🔴" in val: return 'color: #721c24; font-weight: bold; background-color: #f8d7da'
                    if "🟢" in val: return 'color: #155724; font-weight: bold; background-color: #d4edda'
                return ''
            
            st.info("💡 **Note:** The **HRB Spec** column is the original hardness standard (to compare with Target HRB). The **Est. Range** is automatically compared against the **Mech Spec** to trigger bidirectional risk warnings (Low/High).")
            
            c_top1, c_top2 = st.columns(2)
            with c_top1: 
                st.markdown("##### 🔹 Tensile Strength (TS)")
                st.dataframe(pd.DataFrame(rows_ts).style.applymap(sr, subset=["Status"]), use_container_width=True, hide_index=True)
            with c_top2: 
                st.markdown("##### 🔸 Yield Strength (YS)")
                st.dataframe(pd.DataFrame(rows_ys).style.applymap(sr, subset=["Status"]), use_container_width=True, hide_index=True)
            
            st.markdown("##### 🔻 Elongation (EL)")
            st.dataframe(pd.DataFrame(rows_el).style.applymap(sr, subset=["Status"]), use_container_width=True, hide_index=True)
    st.stop()

# ==============================================================================
# ==============================================================================
# ==============================================================================
# 👑 MASTER DICTIONARY EXPORT (VIEW ON SCREEN & DOWNLOAD)
# ==============================================================================
if view_mode == "👑 Master Dictionary Export":
    st.markdown("---")
    st.header("👑 Master Mechanical Properties Dictionary (A118T)")
    st.info("💡 **Interactive View & Export:** Review the logically grouped limits directly on the screen below, then download the formatted Excel file for your records.")
    
    if st.button("🚀 Generate Comprehensive Dictionary", type="primary"):
        master_data = []
        clean_master_df = df_master_full.dropna(subset=['Hardness_LINE', 'TS', 'YS', 'EL'])
        
        # 運算基準 (Calculation Baselines)
        sigma_n = 2.0  # Control Limit (Siết chặt xuống 2 Sigma theo yêu cầu)
        target_k = 1.0 # Target Zone

        for keys, group in clean_master_df.groupby(GROUP_COLS):
            if len(group) < 3: continue 
            
            # --- 核心運算 (Core Calculations) ---
            data = group["Hardness_LINE"]
            mu = data.mean()
            mrs = np.abs(np.diff(data.values))
            sigma_imr = np.mean(mrs) / 1.128 if len(mrs) > 0 else data.std()
            
            # 1. 建議控制界限 (M4: I-MR 2σ)
            c_min, c_max = mu - sigma_n * sigma_imr, mu + sigma_n * sigma_imr
            # 2. 建議目標界限 (Target 1σ)
            t_min, t_max = mu - target_k * sigma_imr, mu + target_k * sigma_imr
            
            # AI 模型用於預測機械性能範圍 (AI Models for predicting Mech Ranges)
            X_train = group[["Hardness_LINE"]].values
            m_ts = LinearRegression().fit(X_train, group["TS"].values)
            m_ys = LinearRegression().fit(X_train, group["YS"].values)
            m_el = LinearRegression().fit(X_train, group["EL"].values)
            
            # 取得原始機械性能規格 (Original Mechanical Specs)
            s_ts_min = group["Standard TS min"].max()
            s_ts_max = group["Standard TS max"].min()
            s_ys_min = group["Standard YS min"].max()
            s_ys_max = group["Standard YS max"].min()
            s_el_min = group["Standard EL min"].max()
            
            def fmt_s(mi, ma):
                if pd.isna(mi): mi = 0
                if pd.isna(ma): ma = 0
                if mi > 0 and 0 < ma < 9000: return f"{mi:.0f}~{ma:.0f}"
                elif mi > 0: return f"≥ {mi:.0f}"
                elif 0 < ma < 9000: return f"≤ {ma:.0f}"
                return "-"

            # 建立完整數據行 (Create full data row)
            master_dict = {col: (keys[idx] if isinstance(keys, tuple) else keys) for idx, col in enumerate(GROUP_COLS)}
            master_dict.update({
                "N Coils": len(group),
                "Current Hardness Spec": f"{group['Limit_Min'].max():.1f}~{group['Limit_Max'].min():.1f}",
                "Proposed Control Limit (2σ)": f"{c_min:.1f} ~ {c_max:.1f}",
                "🎯 Proposed Target Zone (1σ)": f"{t_min:.1f} ~ {t_max:.1f}",
                
                "Spec: TS": fmt_s(s_ts_min, s_ts_max),
                "Exp. TS (at Target)": f"{int(m_ts.predict([[t_min]])[0])}~{int(m_ts.predict([[t_max]])[0])}",
                
                "Spec: YS": fmt_s(s_ys_min, s_ys_max),
                "Exp. YS (at Target)": f"{int(m_ys.predict([[t_min]])[0])}~{int(m_ys.predict([[t_max]])[0])}",
                
                "Spec: EL": f"≥ {s_el_min:.1f}%" if s_el_min > 0 else "-",
                "Exp. EL (at Target)": f"{min(m_el.predict([[t_min]])[0], m_el.predict([[t_max]])[0]):.1f}% ~ {max(m_el.predict([[t_min]])[0], m_el.predict([[t_max]])[0]):.1f}%"
            })
            master_data.append(master_dict)
        
        if master_data:
            df_out = pd.DataFrame(master_data)
            
            # 強制指定欄位排序 (Force column ordering for logical flow)
            ordered_cols = GROUP_COLS + [
                "N Coils", 
                "Current Hardness Spec", "Proposed Control Limit (2σ)", "🎯 Proposed Target Zone (1σ)",
                "Spec: TS", "Exp. TS (at Target)",
                "Spec: YS", "Exp. YS (at Target)",
                "Spec: EL", "Exp. EL (at Target)"
            ]
            final_cols = [c for c in ordered_cols if c in df_out.columns]
            df_out = df_out[final_cols]
            
            # --- 1. 在畫面上直接顯示美化後的預覽表格 (Show styled dataframe on screen) ---
            st.markdown("### 👁️ Preview Master Dictionary")
            
            # 套用與 Excel 相同的顏色邏輯 (Apply the same color logic as Excel to the Streamlit dataframe)
            styled_df = df_out.style.set_properties(**{'background-color': '#FFF2CC', 'color': '#856404'}, subset=[c for c in final_cols if "Spec:" in c or "Current Hardness Spec" in c]) \
                                    .set_properties(**{'background-color': '#D9EAD3', 'color': '#155724', 'font-weight': 'bold'}, subset=[c for c in final_cols if "Target" in c or "Exp." in c]) \
                                    .set_properties(**{'background-color': '#CFE2F3', 'color': '#004085'}, subset=["Proposed Control Limit (2σ)"])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # --- 2. 準備 Excel 匯出檔案 (Prepare Excel Export file) ---
            import datetime as dt
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_out.to_excel(writer, sheet_name='Master_Specs', index=False)
                workbook = writer.book
                worksheet = writer.sheets['Master_Specs']
                
                # Excel 美化格式設定 (Excel beautification settings)
                header_fmt = workbook.add_format({'bold': True, 'bg_color': '#CFE2F3', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
                target_fmt = workbook.add_format({'bg_color': '#D9EAD3', 'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
                spec_fmt = workbook.add_format({'bg_color': '#FFF2CC', 'italic': True, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
                
                worksheet.set_row(0, 30) 
                
                for col_num, value in enumerate(df_out.columns.values):
                    fmt = header_fmt
                    if "Target" in value or "Exp." in value: fmt = target_fmt
                    if "Spec:" in value or "Current Hardness Spec" in value: fmt = spec_fmt
                    
                    worksheet.write(0, col_num, value, fmt)
                    worksheet.set_column(col_num, col_num, max(12, len(value) * 0.8))
            
            st.markdown("### 📥 Download Report")
            st.success(f"✅ Full Master Dictionary created for {len(master_data)} groups!")
            st.download_button("📥 Download Full Dictionary (Excel)", output.getvalue(), f"Full_Master_Dictionary_{dt.datetime.now().strftime('%Y%m%d')}.xlsx")
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
        # 移除 Rolling_Type 欄位 (Remove Rolling_Type column for cleaner view)
        display_df = sub.drop(columns=["Rolling_Type"]) if "Rolling_Type" in sub.columns else sub.copy()
        
        # 找出所有日期欄位並格式化，去除後方的 00:00:00 (Format datetime to remove 00:00:00)
        date_cols = display_df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        for d_col in date_cols:
            display_df[d_col] = display_df[d_col].dt.strftime('%Y-%m-%d')
            
        # 備用方案：如果欄位名稱包含 DATE 但被識別為字串，則強制轉換 (Fallback: force format string columns containing 'DATE')
        for col in display_df.columns:
            if 'DATE' in str(col).upper() and col not in date_cols:
                try:
                    display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d')
                except:
                    pass
        
        # 根據 NG 標籤標記異常行 (Highlight NG rows in red)
        def highlight_ng_rows(row): 
            return ['background-color: #ffe6e6'] * len(row) if row.get('NG', False) else [''] * len(row)
        
        num_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 設定數值格式：硬度保留 1 位小數，其他數值（包含 Limit_Min/Max 等）皆去除小數點 
        # (Format numeric columns: Hardness keeps 1 decimal, Limit_Min/Max and others keep 0 decimals)
        fmt = {}
        for c in num_cols:
            if c in ["Hardness_LINE", "Hardness_LAB"]:
                fmt[c] = "{:.1f}"
            else:
                fmt[c] = "{:.0f}"
                
        st.dataframe(display_df.style.format(fmt).apply(highlight_ng_rows, axis=1), use_container_width=True)

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
            
            if len(line) < 5: st.warning("⚠️ At least 5 coils are required for distribution analysis.")
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
            
            # --- 升級：建立副 Y 軸來獨立顯示 EL，避免擠在圖表底部 (Upgrade: Create secondary Y-axis for EL) ---
            ax2 = ax.twinx()
            
            def p_prop(ax_obj, x, y, ymin, ymax, c, lbl, m):
                ax_obj.plot(x, y, marker=m, color=c, label=lbl, lw=2)
                ax_obj.fill_between(x, ymin, ymax, color=c, alpha=0.1)
            
            # 繪製 TS 和 YS 在主軸 (Plot TS and YS on primary axis)
            p_prop(ax, x, summary["TS_mean"], summary["TS_min"], summary["TS_max"], "#1f77b4", "TS Actual", "o")
            p_prop(ax, x, summary["YS_mean"], summary["YS_min"], summary["YS_max"], "#2ca02c", "YS Actual", "s")
            
            # 繪製 EL 在副軸 (Plot EL on secondary axis)
            p_prop(ax2, x, summary["EL_mean"], summary["EL_min"], summary["EL_max"], "#ff7f0e", "EL Actual", "^")

            # --- 獲取規格界限 (Get Spec Limits) ---
            g_ts_min = summary["Std_TS_min"].max()
            g_ts_max = summary["Std_TS_max"].min()
            g_ys_min = summary["Std_YS_min"].max()
            g_ys_max = summary["Std_YS_max"].min()
            g_el_min = summary["Std_EL_min"].max()

            # --- 繪製規格界限線 (Draw Spec Limit Lines) ---
            if pd.notna(g_ts_min) and g_ts_min > 0: ax.axhline(g_ts_min, color="#1f77b4", linestyle="--", lw=1.5, alpha=0.5, label=f"TS LSL ({g_ts_min:.0f})")
            if pd.notna(g_ts_max) and 0 < g_ts_max < 9000: ax.axhline(g_ts_max, color="#1f77b4", linestyle="--", lw=1.5, alpha=0.5, label=f"TS USL ({g_ts_max:.0f})")
            
            if pd.notna(g_ys_min) and g_ys_min > 0: ax.axhline(g_ys_min, color="#2ca02c", linestyle="-.", lw=1.5, alpha=0.5, label=f"YS LSL ({g_ys_min:.0f})")
            if pd.notna(g_ys_max) and 0 < g_ys_max < 9000: ax.axhline(g_ys_max, color="#2ca02c", linestyle="-.", lw=1.5, alpha=0.5, label=f"YS USL ({g_ys_max:.0f})")
            
            if pd.notna(g_el_min) and g_el_min > 0: ax2.axhline(g_el_min, color="#ff7f0e", linestyle=":", lw=2, alpha=0.6, label=f"EL LSL ({g_el_min:.0f})")

            for j, row in enumerate(summary.itertuples()):
                ts_min, ts_max = row.Std_TS_min, row.Std_TS_max
                ys_min, ys_max = row.Std_YS_min, row.Std_YS_max
                el_spec = row.Std_EL_min if pd.notna(row.Std_EL_min) else 0
                
                ts_fail = (pd.notna(ts_min) and ts_min > 0 and row.TS_mean < ts_min) or (pd.notna(ts_max) and 0 < ts_max < 9000 and row.TS_mean > ts_max)
                ys_fail = (pd.notna(ys_min) and ys_min > 0 and row.YS_mean < ys_min) or (pd.notna(ys_max) and 0 < ys_max < 9000 and row.YS_mean > ys_max)
                el_fail = (el_spec > 0) and (row.EL_mean < el_spec)
                
                ax.annotate(f"{row.TS_mean:.0f}" + (" ❌" if ts_fail else ""), (x[j], row.TS_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="red" if ts_fail else "#1f77b4")
                ax.annotate(f"{row.YS_mean:.0f}" + (" ❌" if ys_fail else ""), (x[j], row.YS_mean), xytext=(0,-15), textcoords="offset points", ha="center", fontsize=9, fontweight='bold', color="red" if ys_fail else "#2ca02c")
                ax2.annotate(f"{row.EL_mean:.1f}%" + (" ❌" if el_fail else ""), (x[j], row.EL_mean), xytext=(0,10), textcoords="offset points", ha="center", fontsize=9, color="red" if el_fail else "#ff7f0e", fontweight=("bold" if el_fail else "normal"))

            ax.set_xticks(x); ax.set_xticklabels(summary["HRB_bin"])
            ax.set_ylabel("Strength (MPa)", fontweight="bold")
            ax2.set_ylabel("Elongation (%)", fontweight="bold", color="#ff7f0e")
            ax.set_title("Hardness vs Mechanical Properties with Spec Limits", fontweight="bold")
            
            # 合併圖例 (Combine legends to the right side)
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left", bbox_to_anchor=(1.08, 0.5))
            
            ax.grid(True, ls="--", alpha=0.5)
            fig.tight_layout() # 防止圖例被裁切 (Prevent legend from being cut off)
            st.pyplot(fig)

            specs_str = f"Specs: {', '.join(str(x) for x in sub['Product_Spec'].dropna().unique())}" if 'Product_Spec' in sub.columns else "Specs: N/A"

            # 於總結表中直接進行評估演算法 (Evaluation algorithm within summary table)
            def check_limit(act_min, act_max, sp_min, sp_max, is_el=False):
                fail = False
                if pd.notna(sp_min) and sp_min > 0 and act_min < sp_min: fail = True
                if not is_el and pd.notna(sp_max) and 0 < sp_max < 9000 and act_max > sp_max: fail = True
                res = f"{act_min:.1f}~{act_max:.1f}" if is_el else f"{act_min:.0f}~{act_max:.0f}"
                return res + (" ❌" if fail else " ✅")

            for row in summary.itertuples():
                bin_data = sub_corr[sub_corr["HRB_bin"] == row.HRB_bin]
                
                ts_act_str = check_limit(row.TS_min, row.TS_max, row.Std_TS_min, row.Std_TS_max)
                ys_act_str = check_limit(row.YS_min, row.YS_max, row.Std_YS_min, row.Std_YS_max)
                el_act_str = check_limit(row.EL_min, row.EL_max, row.Std_EL_min, 0, is_el=True)

                ts_std = bin_data['TS'].std()
                ys_std = bin_data['YS'].std()
                el_std = bin_data['EL'].std()

                corr_bin_summary.append({
                    "Specification List": specs_str, "Material": g.get("Material", "N/A"), "Gauge": g.get("Order_Gauge", "N/A"),
                    "Hardness Bin": row.HRB_bin, "N": row.N_coils,
                    "TS Spec": f"{row.Std_TS_min:.0f}~{row.Std_TS_max:.0f}" if pd.notna(row.Std_TS_max) and row.Std_TS_max < 9000 else (f"≥{row.Std_TS_min:.0f}" if pd.notna(row.Std_TS_min) else "-"),
                    "TS Actual": ts_act_str, "TS Mean": f"{row.TS_mean:.1f}", "TS Std": f"{ts_std:.1f}" if pd.notna(ts_std) else "-",
                    "YS Spec": f"{row.Std_YS_min:.0f}~{row.Std_YS_max:.0f}" if pd.notna(row.Std_YS_max) and row.Std_YS_max < 9000 else (f"≥{row.Std_YS_min:.0f}" if pd.notna(row.Std_YS_min) else "-"),
                    "YS Actual": ys_act_str, "YS Mean": f"{row.YS_mean:.1f}", "YS Std": f"{ys_std:.1f}" if pd.notna(ys_std) else "-",
                    "EL Spec": f"≥{row.Std_EL_min:.0f}" if pd.notna(row.Std_EL_min) else "-",
                    "EL Actual": el_act_str, "EL Mean": f"{row.EL_mean:.1f}", "EL Std": f"{el_std:.1f}" if pd.notna(el_std) else "-"
                })

        if i == len(valid) - 1 and 'corr_bin_summary' in locals() and len(corr_bin_summary) > 0:
            st.markdown("---")
            st.markdown(f"## 📊 Hardness Binning Comprehensive Report")
            df_full = pd.DataFrame(corr_bin_summary)
            
            def d_bin(title, cols, c_code):
                st.markdown(f"#### {title}")
                target_df = df_full[["Specification List", "Material", "Gauge", "Hardness Bin", "N"] + cols]
                
                # 根據圖示為儲存格填色 (Color cells based on icons)
                def hl_status(val):
                    if isinstance(val, str):
                        if '❌' in val: return 'color: #721c24; font-weight: bold; background-color: #f8d7da'
                        if '✅' in val: return 'color: #155724; font-weight: bold'
                    return ''
                
                styled = target_df.style.set_properties(**{'background-color': c_code, 'font-weight': 'bold'}, subset=[c for c in cols if "Std" in c])
                if hasattr(styled, "map"):
                    styled = styled.map(hl_status, subset=[c for c in cols if "Actual" in c])
                else:
                    styled = styled.applymap(hl_status, subset=[c for c in cols if "Actual" in c])
                    
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
                
                axes[j].hist(data, bins=15, color=cfg["color"], alpha=0.5, density=True, label="Actual Data")
                if std > 0:
                    x_p = np.linspace(data.min() - 3*std, data.max() + 3*std, 200)
                    axes[j].plot(x_p, (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_p-mean)/std)**2), color=cfg["color"], lw=2, label="Normal Dist")
                
                if spec_min > 0: axes[j].axvline(spec_min, color="red", linestyle="--", linewidth=2, label=f"Spec Min ({spec_min:.0f})")
                if spec_max > 0 and spec_max < 9000: axes[j].axvline(spec_max, color="red", linestyle="--", linewidth=2, label=f"Spec Max ({spec_max:.0f})")
                axes[j].axvline(lcl_3s, color="blue", linestyle=":", linewidth=2, label=f"-3σ LCL ({lcl_3s:.1f})")
                axes[j].axvline(ucl_3s, color="blue", linestyle=":", linewidth=2, label=f"+3σ UCL ({ucl_3s:.1f})")
                
                axes[j].set_title(f"{cfg['name']}\n(Mean={mean:.1f}, Std={std:.1f})", fontweight="bold")
                axes[j].legend(fontsize=9, loc="upper right")
                
                cpk = None
                if std > 0 and (spec_min > 0 or (spec_max > 0 and spec_max < 9000)):
                    if spec_min > 0 and spec_max > 0 and spec_max < 9000:
                        cpk = min((spec_max - mean) / (3 * std), (mean - spec_min) / (3 * std))
                    elif spec_min > 0:
                        cpk = (mean - spec_min) / (3 * std)
                    elif spec_max > 0 and spec_max < 9000:
                        cpk = (spec_max - mean) / (3 * std)

                row_data = {
                    "Group": group_title, "N": len(data), "Hardness Range (HRB)": hrb_rng,
                    "Limit (Spec)": f"{spec_min:.0f}~{spec_max:.0f}" if (spec_max > 0 and spec_max < 9000) else (f"≥ {spec_min:.0f}" if spec_min > 0 else "N/A"),
                    "Actual Range": f"{data.min():.1f}~{data.max():.1f}",
                    "Mean": f"{mean:.1f}", "Std Dev": f"{std:.1f}", 
                    "LCL (-3σ)": f"{lcl_3s:.1f}", "UCL (+3σ)": f"{ucl_3s:.1f}",
                    "Cpk": f"{cpk:.2f}" if cpk is not None else "-"
                }
                if col == "TS": ts_summary.append(row_data)
                elif col == "YS": ys_summary.append(row_data)
                elif col == "EL": el_summary.append(row_data)
            else: axes[j].set_title(f"{cfg['name']}\n(No Data)")
            axes[j].grid(alpha=0.3, linestyle="--")

        if has_data: st.pyplot(fig)
        else: st.warning("⚠️ Insufficient Mechanical Properties data for the selected group.")

        if i == len(valid) - 1:
            st.markdown("---")
            st.markdown("## 📊 Mechanical Properties Comprehensive Report")
            
            def d_sum(title, data_list, c_code):
                if data_list:
                    st.markdown(f"#### {title}")
                    
                    def cpk_color(val):
                        try:
                            v = float(val)
                            if v >= 1.33: return 'color: green; font-weight: bold'
                            elif v >= 1.0: return 'color: orange; font-weight: bold'
                            else: return 'color: red; font-weight: bold'
                        except: return ''
                        
                    styled_df = pd.DataFrame(data_list).style.set_properties(**{'font-weight': 'bold'}, subset=['Mean']) \
                                        .set_properties(**{'background-color': '#f0f8ff', 'font-weight': 'bold', 'color': '#0056b3'}, subset=['Hardness Range (HRB)']) \
                                        .set_properties(**{'background-color': c_code, 'color': '#004085'}, subset=['LCL (-3σ)', 'UCL (+3σ)']) \
                                        .applymap(cpk_color, subset=['Cpk'])
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
        # 獲取當前組別的機械性能標準規範 (Get current mechanical standard specs for the group)
        spec_ts_min = sub["Standard TS min"].max() if "Standard TS min" in sub.columns else 0
        spec_ts_max = sub["Standard TS max"].min() if "Standard TS max" in sub.columns else 0
        spec_ys_min = sub["Standard YS min"].max() if "Standard YS min" in sub.columns else 0
        spec_ys_max = sub["Standard YS max"].min() if "Standard YS max" in sub.columns else 0
        spec_el_min = sub["Standard EL min"].max() if "Standard EL min" in sub.columns else 0
        
        def fmt_spec(s_min, s_max):
            if pd.isna(s_min): s_min = 0
            if pd.isna(s_max): s_max = 0
            if s_min > 0 and 0 < s_max < 9000: return f"{s_min:.0f} ~ {s_max:.0f}"
            elif s_min > 0: return f"≥ {s_min:.0f}"
            elif 0 < s_max < 9000: return f"≤ {s_max:.0f}"
            return "N/A"

        ts_spec_str = fmt_spec(spec_ts_min, spec_ts_max)
        ys_spec_str = fmt_spec(spec_ys_min, spec_ys_max)
        el_spec_str = fmt_spec(spec_el_min, 0)
        
        # 顯示當前規格基準供主管對照 (Display current spec baseline for manager's reference)
        st.info(f"📋 **Current Standard Specs:** &nbsp;&nbsp;&nbsp; TS: **{ts_spec_str}** &nbsp;&nbsp;|&nbsp;&nbsp; YS: **{ys_spec_str}** &nbsp;&nbsp;|&nbsp;&nbsp; EL: **{el_spec_str}**")
        
        c1, c2, c3 = st.columns(3)
        # 設定輸入框，預設值帶入實際數據的極值 (Set input boxes, default to actual data extremes)
        r_ts_min = c1.number_input("Min TS", value=float(sub['TS'].min()) if not sub['TS'].isna().all() else 0.0, step=5.0, key=f"tmin_{i}")
        r_ts_max = c1.number_input("Max TS", value=float(sub['TS'].max()) if not sub['TS'].isna().all() else 1000.0, step=5.0, key=f"tmax_{i}")
        
        r_ys_min = c2.number_input("Min YS", value=float(sub['YS'].min()) if not sub['YS'].isna().all() else 0.0, step=5.0, key=f"ymin_{i}")
        r_ys_max = c2.number_input("Max YS", value=float(sub['YS'].max()) if not sub['YS'].isna().all() else 1000.0, step=5.0, key=f"ymax_{i}")
        
        r_el_min = c3.number_input("Min EL", value=float(sub['EL'].min()) if not sub['EL'].isna().all() else 0.0, step=1.0, key=f"emin_{i}")
        r_el_max = c3.number_input("Max EL", value=float(sub['EL'].max()) if not sub['EL'].isna().all() else 100.0, step=1.0, key=f"emax_{i}")

        filtered = sub[(sub['YS'] >= r_ys_min) & (sub['YS'] <= r_ys_max) & (sub['TS'] >= r_ts_min) & (sub['TS'] <= r_ts_max) & (sub['EL'] >= r_el_min) & (sub['EL'] <= r_el_max)]
        
        if not filtered.empty:
            st.success(f"✅ Found **{len(filtered)}** coils. Optimal Hardness Range: **{filtered['Hardness_LINE'].min():.1f} ~ {filtered['Hardness_LINE'].max():.1f} HRB**")
            st.dataframe(filtered[['COIL_NO','Hardness_LINE','TS','YS','EL']].style.format("{:.1f}", subset=['Hardness_LINE', 'TS', 'YS', 'EL']), use_container_width=True, hide_index=True)
        else: 
            st.error("❌ No coils found matching these target parameters.")

    elif view_mode == "🧮 Predict TS/YS/EL from Std Hardness":
        st.markdown(f"#### 🧮 AI Prediction Engine: {group_title}")
        train_df = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])
        
        if len(train_df) < 3:
            st.warning("⚠️ At least 3 coils are required to activate AI Prediction.")
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
            
            # 整合規格檢查掃描 (Integrate Spec check scan)
            st.markdown("##### 🏁 Forecast Summary & Spec Evaluation")
            c1, c2, c3 = st.columns(3)
            def get_delta(p, l): return round(p - l, 1)
            last_ts = train_df["TS"].iloc[-1]; last_ys = train_df["YS"].iloc[-1]; last_el = train_df["EL"].iloc[-1]

            ts_m_min = sub["Standard TS min"].max() if "Standard TS min" in sub.columns else 0
            ts_m_max = sub["Standard TS max"].min() if "Standard TS max" in sub.columns else 0
            ys_m_min = sub["Standard YS min"].max() if "Standard YS min" in sub.columns else 0
            ys_m_max = sub["Standard YS max"].min() if "Standard YS max" in sub.columns else 0
            el_m_min = sub["Standard EL min"].max() if "Standard EL min" in sub.columns else 0
            
            def check_sp(val, s_min, s_max, is_el=False):
                s_min = s_min if pd.notna(s_min) else 0
                s_max = s_max if pd.notna(s_max) else 0
                lim_str = f"{s_min:.0f}~{s_max:.0f}" if (0 < s_max < 9000) else (f"≥ {s_min:.0f}" if s_min > 0 else "-")
                if is_el: lim_str = f"≥ {s_min:.1f}" if s_min > 0 else "-"
                
                if s_min > 0 and val < s_min: return "❌ FAIL", lim_str
                if not is_el and 0 < s_max < 9000 and val > s_max: return "❌ FAIL", lim_str
                return "✅ PASS", lim_str

            ts_stat, ts_spec = check_sp(preds['TS'], ts_m_min, ts_m_max)
            ys_stat, ys_spec = check_sp(preds['YS'], ys_m_min, ys_m_max)
            el_stat, el_spec = check_sp(preds['EL'], el_m_min, 0, is_el=True)

            c1.metric(f"Tensile (TS) - {ts_stat}", f"{int(round(preds['TS']))} MPa", f"{get_delta(preds['TS'], last_ts)} vs Last")
            c1.caption(f"**Spec:** {ts_spec} | **R²:** {model_metrics['TS']['r2']:.2f}")

            c2.metric(f"Yield (YS) - {ys_stat}", f"{int(round(preds['YS']))} MPa", f"{get_delta(preds['YS'], last_ys)} vs Last")
            c2.caption(f"**Spec:** {ys_spec} | **R²:** {model_metrics['YS']['r2']:.2f}")

            c3.metric(f"Elongation (EL) - {el_stat}", f"{round(preds['EL'], 1)} %", f"{get_delta(preds['EL'], last_el)} vs Last")
            c3.caption(f"**Spec:** {el_spec} | **R²:** {model_metrics['EL']['r2']:.2f}")

    elif view_mode == "🎛️ Control Limit Calculator (Compare 3 Methods)":
        
        # --- 1. 在視圖頂部顯示一次說明 (Display explanation once at the top) ---
        if i == 0:
            all_groups_summary = []
            st.markdown("### 📘 Control Limit Calculation Methods")
            with st.expander("🔍 Click to view method details", expanded=True):
                st.markdown("""
                | Method | Name | Description |
                | :--- | :--- | :--- |
                | **M1: Standard** | **Standard Statistics** | Calculated based on all data. Limits can be over-stretched if extreme outliers exist. |
                | **M2: IQR Robust** | **Interquartile Range** | Automatically filters out extreme values, making limits more aligned with actual distribution. |
                | **M3: Smart Hybrid** | **Smart Hybrid** | Combines statistical trends and customer specifications to ensure limits stay in safe zones. |
                | **M4: I-MR (SPC)** | **Process Control** | **Optimal approach:** Monitors variation between adjacent coils; highly scientific for process stability. |
                """)

        st.markdown(f"### 🎛️ Control Limits Analysis: {group_title}")
        data = sub["Hardness_LINE"].dropna()
        data_lab = sub["Hardness_LAB"].dropna() if "Hardness_LAB" in sub.columns else pd.Series(dtype=float)
        
        if len(data) < 5: 
            st.warning(f"⚠️ Not enough data for analysis (N={len(data)})")
        else:
            with st.expander("⚙️ Settings", expanded=False):
                c1, c2 = st.columns(2)
                sigma_n = c1.number_input("1. Sigma Multiplier (K)", 1.0, 6.0, 2.0, 0.5, key=f"sig_{i}")
                iqr_k = c2.number_input("2. IQR Sensitivity", 0.1, 3.0, 0.5, 0.1, key=f"iqr_{i}")

            spec_min = lo
            spec_max = hi
            display_max = spec_max if (spec_max > 0 and spec_max < 9000) else 0
            
            mu = data.mean()
            std_dev = data.std()
            
            # 算法 M1 (Algorithm M1)
            m1_min, m1_max = mu - sigma_n*std_dev, mu + sigma_n*std_dev
            
            # 算法 M2 (Algorithm M2)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            clean_data = data[~((data < (Q1 - iqr_k * IQR)) | (data > (Q3 + iqr_k * IQR)))]
            if clean_data.empty: clean_data = data
            mu_clean, sigma_clean = clean_data.mean(), clean_data.std()
            if pd.isna(sigma_clean) or sigma_clean == 0: sigma_clean = std_dev
            m2_min, m2_max = mu_clean - sigma_n*sigma_clean, mu_clean + sigma_n*sigma_clean
            
            # 算法 M3 (Algorithm M3)
            m3_min = max(m2_min, spec_min)
            m3_max = min(m2_max, spec_max) if (spec_max > 0 and spec_max < 9000) else m2_max
            if m3_min >= m3_max: m3_min, m3_max = m2_min, m2_max
            
            # 算法 M4 (Algorithm M4)
            mrs = np.abs(np.diff(data))
            mr_bar = np.mean(mrs) if len(mrs) > 0 else 0
            sigma_imr = mr_bar / 1.128 if mr_bar > 0 else std_dev
            m4_min, m4_max = mu - sigma_n * sigma_imr, mu + sigma_n * sigma_imr

            # --- 計算目標界限 (Calculate Target Limits) ---
            target_k = 1.0 
            new_target_min = mu - target_k * sigma_imr
            new_target_max = mu + target_k * sigma_imr

            spec_str = f"Ctrl: {spec_min:.0f}~{display_max:.0f}"

            all_groups_summary.append({
                "Group": group_title,
                "N": len(data),
                "Current Spec": spec_str,
                "M1: Standard": f"{m1_min:.1f} ~ {m1_max:.1f}",
                "M2: IQR (Robust)": f"{m2_min:.1f} ~ {m2_max:.1f}",
                "M3: Smart Hybrid": f"{m3_min:.1f} ~ {m3_max:.1f}", 
                "M4: I-MR (Optimal)": f"{m4_min:.1f} ~ {m4_max:.1f}",
                "New Core Target (±1.0σ)": f"{new_target_min:.1f} ~ {new_target_max:.1f}",
                "Status": "✅ Stable" if (display_max > 0 and m4_max <= display_max) else "⚠️ Narrow Spec"
            })
            
            # ==================================================================
            # 綜合界限圖表與常態分佈曲線 (Combined Limits Chart with Normal Curve)
            # ==================================================================
            from scipy.stats import norm
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # 繪製實際資料分佈直方圖 (Plot actual data histograms)
            ax.hist(data, bins=15, density=True, alpha=0.6, color="#1f77b4", label="LINE (Production)")
            if not data_lab.empty: ax.hist(data_lab, bins=15, density=True, alpha=0.4, color="#ff7f0e", label="LAB (Ref)")
            
            # 加入常態分佈曲線 (Add normal curve)
            min_cands = [m1_min, m4_min, spec_min, data.min()]
            max_cands = [m1_max, m4_max, display_max, data.max()]
            if not data_lab.empty:
                min_cands.append(data_lab.min())
                max_cands.append(data_lab.max())
                
            x_min_val = min(min_cands) - 5
            x_max_val = max(max_cands) + 5
            x_axis = np.linspace(x_min_val, x_max_val, 500)
            
            ax.plot(x_axis, norm.pdf(x_axis, mu, std_dev), color="#333333", lw=2, alpha=0.8, label=f"Normal Curve (σ={std_dev:.2f})")
            
            # 繪製各方法控制界限 (Plot control limits for each method)
            ax.axvline(m1_min, c="red", ls=":", alpha=0.4, label="M1: Standard")
            ax.axvline(m1_max, c="red", ls=":", alpha=0.4)
            ax.axvline(m2_min, c="blue", ls="--", alpha=0.5, label="M2: IQR")
            ax.axvline(m2_max, c="blue", ls="--", alpha=0.5)
            ax.axvline(m4_min, c="purple", ls="-.", lw=2, label="M4: I-MR (SPC)")
            ax.axvline(m4_max, c="purple", ls="-.", lw=2)
            ax.axvspan(m3_min, m3_max, color="green", alpha=0.15, label="M3: Hybrid Zone")
            
            if spec_min > 0: ax.axvline(spec_min, c="black", lw=2)
            if display_max > 0: ax.axvline(display_max, c="black", lw=2)
            
            ax.set_title(f"Limits Comparison with Normal Distribution (σ={sigma_n})", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize="small")
            st.pyplot(fig)

            # ==================================================================
            # 預估機械性能表與匯出 (Mechanical Estimation Table & Export)
            # ==================================================================
            st.write("---") 
            st.markdown(f"#### 📌 Limit Summary & Mechanical Estimation")
            
            # 從數據中獲取機械性能規格界限 (Get Mech Spec limits from data)
            spec_ts_min = sub["Standard TS min"].max() if "Standard TS min" in sub.columns else 0
            spec_ts_max = sub["Standard TS max"].min() if "Standard TS max" in sub.columns else 0
            spec_ys_min = sub["Standard YS min"].max() if "Standard YS min" in sub.columns else 0
            spec_ys_max = sub["Standard YS max"].min() if "Standard YS max" in sub.columns else 0
            spec_el_min = sub["Standard EL min"].max() if "Standard EL min" in sub.columns else 0
            
            def fmt_spec(s_min, s_max):
                if pd.isna(s_min): s_min = 0
                if pd.isna(s_max): s_max = 0
                if s_min > 0 and 0 < s_max < 9000: return f"{s_min:.0f}~{s_max:.0f}"
                elif s_min > 0: return f"≥ {s_min:.0f}"
                elif 0 < s_max < 9000: return f"≤ {s_max:.0f}"
                return "-"

            # 顯示規格基準 (Display Spec baseline)
            st.info(f"**Mechanical Specs Target:** TS: **{fmt_spec(spec_ts_min, spec_ts_max)}** | YS: **{fmt_spec(spec_ys_min, spec_ys_max)}** | EL: **{fmt_spec(spec_el_min, 0)}**")

            # 使用實際數據訓練線性迴歸模型進行預測 (Train Linear Regression model from actual data for prediction)
            df_train = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])
            has_model = False
            if len(df_train) >= 3:
                has_model = True
                X_train = df_train[["Hardness_LINE"]].values
                m_ts = LinearRegression().fit(X_train, df_train["TS"].values)
                m_ys = LinearRegression().fit(X_train, df_train["YS"].values)
                m_el = LinearRegression().fit(X_train, df_train["EL"].values)

            def get_mech(h_val):
                if not has_model or pd.isna(h_val) or h_val <= 0: return 0, 0, 0
                ts = m_ts.predict([[h_val]])[0]
                ys = m_ys.predict([[h_val]])[0]
                el = m_el.predict([[h_val]])[0]
                return ts, ys, el
                
            def eval_spec(v_min, v_max, s_min, s_max, is_el=False):
                if v_min == 0 and v_max == 0: return "N/A"
                # EL 只有下限 (min)。v_min 是對應最高硬度的最低 EL 點 (EL only has a lower bound)
                if is_el: 
                    if pd.notna(s_min) and s_min > 0 and v_min < s_min: return "❌ Fail"
                    return "✅ Pass"
                
                # TS 與 YS (TS and YS)
                if pd.notna(s_min) and s_min > 0 and v_min < s_min: return "❌ Fail"
                if pd.notna(s_max) and 0 < s_max < 9000 and v_max > s_max: return "❌ Fail"
                return "✅ Pass"

            rows = []
            configs = [
                ("🎯 Old Target Goal", spec_min, display_max, "-"),
                ("🔴 M1: Standard (Historical)", m1_min, m1_max, std_dev),
                ("🔵 M2: IQR (Robust)", m2_min, m2_max, sigma_clean),
                ("🟢 M3: Smart Hybrid", m3_min, m3_max, "-"),
                ("🟣 M4: I-MR (Control Limits)", m4_min, m4_max, sigma_imr),
                (f"🌟 New Core Target (±{target_k}σ)", new_target_min, new_target_max, "-")
            ]

            for cat, l_min, l_max, sig in configs:
                ts_1, ys_1, el_1 = get_mech(l_min)
                ts_2, ys_2, el_2 = get_mech(l_max)
                
                ts_lmin, ts_lmax = min(ts_1, ts_2), max(ts_1, ts_2)
                ys_lmin, ys_lmax = min(ys_1, ys_2), max(ys_1, ys_2)
                el_lmin, el_lmax = min(el_1, el_2), max(el_1, el_2)
                
                ts_eval = eval_spec(ts_lmin, ts_lmax, spec_ts_min, spec_ts_max)
                ys_eval = eval_spec(ys_lmin, ys_lmax, spec_ys_min, spec_ys_max)
                el_eval = eval_spec(el_lmin, el_lmax, spec_el_min, 0, is_el=True)
                
                overall = "✅ Optimal" if (ts_eval == "✅ Pass" and ys_eval == "✅ Pass" and el_eval == "✅ Pass") else "⚠️ Warning"
                if not has_model: overall = "N/A"

                rows.append({
                    "Limit Type": cat,
                    "Hardness Limits": f"{l_min:.1f} ~ {l_max:.1f}",
                    "Variation": f"σ={sig:.2f}" if isinstance(sig, float) else sig,
                    "Est. TS": f"{ts_lmin:.0f} ~ {ts_lmax:.0f}" if has_model else "-",
                    "TS Eval": ts_eval,
                    "Est. YS": f"{ys_lmin:.0f} ~ {ys_lmax:.0f}" if has_model else "-",
                    "YS Eval": ys_eval,
                    "Est. EL (%)": f"{el_lmin:.1f} ~ {el_lmax:.1f}" if has_model else "-",
                    "EL Eval": el_eval,
                    "Overall Proposal": overall
                })

            df_summary = pd.DataFrame(rows)
            
            def highlight_status(val):
                if isinstance(val, str):
                    if "✅" in val: return 'color: #155724; font-weight: bold'
                    if "❌" in val: return 'color: #721c24; font-weight: bold; background-color: #f8d7da'
                    if "⚠️" in val: return 'color: #856404; font-weight: bold'
                return ''

            def highlight_new_target(s):
                if "🌟 New Core Target" in str(s['Limit Type']): return ['background-color: #e2efda'] * len(s)
                return [''] * len(s)

            styled_df = df_summary.style.apply(highlight_new_target, axis=1) \
                                        .applymap(highlight_status, subset=['TS Eval', 'YS Eval', 'EL Eval', 'Overall Proposal'])

            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            if not has_model: st.caption("*(⚠️ Not enough actual data (N<3) for AI to estimate mechanical properties.)*")
            else: st.caption("*(**) Estimated values are generated by AI Linear Regression using actual group data. A ✅ Pass status indicates the estimated variation remains within the Mechanical Specifications.*")

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
                label="📥 Download Estimation Summary (Excel)",
                data=buffer.getvalue(),
                file_name=f"Mech_Estimation_{safe_group_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_sum_{i}"
            )
            
        # --- 顯示所有分組的整體總結表 (Display overall summary table for all groups) ---
        if i == len(valid) - 1 and 'all_groups_summary' in locals() and len(all_groups_summary) > 0:
            st.markdown("---")
            st.markdown("## 📊 Summary of Control Limits")
            df_total = pd.DataFrame(all_groups_summary)
            
            # 設定樣式：凸顯 M4 和 New Target (Style: Highlight M4 and New Target)
            styled_df = df_total.style.applymap(lambda v: 'color: red; font-weight: bold' if 'Narrow' in v else 'color: green; font-weight: bold', subset=['Status']) \
                                      .set_properties(**{'background-color': '#e6f2ff', 'color': '#004085', 'font-weight': 'bold'}, subset=['M4: I-MR (Optimal)']) \
                                      .set_properties(**{'background-color': '#e2efda', 'color': '#155724', 'font-weight': 'bold'}, subset=['New Core Target (±1.0σ)'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # 轉換為 Excel 檔案並自動調整欄寬 (Convert to Excel file and auto-adjust column width)
            import io
            buffer_spc = io.BytesIO()
            with pd.ExcelWriter(buffer_spc, engine='xlsxwriter') as writer:
                df_total.to_excel(writer, sheet_name='SPC_Summary', index=False)
                worksheet = writer.sheets['SPC_Summary']
                for idx, col_name in enumerate(df_total.columns):
                    max_len = max(df_total[col_name].astype(str).map(len).max(), len(col_name)) + 2
                    worksheet.set_column(idx, idx, max_len)

            st.download_button(
                label="📥 Export Complete SPC Summary (Excel)",
                data=buffer_spc.getvalue(),
                file_name=f"SPC_Summary_A118T_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
