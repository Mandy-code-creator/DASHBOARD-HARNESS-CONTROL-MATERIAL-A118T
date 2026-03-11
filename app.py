# ================================
# FULL STREAMLIT APP – FINAL STABLE VERSION (A118T FILTERED - NO CUT SCRAP)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================
# PAGE CONFIG
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

def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# ================================
# LOAD MAIN DATA
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1hC5nnxqDLjF8-wUm8gtj11_5HFMxBlogY84Z0cRCj2s/export?format=csv"

@st.cache_data
def load_main():
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

raw = load_main()

# Tự động làm sạch tên cột (tránh lỗi dư khoảng trắng từ Google Sheet)
raw.columns = raw.columns.str.strip().str.upper()

# ================================
# PRE-PROCESSING & DATE HANDLING
# ================================
data_period_str = "N/A"
if "PRODUCTION DATE" in raw.columns:
    raw["PRODUCTION DATE"] = pd.to_datetime(raw["PRODUCTION DATE"], errors='coerce')
    min_date = raw["PRODUCTION DATE"].min()
    max_date = raw["PRODUCTION DATE"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        data_period_str = f"{min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}"

current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
st.markdown(f"""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
    <strong>🕒 Report Generated:</strong> {current_time} &nbsp;&nbsp;|&nbsp;&nbsp; 
    <strong>📅 Data Period:</strong> {data_period_str}
</div>
""", unsafe_allow_html=True)

# Đổi tên cột linh hoạt
df = raw.rename(columns={
    "PRODUCT SPECIFICATION CODE": "Product_Spec",
    "HR STEEL GRADE": "Material",
    "CLAASIFY MATERIAL": "Rolling_Type",
    "CLASSIFY MATERIAL": "Rolling_Type",
    "TOP COATMASS": "Top_Coatmass",
    "ORDER GAUGE": "Order_Gauge",
    "COIL NO": "COIL_NO",
    "QUALITY_CODE": "Quality_Code",
    "QUALITY CODE": "Quality_Code",
    "STANDARD HARDNESS": "Std_Text",
    "HARDNESS 冶金": "Hardness_LAB",
    "HARDNESS 鍍鋅線 C": "Hardness_LINE",
    "TENSILE_YIELD": "YS",
    "TENSILE_TENSILE": "TS",
    "TENSILE_ELONG": "EL",
    "STANDARD TS MIN": "Standard TS min",
    "STANDARD TS MAX": "Standard TS max",
    "STANDARD YS MIN": "Standard YS min",
    "STANDARD YS MAX": "Standard YS max",
    "STANDARD EL MIN": "Standard EL min",
    "STANDARD EL MAX": "Standard EL max"
})

# Chốt chặn lỗi COIL_NO
if "COIL_NO" not in df.columns:
    possible_cols = [c for c in df.columns if 'COIL' in str(c)]
    if possible_cols:
        df["COIL_NO"] = df[possible_cols[0]]
    else:
        df["COIL_NO"] = df.index 

# LỌC ĐỘC QUYỀN CHO MÃ A118T
if "Material" in df.columns and "Product_Spec" in df.columns:
    df = df[(df["Material"].astype(str).str.upper() == "A118T") | 
            (df["Product_Spec"].astype(str).str.upper() == "A118T")]
elif "Material" in df.columns:
    df = df[df["Material"].astype(str).str.upper() == "A118T"]

if df.empty:
    st.error("⚠️ Không tìm thấy dữ liệu nào cho mã A118T trong tệp nguồn.")
    st.stop()

def split_std(x):
    if isinstance(x, str) and "~" in x:
        lo, hi = x.split("~")
        return float(lo), float(hi)
    return np.nan, np.nan

if "Std_Text" in df.columns:
    df[["Std_Min","Std_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))
else:
    df[["Std_Min","Std_Max"]] = np.nan, np.nan

numeric_cols = ["Hardness_LAB", "Hardness_LINE", "YS", "TS", "EL", "Order_Gauge", "Standard TS min", "Standard TS max", "Standard YS min", "Standard YS max", "Standard EL min", "Standard EL max"]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "Quality_Code" in df.columns:
    df["Quality_Group"] = df["Quality_Code"].replace({"CQ00": "CQ00 / CQ06", "CQ06": "CQ00 / CQ06"})
else:
    df["Quality_Group"] = "Default"

if "Quality_Code" in df.columns:
    df = df[~(df["Quality_Code"].astype(str).str.startswith("GE") & ((df["Hardness_LAB"] < 88) | (df["Hardness_LINE"] < 88)))]

def apply_company_rules(row):
    std_min = row["Std_Min"] if "Std_Min" in row and pd.notna(row["Std_Min"]) else 0
    std_max = row["Std_Max"] if "Std_Max" in row and pd.notna(row["Std_Max"]) else 0
    lab_min, lab_max = 0, 0
    rule_name = "Standard (Excel)"

    is_cold = "COLD" in str(row.get("Rolling_Type", "")).upper()
    q_grp = str(row.get("Quality_Group", ""))
    target_qs = ["CQ00", "CQ06", "CQ07", "CQB0"]
    is_target_q = any(q in q_grp for q in target_qs)

    if is_cold and is_target_q:
        mat = str(row.get("Material", "")).upper().strip()
        if mat in ["A1081","A1081B"]: return 56.0, 62.0, 52.0, 70.0, "Rule A1081 (Cold)"
        elif mat in ["A108M","A108MR"]: return 60.0, 68.0, 55.0, 72.0, "Rule A108M (Cold)"
        elif mat in ["A108", "A108G", "A108R"]: return 58.0, 62.0, 52.0, 65.0, "Rule A108 (Cold)"

    return std_min, std_max, lab_min, lab_max, rule_name

df[['Limit_Min', 'Limit_Max', 'Lab_Min', 'Lab_Max', 'Rule_Name']] = df.apply(apply_company_rules, axis=1, result_type="expand")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

GAUGE_URL = "https://docs.google.com/spreadsheets/d/1utstALOQXfPSEN828aMdkrM1xXF3ckjBsgCUdJbwUdM/export?format=csv"
@st.cache_data
def load_gauge():
    return pd.read_csv(GAUGE_URL)

try:
    gauge_df = load_gauge()
    gauge_df.columns = gauge_df.columns.str.strip()
    gauge_col = next(c for c in gauge_df.columns if "RANGE" in c.upper())

    def parse_range(text):
        nums = re.findall(r"\d+\.\d+|\d+", str(text))
        if len(nums) < 2: return None, None
        return float(nums[0]), float(nums[-1])

    ranges = []
    for _, r in gauge_df.iterrows():
        lo, hi = parse_range(r[gauge_col])
        if lo is not None: ranges.append((lo, hi, r[gauge_col]))

    def map_gauge(val):
        for lo, hi, name in ranges:
            if lo <= val < hi: return name
        return None

    df["Gauge_Range"] = df["Order_Gauge"].apply(map_gauge)
    df = df.dropna(subset=["Gauge_Range"])
except Exception as e:
    df["Gauge_Range"] = "All Ranges"

# ================================
# SIDEBAR FILTER
# ================================
st.sidebar.header("🎛 FILTER")

all_rolling = sorted(df["Rolling_Type"].astype(str).unique()) if "Rolling_Type" in df else ["Default"]
metal_col_exists = "METALLIC_TYPE" in df.columns or "Metallic_Type" in df.columns
if not metal_col_exists: df["Metallic_Type"] = "Default"
all_metal = sorted(df["Metallic_Type"].astype(str).unique())
all_qgroup = sorted(df["Quality_Group"].astype(str).unique()) if "Quality_Group" in df else ["Default"]

rolling = st.sidebar.radio("Rolling Type", all_rolling)
metal   = st.sidebar.radio("Metallic Type", all_metal)
qgroup  = st.sidebar.radio("Quality Group", all_qgroup)

df_master_full = df.copy() 

if "Rolling_Type" in df: df = df[df["Rolling_Type"].astype(str) == rolling]
if "Metallic_Type" in df: df = df[df["Metallic_Type"].astype(str) == metal]
if "Quality_Group" in df: df = df[df["Quality_Group"].astype(str) == qgroup]

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

GROUP_COLS = [c for c in ["Rolling_Type","Metallic_Type","Quality_Group","Gauge_Range","Material"] if c in df.columns]
cnt = df.groupby(GROUP_COLS).agg(N_Coils=("COIL_NO","nunique")).reset_index()
valid = cnt[cnt["N_Coils"] >= 1] 

if valid.empty:
    st.warning("⚠️ No valid coils found for the current filter.")
    st.stop()

# ==============================================================================
# 0. EXECUTIVE KPI DASHBOARD (OVERVIEW)
# ==============================================================================
if view_mode == "📊 Executive KPI Dashboard":
    st.markdown("## 📊 Executive KPI Dashboard (Overall Quality Overview)")
    
    extracted_dfs = []
    for _, grp in valid.iterrows():
        sub_df = df[
            (df["Gauge_Range"] == grp.get("Gauge_Range")) &
            (df["Material"] == grp.get("Material"))
        ]
        extracted_dfs.append(sub_df)
    
    if len(extracted_dfs) == 0:
        st.warning("⚠️ No data matches the current filter.")
    else:
        full_df = pd.concat(extracted_dfs)
        df_kpi = full_df.dropna(subset=['TS', 'YS', 'EL', 'Hardness_LINE']).copy()
        
        if df_kpi.empty:
            st.warning("⚠️ The coils in this filter lack sufficient data to generate KPIs.")
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
            
            yield_rate = df_kpi['All_Pass'].mean() * 100
            hrb_yield = df_kpi['HRB_Pass'].mean() * 100 
            ts_yield = df_kpi['TS_Pass'].mean() * 100
            ys_yield = df_kpi['YS_Pass'].mean() * 100
            el_yield = df_kpi['EL_Pass'].mean() * 100

            st.markdown("### 🏆 Overall Quality Metrics")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            col1.metric("📦 Total Coils Tested", f"{total_coils:,}")
            
            delta_mech = clean_num(yield_rate - 100, True) if yield_rate < 100 else "Perfect"
            col2.metric("✅ Mech Yield Rate", clean_num(yield_rate, True), delta_mech, delta_color="normal" if yield_rate == 100 else "inverse")
            
            delta_hrb = clean_num(hrb_yield - 100, True) if hrb_yield < 100 else "In Control"
            col3.metric("🎯 HRB Yield Rate", clean_num(hrb_yield, True), delta_hrb, delta_color="normal" if hrb_yield == 100 else "inverse")
            
            col4.metric("TS Pass", clean_num(ts_yield, True))
            col5.metric("YS Pass", clean_num(ys_yield, True))
            col6.metric("EL Pass", clean_num(el_yield, True))
            
            st.markdown("---")
            st.success("Dữ liệu cơ tính và độ cứng đã được phân tích. Vui lòng chuyển các View Mode ở Sidebar để xem biểu đồ chi tiết.")
    st.stop()

# ==============================================================================
# MAIN LOOP FOR ALL OTHER VIEWS (REMAINS UNAFFECTED)
# ==============================================================================
for i, (_, g) in enumerate(valid.iterrows()):
    sub = df[
        (df["Gauge_Range"] == g.get("Gauge_Range")) &
        (df["Material"] == g.get("Material"))
    ].sort_values("COIL_NO")

    lo, hi = sub.iloc[0][["Limit_Min", "Limit_Max"]] 
    rule_used = sub.iloc[0]["Rule_Name"]
    l_lo, l_hi = sub.iloc[0][["Lab_Min", "Lab_Max"]]

    sub["NG_LAB"] = (sub["Hardness_LAB"] < lo) | (sub["Hardness_LAB"] > hi)
    sub["NG_LINE"] = (sub["Hardness_LINE"] < lo) | (sub["Hardness_LINE"] > hi)
    sub["NG"] = sub["NG_LAB"] | sub["NG_LINE"] 

    specs = ", ".join(sorted(sub["Product_Spec"].dropna().astype(str).unique())) if "Product_Spec" in sub.columns else "N/A"
    q_grp_disp = g.get('Quality_Group', 'N/A')
    
    st.markdown(f"### 🧱 {q_grp_disp} | {g.get('Material')} | {g.get('Gauge_Range')}")
    st.markdown(f"**Specs:** {specs} | **Coils:** {sub['COIL_NO'].nunique()} | **Limit:** {lo:.1f}~{hi:.1f}")
        
    if view_mode == "📋 Data Inspection":
        def highlight_ng_rows(row): return ['background-color: #ffe6e6'] * len(row) if row['NG'] else [''] * len(row)
        num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(sub.style.format("{:.0f}", subset=num_cols).apply(highlight_ng_rows, axis=1), use_container_width=True)

    elif view_mode == "📉 Hardness Analysis (Trend & Dist)":
        x = np.arange(1, len(sub)+1)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(x, sub["Hardness_LAB"], marker="o", linewidth=2, label="LAB", alpha=0.5)
        ax.plot(x, sub["Hardness_LINE"], marker="s", linewidth=2, label="LINE", alpha=0.9) 
        ax.axhline(lo, linestyle="--", linewidth=2, color="red", label=f"Control LSL={lo}")
        ax.axhline(hi, linestyle="--", linewidth=2, color="red", label=f"Control USL={hi}")
        ax.set_title("Hardness Trend by Coil Sequence", weight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=4)
        plt.tight_layout(); st.pyplot(fig)

    elif view_mode == "🧮 Predict TS/YS/EL from Std Hardness":
        st.markdown(f"### 🧮 AI Prediction: {g.get('Material')}")
        train_df = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])
        if len(train_df) < 5:
            st.warning("⚠️ Cần ít nhất 5 cuộn thép (coils) để kích hoạt AI Prediction.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                target_h = st.number_input("🎯 Target Hardness", value=float(round(train_df["Hardness_LINE"].mean(), 1)), step=0.1, key=f"ai_fix_{i}")
            
            X_train = train_df[["Hardness_LINE"]].values
            preds = {}
            for col in ["TS", "YS", "EL"]:
                model = LinearRegression().fit(X_train, train_df[col].values)
                preds[col] = model.predict([[target_h]])[0] 

            st.markdown("#### 🏁 Forecast Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tensile Strength (TS)", f"{int(round(preds['TS']))} MPa")
            c2.metric("Yield Strength (YS)", f"{int(round(preds['YS']))} MPa")
            c3.metric("Elongation (EL)", f"{round(preds['EL'], 1)} %")
