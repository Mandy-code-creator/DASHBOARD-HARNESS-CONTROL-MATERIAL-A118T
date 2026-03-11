# ================================
# FULL STREAMLIT APP – A118T STREAMLINED VERSION
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

def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# ================================
# ================================
# LOAD & CLEAN DATA
# ================================
DATA_URL = "https://docs.google.com/spreadsheets/d/1hC5nnxqDLjF8-wUm8gtj11_5HFMxBlogY84Z0cRCj2s/export?format=csv"

@st.cache_data
def load_main():
    r = requests.get(DATA_URL)
    r.encoding = "utf-8"
    return pd.read_csv(StringIO(r.text))

raw = load_main()

# ---------------------------------------------------------
# TỰ ĐỘNG LÀM SẠCH TÊN CỘT (Super Clean)
# ---------------------------------------------------------
# Dùng Regex (r'\s+') để gom TẤT CẢ khoảng trắng, dấu xuống dòng, tab... thành 1 dấu cách duy nhất
raw.columns = raw.columns.str.replace(r'\s+', ' ', regex=True).str.replace('_', ' ', regex=False).str.strip().str.upper()

# Đổi tên cột chuẩn hóa
df = raw.rename(columns={
    "PRODUCT SPECIFICATION CODE": "Product_Spec",
    "HR STEEL GRADE": "Material",
    "CLASSIFY": "Rolling_Type",       
    "QUALITY CODE": "Quality_Code",   
    "ORDER GAUGE": "Order_Gauge",
    "TOP COATMASS": "Top_Coatmass",
    "METALLIC COATING TYPE": "Metallic_Type",
    "COIL NO": "COIL_NO",
    "STANDARD HARDNESS": "Std_Text",
    "HARDNESS 冶金": "Hardness_LAB",
    "HARDNESS 鍍鋅線 C": "Hardness_LINE", 
    "TENSILE YIELD": "YS",            
    "TENSILE TENSILE": "TS",          
    "TENSILE ELONG": "EL",            
    "STANDARD TS MIN": "Standard TS min",
    "STANDARD TS MAX": "Standard TS max",
    "STANDARD YS MIN": "Standard YS min",
    "STANDARD YS MAX": "Standard YS max",
    "STANDARD EL MIN": "Standard EL min",
    "STANDARD EL MAX": "Standard EL max"
})

# --- BỘ QUÉT & CHỐT CHẶN AN TOÀN ---

# 1. Chốt chặn cho COIL_NO
if "COIL_NO" not in df.columns:
    possible_cols = [c for c in df.columns if 'COIL' in str(c)]
    df["COIL_NO"] = df[possible_cols[0]] if possible_cols else df.index 

# 2. Chốt chặn cho Material
if "Material" not in df.columns:
    possible_mats = [c for c in df.columns if any(k in str(c) for k in ['GRADE', 'MATERIAL', 'STEEL', 'MAC'])]
    df["Material"] = df[possible_mats[0]] if possible_mats else "A118T"

# 3. Chốt chặn cho Hardness_LINE (Trị dứt điểm KeyError)
if "Hardness_LINE" not in df.columns:
    # Tìm cột chứa chữ '鍍鋅線 C' hoặc có cả 'HARDNESS' và 'C'
    possible_line = [c for c in df.columns if '鍍鋅線 C' in str(c) or ('HARDNESS' in str(c) and 'C' in str(c))]
    if possible_line:
        df["Hardness_LINE"] = df[possible_line[0]]
    else:
        # Nếu vẫn không tìm ra, lấy bừa cột HARDNESS nào đó (không phải 冶金) làm LINE
        fallback_hard = [c for c in df.columns if 'HARDNESS' in str(c) and '冶金' not in str(c)]
        if fallback_hard:
            df["Hardness_LINE"] = df[fallback_hard[0]]
        else:
            st.error("⚠️ Không tìm thấy cột dữ liệu độ cứng Line (HARDNESS 鍍鋅線 C). Vui lòng kiểm tra lại file Google Sheet.")
            st.stop()

# 4. Chốt chặn cho Hardness_LAB
if "Hardness_LAB" not in df.columns:
    possible_lab = [c for c in df.columns if '冶金' in str(c) or 'LAB' in str(c)]
    df["Hardness_LAB"] = df[possible_lab[0]] if possible_lab else df["Hardness_LINE"]

# ================================

# ================================
# LỌC ĐỘC QUYỀN A118T & XỬ LÝ SỐ LIỆU
# ================================
df = df[(df["Material"].astype(str).str.upper().str.contains("A118T")) | 
        (df.get("Product_Spec", pd.Series(dtype=str)).astype(str).str.upper().str.contains("A118T"))].copy()

if df.empty:
    st.error("⚠️ Không tìm thấy dữ liệu nào cho mã A118T trong tệp nguồn.")
    st.stop()

# Xử lý giới hạn độ cứng từ Standard Hardness (ví dụ: "80~93" -> 80 và 93)
def split_std(x):
    if isinstance(x, str) and "~" in x:
        try:
            lo, hi = x.split("~")
            return float(lo.strip()), float(hi.strip())
        except: pass
    return np.nan, np.nan

if "Std_Text" in df.columns:
    df[["Limit_Min","Limit_Max"]] = df["Std_Text"].apply(lambda x: pd.Series(split_std(x)))
else:
    df[["Limit_Min","Limit_Max"]] = 80.0, 93.0 # Fallback default for A118T if column missing

df["Lab_Min"] = df["Limit_Min"]
df["Lab_Max"] = df["Limit_Max"]
df["Rule_Name"] = "Direct Spec"

# Ép kiểu dữ liệu số
numeric_cols = ["Hardness_LAB", "Hardness_LINE", "YS", "TS", "EL", "Order_Gauge", 
                "Standard TS min", "Standard TS max", "Standard YS min", "Standard YS max", 
                "Standard EL min", "Standard EL max", "Limit_Min", "Limit_Max"]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop dữ liệu thiếu mốc Hardness LINE
df = df.dropna(subset=["Hardness_LINE"])

# ================================
# SIDEBAR FILTER (Đã làm gọn)
# ================================
st.sidebar.header("🎛 FILTER (A118T)")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

all_rolling = sorted(df["Rolling_Type"].dropna().astype(str).unique()) if "Rolling_Type" in df else ["Default"]
all_coatmass = sorted(df["Top_Coatmass"].dropna().astype(str).unique()) if "Top_Coatmass" in df else ["Default"]
all_gauge = sorted(df["Order_Gauge"].dropna().astype(float).unique()) if "Order_Gauge" in df else [0.0]

rolling = st.sidebar.selectbox("1. Rolling Type", ["All"] + list(all_rolling))
coatmass = st.sidebar.selectbox("2. Top Coatmass", ["All"] + list(all_coatmass))
gauge = st.sidebar.selectbox("3. Order Gauge (Thickness)", ["All"] + list(all_gauge))

# Két sắt dữ liệu
df_master_full = df.copy() 

# Áp dụng bộ lọc
if rolling != "All" and "Rolling_Type" in df: df = df[df["Rolling_Type"].astype(str) == rolling]
if coatmass != "All" and "Top_Coatmass" in df: df = df[df["Top_Coatmass"].astype(str) == str(coatmass)]
if gauge != "All" and "Order_Gauge" in df: df = df[df["Order_Gauge"].astype(float) == float(gauge)]

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

# Nhóm dữ liệu đơn giản hóa
GROUP_COLS = [c for c in ["Rolling_Type", "Top_Coatmass", "Order_Gauge", "Material"] if c in df.columns]
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
        st.warning("⚠️ The coils in this filter lack sufficient mechanical data to generate KPIs.")
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
        st.success("Dữ liệu cơ tính và độ cứng đã được phân tích thành công. Vui lòng chuyển các View Mode ở Sidebar để xem biểu đồ chi tiết.")
    st.stop()

# ==============================================================================
# 🚀 GLOBAL SUMMARY DASHBOARD
# ==============================================================================
if view_mode == "🚀 Global Summary Dashboard":
    st.markdown("## 🚀 Global Process Dashboard (A118T)")
    tab1, tab2 = st.tabs(["📊 1. Performance Overview", "🧠 2. Decision Support (Risk AI)"])

    with tab1:
        st.info("ℹ️ Color Guide: 🟢 High Pass Rate (>98%) | 🔴 Low Pass Rate (<90%)")
        stats_rows = []
        for _, g in valid.iterrows():
            # Tạo chuỗi query động dựa trên GROUP_COLS
            conditions = []
            for col in GROUP_COLS:
                conditions.append(df[col] == g[col])
            sub_grp = df[np.logical_and.reduce(conditions)].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 1: continue

            l_min_val = sub_grp['Limit_Min'].min(); l_max_val = sub_grp['Limit_Max'].max()
            lim_hrb = f"{l_min_val:.0f}~{l_max_val:.0f}"
            
            def get_limit_str(s_min, s_max):
                v_min = sub_grp[s_min].max() if s_min in sub_grp else 0 
                v_max = sub_grp[s_max].min() if s_max in sub_grp else 0 
                if pd.isna(v_min): v_min = 0
                if pd.isna(v_max): v_max = 0
                if v_min > 0 and v_max > 0 and v_max < 9000: return f"{v_min:.0f}~{v_max:.0f}"
                elif v_min > 0: return f"≥ {v_min:.0f}"
                elif v_max > 0 and v_max < 9000: return f"≤ {v_max:.0f}"
                else: return "-"

            n_total = len(sub_grp)
            n_ng = sub_grp[(sub_grp["Hardness_LINE"] < sub_grp["Limit_Min"]) | (sub_grp["Hardness_LINE"] > sub_grp["Limit_Max"])].shape[0]
            pass_rate = ((n_total - n_ng) / n_total) * 100

            row_data = {col: g[col] for col in GROUP_COLS}
            row_data.update({
                "HRB Limit": lim_hrb, "N": n_total, "Pass Rate": pass_rate,
                "HRB (Avg)": sub_grp["Hardness_LINE"].mean(), "TS (Avg)": sub_grp["TS"].mean(),
                "YS (Avg)": sub_grp["YS"].mean(), "EL (Avg)": sub_grp["EL"].mean(),
                "TS Limit": get_limit_str("Standard TS min", "Standard TS max"), 
                "YS Limit": get_limit_str("Standard YS min", "Standard YS max"), 
                "EL Limit": get_limit_str("Standard EL min", "Standard EL max")
            })
            stats_rows.append(row_data)

        if stats_rows:
            df_stats = pd.DataFrame(stats_rows)
            def color_pass_rate(val):
                color = '#d4edda' if val >= 98 else ('#fff3cd' if val >= 90 else '#f8d7da')
                text_color = '#155724' if val >= 98 else ('#856404' if val >= 90 else '#721c24')
                return f'background-color: {color}; color: {text_color}; font-weight: bold'

            st.dataframe(
                df_stats.style.format("{:.1f}", subset=[c for c in df_stats.columns if "(Avg)" in c or "Pass" in c])
                .applymap(color_pass_rate, subset=["Pass Rate"])
                .background_gradient(subset=["HRB (Avg)"], cmap="Blues"),
                use_container_width=True
            )
        else: st.warning("Insufficient data.")

    with tab2:
        st.markdown("#### 🧠 AI Decision Support (Risk-Based)")
        col_in1, col_in2 = st.columns([1, 1])
        with col_in1: user_hrb = st.number_input("1️⃣ Target HRB", value=85.0, step=0.5, format="%.1f")
        with col_in2: safety_k = st.selectbox("2️⃣ Select Safety Factor:", [1.0, 2.0, 3.0], index=1)

        rows_ts, rows_ys, rows_el = [], [], []
        
        for _, g in valid.iterrows():
            conditions = [df[col] == g[col] for col in GROUP_COLS]
            sub_grp = df[np.logical_and.reduce(conditions)].dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])

            if len(sub_grp) < 3: continue 

            spec_ts_min = sub_grp["Standard TS min"].max() if "Standard TS min" in sub_grp else 0
            spec_ys_min = sub_grp["Standard YS min"].max() if "Standard YS min" in sub_grp else 0
            spec_el_min = sub_grp["Standard EL min"].max() if "Standard EL min" in sub_grp else 0
            
            X = sub_grp[["Hardness_LINE"]].values
            
            def get_pred_risk(model_target, spec_min):
                m = LinearRegression().fit(X, sub_grp[model_target].values)
                pred = m.predict([[user_hrb]])[0]
                err = np.sqrt(mean_squared_error(sub_grp[model_target], m.predict(X)))
                safe = pred - (safety_k * err)
                risk = "🔴 High Risk" if (spec_min > 0 and safe < spec_min) else "🟢 Safe"
                return pred, safe, risk

            try:
                base_dict = {col: g[col] for col in GROUP_COLS}
                
                pred_ts, safe_ts, risk_ts = get_pred_risk("TS", spec_ts_min)
                r_ts = base_dict.copy(); r_ts.update({"Pred TS": f"{pred_ts:.0f}", "Worst Case": f"{safe_ts:.0f}", "Limit": f"≥ {spec_ts_min:.0f}" if spec_ts_min > 0 else "-", "Status": risk_ts}); rows_ts.append(r_ts)
                
                pred_ys, safe_ys, risk_ys = get_pred_risk("YS", spec_ys_min)
                r_ys = base_dict.copy(); r_ys.update({"Pred YS": f"{pred_ys:.0f}", "Worst Case": f"{safe_ys:.0f}", "Limit": f"≥ {spec_ys_min:.0f}" if spec_ys_min > 0 else "-", "Status": risk_ys}); rows_ys.append(r_ys)
                
                pred_el, safe_el, risk_el = get_pred_risk("EL", spec_el_min)
                r_el = base_dict.copy(); r_el.update({"Pred EL": f"{pred_el:.1f}", "Worst Case": f"{safe_el:.1f}", "Limit": f"≥ {spec_el_min:.1f}" if spec_el_min > 0 else "-", "Status": risk_el}); rows_el.append(r_el)
            except: pass

        if rows_ts:
            def style_risk(val): return 'color: red; font-weight: bold' if "🔴" in val else 'color: green; font-weight: bold'
            c_top1, c_top2 = st.columns(2)
            with c_top1:
                st.markdown("##### 🔹 Tensile Strength (TS)")
                st.dataframe(pd.DataFrame(rows_ts).style.applymap(style_risk, subset=["Status"]), use_container_width=True, hide_index=True)
            with c_top2:
                st.markdown("##### 🔸 Yield Strength (YS)")
                st.dataframe(pd.DataFrame(rows_ys).style.applymap(style_risk, subset=["Status"]), use_container_width=True, hide_index=True)
            st.markdown("---")
            st.markdown("##### 🔻 Elongation (EL)")
            st.dataframe(pd.DataFrame(rows_el).style.applymap(style_risk, subset=["Status"]), use_container_width=True, hide_index=True)
        else: st.warning("Insufficient data.")
    st.stop()

# ==============================================================================
# MASTER DICTIONARY EXPORT (FULL VIEW)
# ==============================================================================
if view_mode == "👑 Master Dictionary Export":
    st.markdown("---")
    st.header("👑 Master Mechanical Properties Dictionary (A118T)")
    st.info("Scan toàn bộ nhà máy để thiết lập giới hạn sản xuất tối ưu (Target Limits) và giới hạn kiểm soát thống kê (Control Limits).")

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
        tab_trend, tab_dist = st.tabs(["📈 Trend Analysis", "📊 Distribution"])
        with tab_trend:
            x = np.arange(1, len(sub)+1)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            if "Hardness_LAB" in sub.columns and not sub["Hardness_LAB"].isna().all():
                ax.plot(x, sub["Hardness_LAB"], marker="o", linewidth=2, label="LAB", alpha=0.5)
            ax.plot(x, sub["Hardness_LINE"], marker="s", linewidth=2, label="LINE", alpha=0.9) 
            ax.axhline(lo, linestyle="--", linewidth=2, color="red", label=f"LSL={lo}")
            ax.axhline(hi, linestyle="--", linewidth=2, color="red", label=f"USL={hi}")
            ax.set_title("Hardness Trend by Coil Sequence", weight="bold")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=4)
            plt.tight_layout(); st.pyplot(fig)
        with tab_dist:
            st.info("Sử dụng Control Limit Calculator để xem phân phối chi tiết.")

    elif view_mode == "🔗 Correlation: Hardness vs Mech Props":
        st.dataframe(sub[["Hardness_LINE", "TS", "YS", "EL"]].corr().style.background_gradient(cmap='coolwarm'), use_container_width=True)

    elif view_mode == "⚙️ Mech Props Analysis":
        sub_mech = sub.dropna(subset=["TS","YS","EL"])
        if sub_mech.empty: st.warning("No Mech Data")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for j, col in enumerate(["TS", "YS", "EL"]):
                data = sub_mech[col]
                axes[j].hist(data, bins=20, alpha=0.7, color=['#1f77b4', '#2ca02c', '#ff7f0e'][j])
                axes[j].set_title(f"{col} Distribution\n(Mean: {data.mean():.1f})")
                axes[j].grid(alpha=0.3)
            st.pyplot(fig)

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
        train_df = sub.dropna(subset=["Hardness_LINE", "TS", "YS", "EL"])
        if len(train_df) < 3:
            st.warning("⚠️ Need at least 3 coils to activate AI Prediction.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                target_h = st.number_input("🎯 Target Hardness", value=float(round(train_df["Hardness_LINE"].mean(), 1)), step=0.1, key=f"ai_{i}")
            
            X_train = train_df[["Hardness_LINE"]].values
            preds = {}
            for col in ["TS", "YS", "EL"]:
                model = LinearRegression().fit(X_train, train_df[col].values)
                preds[col] = model.predict([[target_h]])[0] 

            c1, c2, c3 = st.columns(3)
            c1.metric("Tensile Strength (TS)", f"{int(round(preds['TS']))} MPa")
            c2.metric("Yield Strength (YS)", f"{int(round(preds['YS']))} MPa")
            c3.metric("Elongation (EL)", f"{round(preds['EL'], 1)} %")

    elif view_mode == "🎛️ Control Limit Calculator (Compare 3 Methods)":
        data = sub["Hardness_LINE"].dropna()
        if len(data) < 5: st.warning(f"⚠️ Not enough data (N={len(data)})")
        else:
            mu, std_dev = data.mean(), data.std()
            mrs = np.abs(np.diff(data)); mr_bar = np.mean(mrs); sigma_imr = mr_bar / 1.128
            
            m1_min, m1_max = mu - 3*std_dev, mu + 3*std_dev
            m4_min, m4_max = mu - 3*sigma_imr, mu + 3*sigma_imr
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(data, bins=15, density=True, alpha=0.6, color="#1f77b4")
            ax.axvline(m1_min, c="red", ls=":", label="M1: Standard (3σ)")
            ax.axvline(m1_max, c="red", ls=":")
            ax.axvline(m4_min, c="purple", ls="-.", lw=2, label="M4: I-MR (SPC)")
            ax.axvline(m4_max, c="purple", ls="-.", lw=2)
            if lo > 0: ax.axvline(lo, c="black", lw=2, label="Standard Limit")
            if hi > 0 and hi < 9000: ax.axvline(hi, c="black", lw=2)
            ax.set_title(f"Control Limits Comparison")
            ax.legend()
            st.pyplot(fig)
