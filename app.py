# 3. Tự động làm sạch và Rename Columns
# Xóa khoảng trắng thừa và viết hoa tất cả tiêu đề để tránh lỗi gõ sai từ file nguồn
raw.columns = raw.columns.str.strip().str.upper()

df = raw.rename(columns={
    "PRODUCT SPECIFICATION CODE": "Product_Spec",
    "HR STEEL GRADE": "Material",
    "CLAASIFY MATERIAL": "Rolling_Type",
    "CLASSIFY MATERIAL": "Rolling_Type", # Dự phòng đúng/sai chính tả
    "TOP COATMASS": "Top_Coatmass",
    "ORDER GAUGE": "Order_Gauge",
    "COIL NO": "COIL_NO",
    "COIL_NO": "COIL_NO", # Dự phòng nếu file đã có sẵn dấu gạch dưới
    "QUALITY_CODE": "Quality_Code",
    "QUALITY CODE": "Quality_Code", # Dự phòng thiếu gạch dưới
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

# --- BỔ SUNG CHỐT CHẶN AN TOÀN CHO COIL_NO ---
if "COIL_NO" not in df.columns:
    # Quét tìm cột nào có chữ 'COIL' để lấy làm mã cuộn
    possible_cols = [c for c in df.columns if 'COIL' in str(c)]
    if possible_cols:
        df["COIL_NO"] = df[possible_cols[0]]
    else:
        # Nếu file hoàn toàn không có cột mã cuộn, tạo ID tự động dựa trên số thứ tự dòng
        df["COIL_NO"] = df.index
