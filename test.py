import pandas as pd
import os

# Danh sách các đường dẫn tệp .xlsx
xlsx_paths = [
    "/media/kaisinyru/UUO/AI_lord/data/Diem_trung_tuyen_nam_gan_nhat_2024.xlsx",
    "/media/kaisinyru/UUO/AI_lord/data/Hoc_phi_2024.xlsx",
    "/media/kaisinyru/UUO/AI_lord/data/nganh_2024.xlsx",
    "/media/kaisinyru/UUO/AI_lord/data/Phuong_thuc_2_xettuyenhocba_2024.xlsx",
    "/media/kaisinyru/UUO/AI_lord/data/Phuong_thuc_3_danhgianangluc2024.xlsx",
    "/media/kaisinyru/UUO/AI_lord/data/Phuong_thuc_4_dieukienuutien_2024.xlsx",
    "/media/kaisinyru/UUO/AI_lord/data/Tinh_hinh_viec_lam_2024.xlsx",
    "/media/kaisinyru/UUO/AI_lord/data/Quy_mo_dao_tao_2024.xlsx",
    "/media/kaisinyru/UUO/AI_lord/data/thong_tin_co_so_vc_2024.xlsx"
]

# Chuyển đổi từng tệp .xlsx thành .csv và thay thế tệp gốc
for xlsx_path in xlsx_paths:
    try:
        # Đọc tệp .xlsx với engine là 'openpyxl'
        df = pd.read_excel(xlsx_path, engine='openpyxl')
        
        # Tạo đường dẫn tệp .csv
        csv_path = xlsx_path.replace('.xlsx', '.csv')
        
        # Lưu tệp .csv
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # Sử dụng encoding 'utf-8-sig' để hỗ trợ tiếng Việt
        
        # Xóa tệp gốc .xlsx
        os.remove(xlsx_path)

        # Đổi tên tệp .csv thành tệp gốc
        os.rename(csv_path, xlsx_path)

        print(f"Chuyển đổi {xlsx_path} thành {csv_path} thành công!")
    
    except Exception as e:
        print(f"Xảy ra lỗi khi xử lý {xlsx_path}: {e}")

print("Chuyển đổi và thay thế tệp gốc hoàn tất!")
