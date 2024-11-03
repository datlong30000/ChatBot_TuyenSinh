from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.agents import tool
from datetime import datetime
from dotenv import load_dotenv
import sqlite3
import json
import sqlite3

load_dotenv()

embeddings = OpenAIEmbeddings()

index_names = {
    "de_an": "vector_database/faiss_index_de-an-tuyen-sinh-dhntt-2024-v02-web",
    "nang_khieu": "vector_database/faiss_index_nang_khieu",
    "diem": "vector_database/faiss_index_Diem_trung_tuyen_nam_gan_nhat_2024",
    "hoc_phi": "vector_database/faiss_index_Hoc_phi_2024",
    "nganh": "vector_database/faiss_index_nganh_2024",
    "phuong_thuc_hoc_ba": "vector_database/faiss_index_Phuong_thuc_2_xettuyenhocba_2024",
    "phuong_thuc_nang_luc": "vector_database/faiss_index_Phuong_thuc_3_danhgianangluc2024",
    "phuong_thuc_uu_tien": "vector_database/faiss_index_Phuong_thuc_4_dieukienuutien_2024",
    "tinh_hinh_viec_lam": "vector_database/faiss_index_Tinh_hinh_viec_lam_2024"
}

retrievers = {key: FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 8}) for key, path in index_names.items()}

tool_tavily = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

@tool
def tavily_search_web(query:str) -> list:
    """Use this tool to get information from website by Tavily.

    Args:
        query: The specific question or topic that doesn't include in the dataset.

    Returns:
        A list containing the information about the user's question.
    """
    return tool_tavily.invoke({"query": query})

@tool
def de_an_tools(query: str) -> list:
    """Use this tool to get information about admission plans at Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to admission plans.

    Returns:
        A list containing the top 8 pieces of information about the admission plans based on the query.
    """
    return retrievers["de_an"].invoke(query)

@tool
def diem_trung_tuyen_tools(query: str) -> list:
    """Use this tool to get information about scores and grade requirements at Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to scores or grade requirements.

    Returns:
        A list containing the top 8 pieces of information about scores or grade requirements based on the query.
    """
    return retrievers["diem"].invoke(query)

@tool
def hoc_phi_tools(query: str) -> list:
    """Use this tool to get information about tuition fees at Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to tuition fees.

    Returns:
        A list containing the top 8 pieces of information about tuition fees based on the query.
    """
    return retrievers["hoc_phi"].invoke(query)

@tool
def nang_khieu_tools(query: str) -> list:
    """Use this tool to get information about special talents or skills requirements at Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to special talents or skills requirements.

    Returns:
        A list containing the top 8 pieces of information about special talents or skills requirements based on the query.
    """
    return retrievers["nang_khieu"].invoke(query)

@tool
def nganh_hoc_tools(query: str) -> list:
    """Use this tool to get information about majors and programs offered at Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to majors or programs.

    Returns:
        A list containing the top 8 pieces of information about majors or programs based on the query.
    """
    return retrievers["nganh"].invoke(query)

@tool
def pt_hoc_ba_tools(query: str) -> list:
    """Use this tool to get information about admission methods based on academic records at Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to admission methods based on academic records.

    Returns:
        A list containing the top 8 pieces of information about admission methods based on academic records for the query.
    """
    return retrievers["phuong_thuc_hoc_ba"].invoke(query)

@tool
def pt_nang_luc_tools(query: str) -> list:
    """Use this tool to get information about admission methods based on competency assessment at Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to admission methods based on competency assessment.

    Returns:
        A list containing the top 8 pieces of information about admission methods based on competency assessment for the query.
    """
    return retrievers["phuong_thuc_nang_luc"].invoke(query)

@tool
def pt_uu_tien_tools(query: str) -> list:
    """Use this tool to get information about priority admission methods at Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to priority admission methods.

    Returns:
        A list containing the top 8 pieces of information about priority admission methods based on the query.
    """
    return retrievers["phuong_thuc_uu_tien"].invoke(query)

@tool
def tinh_hinh_viec_lam_tools(query: str) -> list:
    """Use this tool to retrieve information about the employment situation at Nguyen Tat Thanh University.

    Args: 
        query: The specific question or topic related to the employment situation.

    Returns:
        A list containing the top 8 pieces of information about the employment situation based on the query.
    """
    return retrievers["tinh_hinh_viec_lam"].invoke(query)

## các hàm kết hợp so sánh
@tool
def hoc_phi_nganh_tools(query: str) -> list:
    """Use this tool to get combined information about tuition fees and majors at Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to tuition fees and majors.

    Returns:
        A list containing the top 8 pieces of combined information about tuition fees and majors based on the query.
    """
    hoc_phi_info = retrievers["hoc_phi"].invoke(query)
    nganh_info = retrievers["nganh"].invoke(query)
    combined_info = hoc_phi_info + nganh_info
    return combined_info[:8]  # Return only the top 8 results

@tool
def thoi_gian_hien_tai(time: str) -> dict:
    """
    Use this tool to get current time.

    Args:
        query: Determine time.

    Returns:
        an answer of time.
    """
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return ["thời gian hiện tại", current_datetime]

@tool
def thoi_gian_het_han_tools(query: str) -> list:
    """
    Use this tool to compare the current time and the time of university student recruitment rounds Nguyen Tat Thanh University.

    Args:
        query: The specific question or topic related to date of maturity.

    Returns:
        A list containing the top 8 pieces of combined information about current time and date of maturity.
    """
    tg_xet_tuyen_hb = retrievers["phuong_thuc_hoc_ba"].invoke(query)
    tg_xet_tuyen_ut = retrievers["phuong_thuc_uu_tien"].invoke(query)
    tg_xet_tuyen_dgnl = retrievers["phuong_thuc_nang_luc"].invoke(query)
    combined_info = tg_xet_tuyen_hb + tg_xet_tuyen_ut + tg_xet_tuyen_dgnl
    return combined_info

@tool
def xin_nghi_tools(query: str) -> str:
    """
    Use this tool to help student make a form for leave request.
    
    Args:
        query: A JSON string containing student ID, reason, start_date, end_date.
    
    Returns:
        A success message if the query is added to the SQL database, otherwise it will return errors.
    """
    try:
        # Chuyển đổi chuỗi JSON thành đối tượng Python
        json_data = json.loads(query)
        student_id = json_data.get('student_id')
        reason = json_data.get('reason')
        start_date = json_data.get('start_date')
        end_date = json_data.get('end_date')

        # Kiểm tra xem tất cả thông tin cần thiết đã được cung cấp chưa
        if student_id is None or reason is None or start_date is None or end_date is None:
            return "Không thành công: Thiếu thông tin cần thiết."

        # Kết nối tới cơ sở dữ liệu SQLite
        conn = sqlite3.connect('database/my_database.db')
        cursor = conn.cursor()

        # Kiểm tra xem MSSV có tồn tại trong bảng students hay không
        cursor.execute('''
            SELECT student_id FROM students WHERE student_id = ?
        ''', (student_id,))
        result = cursor.fetchone()

        # Nếu MSSV không tồn tại, trả về thông báo lỗi
        if result is None:
            conn.close()  # Đóng kết nối trước khi trả về
            return "Không thành công: MSSV bạn cung cấp không có trong cơ sở dữ liệu, bạn thử kiểm tra lại."

        # Thực hiện truy vấn INSERT vào bảng leave_requests
        cursor.execute(''' 
            INSERT INTO leave_requests (student_id, start_date, end_date, reason) 
            VALUES (?, ?, ?, ?) 
        ''', (student_id, start_date, end_date, reason))

        # Lưu thay đổi
        conn.commit()
        conn.close()

        return "Cập nhật thành công"

    except json.JSONDecodeError:
        return "Không thành công: Query không phải là định dạng JSON hợp lệ."
    except sqlite3.Error as e:
        return f"Không thành công: Lỗi cơ sở dữ liệu - {str(e)}"
    except Exception as e:
        return f"Không thành công: Đã xảy ra lỗi - {str(e)}"

