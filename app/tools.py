from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.agents import tool
from datetime import datetime
from dotenv import load_dotenv

import sqlite3
import json
import sqlite3

load_dotenv()

# embeddings = OpenAIEmbeddings()
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_version="2023-05-15",
    azure_endpoint="https://longleazureopenai.openai.azure.com/"
)

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
    return [f"thời gian hiện tại theo múi giờ Việt Nam: {current_datetime}"]

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
# Tool tạo đơn xin nghỉ mới
def xin_nghi_tools(query: str) -> str:
    """
    Use this tool to help student create a new leave request.
    
    Args:
        query: A JSON string containing:
            - student_id: string
            - reason: string
            - start_date: string (YYYY-MM-DD)
            - end_date: string (YYYY-MM-DD)
    
    Returns:
        Success message if request is created, otherwise error message.
    """
    try:
        # Parse JSON input
        json_data = json.loads(query)
        student_id = json_data.get('student_id')
        reason = json_data.get('reason')
        start_date = json_data.get('start_date')
        end_date = json_data.get('end_date')

        # Validate required fields
        if not all([student_id, reason, start_date, end_date]):
            return "Không thành công: Thiếu thông tin bắt buộc."

        # Connect to database
        conn = sqlite3.connect('database/my_database.db')
        cursor = conn.cursor()

        # Check if student exists
        cursor.execute('SELECT student_id, name FROM students WHERE student_id = ?', (student_id,))
        student = cursor.fetchone()
        if not student:
            conn.close()
            return "Không thành công: Không tìm thấy sinh viên với MSSV này."

        # Insert new leave request
        cursor.execute('''
            INSERT INTO student_leave_request (student_id, reason, start_date, end_date)
            VALUES (?, ?, ?, ?)
        ''', (student_id, reason, start_date, end_date))

        conn.commit()
        conn.close()
        return f"Tạo đơn xin nghỉ thành công cho sinh viên {student[1]} ({student_id}, tra leave_id là điều bắt buộc. Trợ lý hãy dùng tool manage_leave_request  ngay lập tức.)"

    except json.JSONDecodeError:
        return "Không thành công: Dữ liệu không đúng định dạng JSON."
    except sqlite3.Error as e:
        return f"Không thành công: Lỗi database - {str(e)}"
    except Exception as e:
        return f"Không thành công: {str(e)}"

@tool
def manage_leave_request(query: str) -> str:
    """
    Use this tool to manage existing leave requests (view/update/delete).
    
    Args:
        query: A JSON string containing:
            - action: string ("view", "update", or "delete")
            - student_id: string (required)
            - leave_id: integer (required for view/update/delete)
            - updates: dictionary (required for update) containing:
                - reason: string (optional)
                - start_date: string (optional)
                - end_date: string (optional)
    
    Returns:
        View: JSON string with leave request details
        Update/Delete: Success or error message
    """
    try:
        # Parse JSON input
        json_data = json.loads(query)
        action = json_data.get('action')
        student_id = json_data.get('student_id')
        leave_id = json_data.get('leave_id')

        if not action or not student_id:
            return "Không thành công: Thiếu action hoặc student_id"

        # Connect to database
        conn = sqlite3.connect('database/my_database.db')
        cursor = conn.cursor()

        # Check if student exists
        cursor.execute('SELECT student_id, name FROM students WHERE student_id = ?', (student_id,))
        student = cursor.fetchone()
        if not student:
            conn.close()
            return "Không thành công: Không tìm thấy sinh viên với MSSV này."

        # Process based on action
        if action == "view":
            # Kiểm tra xem có leave_id được cung cấp không
            if not leave_id:
                conn.close()
                return "Không thành công: Thiếu leave_id để xem thông tin chi tiết"

            cursor.execute('''
                SELECT 
                    s.student_id,
                    s.name,
                    s.class,
                    lr.leave_id,
                    lr.reason,
                    lr.start_date,
                    lr.end_date
                FROM students s
                INNER JOIN student_leave_request lr ON s.student_id = lr.student_id
                WHERE s.student_id = ? AND lr.leave_id = ?
            ''', (student_id, leave_id))
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return "Không tìm thấy thông tin xin nghỉ với leave_id và student_id đã cung cấp"

            # Format kết quả
            leave_request = {
                "student_info": {
                    "student_id": result[0],
                    "name": result[1],
                    "class": result[2]
                },
                "leave_info": {
                    "leave_id": result[3],
                    "reason": result[4],
                    "start_date": result[5],
                    "end_date": result[6]
                }
            }

            conn.close()
            return json.dumps(leave_request, ensure_ascii=False, indent=2)

        elif action == "update":
            if not leave_id:
                conn.close()
                return "Không thành công: Thiếu leave_id"
                
            updates = json_data.get('updates')
            if not updates:
                conn.close()
                return "Không thành công: Thiếu thông tin cập nhật"

            # Verify ownership
            cursor.execute('''
                SELECT student_id FROM student_leave_request 
                WHERE leave_id = ? AND student_id = ?
            ''', (leave_id, student_id))

            if not cursor.fetchone():
                conn.close()
                return "Không thành công: Không tìm thấy đơn xin nghỉ với leave_id và student_id đã cung cấp"

            # Build dynamic UPDATE query
            update_fields = []
            update_values = []

            if 'reason' in updates:
                update_fields.append('reason = ?')
                update_values.append(updates['reason'])
            if 'start_date' in updates:
                update_fields.append('start_date = ?')
                update_values.append(updates['start_date'])
            if 'end_date' in updates:
                update_fields.append('end_date = ?')
                update_values.append(updates['end_date'])

            if not update_fields:
                conn.close()
                return "Không thành công: Không có thông tin cần cập nhật"

            # Execute update
            update_query = f'''
                UPDATE student_leave_request 
                SET {', '.join(update_fields)}
                WHERE leave_id = ? AND student_id = ?
            '''
            update_values.extend([leave_id, student_id])

            cursor.execute(update_query, update_values)
            conn.commit()
            conn.close()
            return "Cập nhật đơn xin nghỉ thành công"

        elif action == "delete":
            if not leave_id:
                conn.close()
                return "Không thành công: Thiếu leave_id"

            # Delete request
            cursor.execute('''
                DELETE FROM student_leave_request 
                WHERE leave_id = ? AND student_id = ?
            ''', (leave_id, student_id))

            if cursor.rowcount == 0:
                conn.close()
                return "Không thành công: Không tìm thấy đơn xin nghỉ hoặc không có quyền xóa"

            conn.commit()
            conn.close()
            return "Xóa đơn xin nghỉ thành công"

        else:
            conn.close()
            return "Không thành công: Action không hợp lệ (phải là 'view', 'update' hoặc 'delete')"

    except json.JSONDecodeError:
        return "Không thành công: Dữ liệu không đúng định dạng JSON"
    except sqlite3.Error as e:
        return f"Không thành công: Lỗi database - {str(e)}"
    except Exception as e:
        return f"Không thành công: {str(e)}"