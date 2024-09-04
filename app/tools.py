from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.agents import tool
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings()

index_names = {
    "de_an": "/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/vector_database/faiss_index_de-an-tuyen-sinh-dhntt-2024-v02-web",
    "nang_khieu": "/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/vector_database/faiss_index_nang_khieu",
    "diem": "/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/vector_database/faiss_index_Diem_trung_tuyen_nam_gan_nhat_2024",
    "hoc_phi": "/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/vector_database/faiss_index_Hoc_phi_2024",
    "nganh": "/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/vector_database/faiss_index_nganh_2024",
    "phuong_thuc_hoc_ba": "/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/vector_database/faiss_index_Phuong_thuc_2_xettuyenhocba_2024",
    "phuong_thuc_nang_luc": "/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/vector_database/faiss_index_Phuong_thuc_3_danhgianangluc2024",
    "phuong_thuc_uu_tien": "/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/vector_database/faiss_index_Phuong_thuc_4_dieukienuutien_2024",
    "tinh_hinh_viec_lam": "/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/vector_database/faiss_index_Tinh_hinh_viec_lam_2024"
}

retrievers = {key: FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 8}) for key, path in index_names.items()}

from langchain.tools import tool

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
    return combined_info[:8]  # Return only the top 8 results