import json
import os
from datetime import datetime
from typing import Any, Dict, List

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")

    # 1. Kiểm tra file có tồn tại không
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    # 2. Đọc file JSON và lọc sản phẩm
    with open(catalog_file, "r", encoding="utf-8") as f:
        products = json.load(f)

    # 3. Lọc theo danh mục và giá tối đa
    filtered_products = [
        p
        for p in products
        if p.get("category") == category and p.get("price_vnd", 0) <= max_price
    ]

    return filtered_products


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, customer_name, status, ...
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # 1. Đọc dữ liệu ticket hiện có (nếu file chưa tồn tại thì khởi tạo danh sách rỗng)
    if os.path.exists(tickets_file):
        try:
            with open(tickets_file, "r", encoding="utf-8") as f:
                tickets = json.load(f)
        except (json.JSONDecodeError, IOError):
            tickets = []
    else:
        tickets = []

    # 2. Sinh ticket_id theo định dạng: TK-{today}-{seq:03d}
    today = datetime.now().strftime("%Y%m%d")
    today_tickets_count = sum(
        1
        for t in tickets
        if str(t.get("ticket_id", "")).startswith(f"TK-{today}")
    )
    seq = today_tickets_count + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    # 3. Tạo bản ghi ticket mới và append vào danh sách
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }
    tickets.append(new_ticket)

    # 4. Ghi đè lại file support_tickets.json
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(tickets, f, ensure_ascii=False, indent=2)

    # 5. Trả về đầy đủ thông tin ticket vừa tạo
    return new_ticket


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, gói du lịch"
            " Vinpearl) theo danh mục và giá tối đa."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Loại sản phẩm: 'xe_dien' cho xe điện VinFast hoặc"
                        " 'du_lich' cho các gói du lịch Vinpearl."
                    ),
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": (
                        "Giá tối đa tính bằng VNĐ. Ví dụ: 1000000000. Mặc định"
                        " không giới hạn."
                    ),
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Ghi nhận yêu cầu hỗ trợ hoặc phản hồi của khách hàng vào hệ thống"
            " ticket."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Tên khách hàng yêu cầu hỗ trợ.",
                },
                "issue_description": {
                    "type": "string",
                    "description": (
                        "Mô tả chi tiết vấn đề hoặc khiếu nại cần hỗ trợ."
                    ),
                },
                "priority": {
                    "type": "string",
                    "description": (
                        "Mức độ ưu tiên của yêu cầu: 'low', 'medium', hoặc"
                        " 'high'."
                    ),
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]

# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}