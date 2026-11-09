"""Lab #4: System Prompt Engineering & Tool Calling Engine

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + ReAct Loop + Tool Calling.
"""

from datetime import datetime
import json
import os
import re
from typing import Any, Dict, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant - trợ lý ảo AI chính thức và chuyên nghiệp của Tập đoàn Vingroup.

1. PERSONA:
- Phong cách giao tiếp: Lịch sự, chu đáo, chuẩn mực, cầu thị và thân thiện.
- Xưng hô: "Em" hoặc "VinAssistant" và gọi người dùng là "Quý khách" / "Anh/Chị".

2. AVAILABLE TOOLS:
- `search_product_catalog(category: str, max_price: int)`: Tra cứu danh mục sản phẩm (category gồm 'xe_dien' hoặc 'du_lich') theo ngân sách tối đa.
- `submit_support_ticket(customer_name: str, issue_description: str, priority: str)`: Tạo yêu cầu hỗ trợ/khiếu nại kỹ thuật và chăm sóc khách hàng vào hệ thống ticket.

3. CORE RULES (GROUND TRUTH & ANTI-HALLUCINATION):
- TUYỆT ĐỐI KHÔNG tự bịa đặt giá cả, thông số kỹ thuật xe hay chi tiết gói tour nếu không có trong kết quả trả về từ Tools.
- Khi người dùng hỏi về sản phẩm, giá bán, chính sách tour: Bắt buộc dùng `search_product_catalog`.
- Khi người dùng phàn nàn, báo lỗi kỹ thuật, cần khiếu nại hoặc đặt lịch bảo hành: Bắt buộc dùng `submit_support_ticket`.
- Nếu kết quả từ tool rỗng hoặc gặp lỗi: Báo rõ cho khách hàng một cách lịch sự, không suy đoán.

4. OPERATIONAL BOUNDARIES:
- Chỉ trả lời các nội dung liên quan trực tiếp đến hệ sinh thái Vingroup (VinFast, Vinpearl, Vinhomes, Vinmec, VinBigdata...).
- Từ chối lịch sự nếu người dùng hỏi về các chủ đề ngoài phạm vi hoạt động của tập đoàn hoặc vi phạm pháp luật, thuần phong mỹ tục.

5. OUTPUT CONTRACT (ReAct Loop):
Mỗi bước suy luận phải tuân thủ nghiêm ngặt format:
Thought: <Suy nghĩ về ý định của khách và quyết định công cụ cần gọi>
Action: <Tên tool cần gọi: search_product_catalog HOẶC submit_support_ticket>
Action Input: <Tham số đầu vào dạng JSON>
Observation: <Kết quả thực tế sau khi thực thi công cụ>
... (Lặp lại nếu cần thêm thông tin)
Final Answer: <Câu trả lời hoàn chỉnh, chính xác, định dạng rõ ràng gửi đến khách hàng>
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling (dễ bị Hallucination)."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """Giả lập phản hồi của LLM thuần khi không có dữ liệu thực tế."""
        simulated_hallucination = (
            f"[Chatbot Baseline - Phản hồi không dùng Tool]\n"
            f"Dạ chào Quý khách! Về yêu cầu '{user_input}': Hiện tại VinFast có rất nhiều mẫu xe "
            f"như VF e34 giá chỉ khoảng 490 triệu, tặng kèm voucher nghỉ dưỡng Vinpearl 50 triệu và miễn phí sạc 10 năm! "
            f"(Lưu ý: Dữ liệu này do mô hình tự suy đoán và có thể không chính xác so với cơ sở dữ liệu thực tế)."
        )
        return {
            "answer": simulated_hallucination,
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent thực thi quy trình suy luận (ReAct) và gọi Tool chính xác."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _parse_price(self, text: str) -> int:
        """Trích xuất giá trị tiền tệ từ ngôn ngữ tự nhiên."""
        match_tr = re.search(r'(\d+[\.,]?\d*)\s*(triệu|tr|trieu)', text, re.IGNORECASE)
        if match_tr:
            val = float(match_tr.group(1).replace(',', '.'))
            return int(val * 1_000_000)

        match_ty = re.search(r'(\d+[\.,]?\d*)\s*(tỷ|ty)', text, re.IGNORECASE)
        if match_ty:
            val = float(match_ty.group(1).replace(',', '.'))
            return int(val * 1_000_000_000)

        match_digits = re.search(r'\b(\d{6,12})\b', text)
        if match_digits:
            return int(match_digits.group(1))

        return 999_999_999_999

    def _extract_customer_name(self, text: str) -> str:
        """Trích xuất tên khách hàng từ câu giới thiệu."""
        name_match = re.search(
            r'(?:tôi tên là|tôi tên|tên tôi là|tên là|tôi là)\s+([^,\.\n]+)',
            text,
            re.IGNORECASE
        )
        if name_match:
            return name_match.group(1).strip()
        return "Khách hàng"

    def _extract_intents(self, user_input: str) -> List[Dict[str, Any]]:
        """TODO 3: Phân tích ý định người dùng bằng quy tắc/regex để sinh kế hoạch gọi Tool."""
        actions = []
        lower_input = user_input.lower()

        # 1. Nhận diện câu hỏi FAQ chung (chính sách, thời hạn bảo hành...) -> Không gọi tool
        is_faq_query = any(
            k in lower_input
            for k in ["chính sách", "bao lâu", "quy định", "như thế nào", "ở đâu"]
        )
        if is_faq_query:
            return []

        # 2. Nhận diện intent báo lỗi / sự cố / khiếu nại cần tạo Ticket
        has_support_keyword = any(
            k in lower_input
            for k in [
                "bị lỗi", "lỗi", "hỏng", "khiếu nại", "phàn nàn",
                "sự cố", "xử lý gấp", "nghiêm trọng", "không nhận"
            ]
        )

        # 3. Nhận diện intent tra cứu sản phẩm / giá bán
        has_price_or_search = any(
            k in lower_input
            for k in ["giá", "dưới", "muốn xem", "tìm", "tư vấn", "có xe", "gói", "nghỉ dưỡng", "du lịch", "vinpearl"]
        )
        has_car_keyword = any(k in lower_input for k in ["xe", "vinfast", "vf", "ô tô", "oto"])
        has_tour_keyword = any(k in lower_input for k in ["du lịch", "vinpearl", "phòng", "tour", "nghỉ dưỡng", "vé", "villa", "nha trang", "phú quốc"])

        # Nếu có từ khóa tìm kiếm sản phẩm (hoặc không phải câu báo lỗi đơn thuần)
        if (has_car_keyword or has_tour_keyword) and (has_price_or_search or not has_support_keyword):
            category = "du_lich" if has_tour_keyword and not has_car_keyword else "xe_dien"
            max_price = self._parse_price(user_input)
            actions.append({
                "tool": "search_product_catalog",
                "args": {"category": category, "max_price": max_price},
                "thought": f"Khách hàng muốn tìm sản phẩm thuộc nhóm '{category}' với ngân sách tối đa {max_price:,} VNĐ."
            })

        # Nếu có từ khóa báo sự cố kỹ thuật / khiếu nại
        if has_support_keyword:
            priority = "high" if any(
                k in lower_input for k in ["gấp", "ngay", "nghiêm trọng", "nguy hiểm", "cháy", "khẩn cấp"]
            ) else "medium"
            customer_name = self._extract_customer_name(user_input)
            actions.append({
                "tool": "submit_support_ticket",
                "args": {
                    "customer_name": customer_name,
                    "issue_description": user_input,
                    "priority": priority
                },
                "thought": f"Khách hàng {customer_name} gặp sự cố cần hỗ trợ với mức độ ưu tiên '{priority}'. Cần tạo support ticket."
            })

        return actions

    def run(self, user_input: str) -> Dict[str, Any]:
        """TODO 4: Vòng lặp Agent Loop (ReAct: Thought -> Action -> Observation -> Final Answer)."""
        self.trace = []
        self.trace.append({
            "step": "init",
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        })

        planned_actions = self._extract_intents(user_input)
        iteration = 0
        observations = []

        # Trường hợp FAQ (không cần gọi tool)
        if not planned_actions:
            self.trace.append({
                "iteration": 1,
                "step": "thought",
                "thought": "Đây là câu hỏi FAQ về chính sách chung, không cần gọi công cụ tra cứu catalog hay tạo ticket."
            })
            final_answer = self._synthesize_final_answer(user_input, observations)
            self.trace.append({
                "step": "final_answer",
                "content": final_answer,
                "total_iterations": 1
            })
            return {
                "answer": final_answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # ReAct Loop khi có Tool cần gọi
        while iteration < len(planned_actions) and iteration < self.max_iterations:
            current_action = planned_actions[iteration]
            tool_name = current_action["tool"]
            tool_args = current_action["args"]
            thought = current_action["thought"]

            # Log Thought & Action
            self.trace.append({
                "iteration": iteration + 1,
                "step": "thought_and_action",
                "thought": thought,
                "action": tool_name,
                "action_input": tool_args
            })

            # Thực thi Tool qua TOOL_MAP
            tool_func = TOOL_MAP.get(tool_name)
            if tool_func:
                try:
                    observation_result = tool_func(**tool_args)
                except Exception as e:
                    observation_result = {"error": f"Lỗi thực thi công cụ {tool_name}: {str(e)}"}
            else:
                observation_result = {"error": f"Không tìm thấy công cụ '{tool_name}'"}

            # Log Observation
            self.trace.append({
                "iteration": iteration + 1,
                "step": "observation",
                "tool": tool_name,
                "action": tool_name,
                "observation": observation_result
            })

            observations.append({"tool": tool_name, "result": observation_result})
            iteration += 1

        # Tổng hợp câu trả lời cuối cùng (Final Answer synthesis)
        final_answer = self._synthesize_final_answer(user_input, observations)

        self.trace.append({
            "step": "final_answer",
            "content": final_answer,
            "total_iterations": iteration
        })

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }

    def _synthesize_final_answer(self, user_input: str, observations: List[Dict[str, Any]]) -> str:
        """Tổng hợp kết quả quan sát thành văn bản theo quy chuẩn VinAssistant."""
        if not observations:
            lower_q = user_input.lower()
            if "bảo hành" in lower_q or "pin" in lower_q:
                return (
                    "Kính chào Quý khách! Chính sách bảo hành pin xe điện VinFast kéo dài từ 7 đến 10 năm "
                    "(hoặc 160.000 km - không giới hạn số km tùy theo dòng xe). Nếu cần tra cứu chi tiết hơn, "
                    "Quý khách vui lòng liên hệ tổng đài 1900 23 23 89 ạ."
                )
            return (
                "Kính chào Quý khách! VinAssistant rất sẵn lòng hỗ trợ thông tin về hệ sinh thái Vingroup. "
                "Quý khách vui lòng cho biết thêm chi tiết nhu cầu cần tư vấn ạ!"
            )

        responses = ["Kính chào Quý khách, VinAssistant xin phép phản hồi thông tin chi tiết như sau:\n"]

        for obs in observations:
            tool_name = obs["tool"]
            result = obs["result"]

            if tool_name == "search_product_catalog":
                if isinstance(result, list) and len(result) > 0 and not result[0].get("error"):
                    responses.append("📌 **DANH SÁCH SẢN PHẨM PHÙ HỢP VỚI YÊU CẦU:**")
                    for idx, item in enumerate(result, 1):
                        name = item.get("name", "N/A")
                        price = item.get("price_vnd", 0)
                        desc = item.get("description", "")
                        responses.append(f"{idx}. **{name}** - Giá: **{price:,.0f} VNĐ**\n   - Chi tiết: {desc}")
                elif isinstance(result, list) and len(result) > 0 and result[0].get("error"):
                    responses.append(f"⚠️ Không thể đọc dữ liệu danh mục: {result[0]['error']}")
                else:
                    responses.append(
                        "Rất tiếc, hiện tại trong danh mục không tìm thấy sản phẩm nào phù hợp với mức giá hoặc điều kiện trên của Quý khách."
                    )

            elif tool_name == "submit_support_ticket":
                t_id = result.get("ticket_id", "N/A")
                c_name = result.get("customer_name", "Quý khách")
                prio = result.get("priority", "medium")
                status = result.get("status", "open")
                responses.append(
                    f"🎫 **ĐÃ TIẾP NHẬN YÊU CẦU HỖ TRỢ THÀNH CÔNG:**\n"
                    f"- Mã Ticket: **{t_id}**\n"
                    f"- Khách hàng: **{c_name}**\n"
                    f"- Mức độ ưu tiên: **{prio}**\n"
                    f"- Trạng thái: **{status}**\n"
                    f"- Đội ngũ kỹ thuật Vingroup sẽ liên hệ xử lý trong thời gian sớm nhất."
                )

        return "\n".join(responses)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nghiệm
# ═══════════════════════════════════════════════════════════════════════════

def main():
    test_queries = [
        "Tôi muốn xem xe điện VinFast giá dưới 600 triệu.",
        "Tôi tên Lê Minh Khoa, xe VF 8 của tôi bị lỗi hệ thống ADAS. Đây là vấn đề nghiêm trọng, cần xử lý gấp."
    ]

    for user_query in test_queries:
        print("\n" + "=" * 70)
        print(f"BÀI TEST VỚI CÂU HỎI: \"{user_query}\"")
        print("=" * 70)

        print("\n--- [1] RUNNING CHATBOT BASELINE ---")
        chatbot = ChatbotBaseline()
        res_base = chatbot.query(user_query)
        print(res_base["answer"])

        print("\n--- [2] RUNNING TOOL CALLING AGENT ---")
        agent = ToolCallingAgent(max_iterations=5)
        res_agent = agent.run(user_query)
        print("Kết quả phản hồi:")
        print(res_agent["answer"])
        print("\nTrace Log (Lịch sử ReAct Loop):")
        print(json.dumps(res_agent["trace"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()