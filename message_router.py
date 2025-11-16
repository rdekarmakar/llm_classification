"""
Message routing based on classification results.
"""

import json
import logging
from models import TicketCategory

logger = logging.getLogger(__name__)


class MessageRouter:
    """Routes classified messages to appropriate teams based on category and urgency."""
    
    CATEGORY_ROUTING = {
        TicketCategory.COVERAGE_INQUIRY.value: "Product Support Team",
        TicketCategory.CLAIM_DENIAL.value: "Claims Department",
        "claim_status": "Claims Department",  # Legacy support
        TicketCategory.BILLING_ISSUE.value: "Billing Team",
        TicketCategory.DEPENDENT_COVERAGE_ISSUE.value: "IT Support",
        "technical_issue": "IT Support",  # Legacy support
        TicketCategory.ACCOUNT_ACCESS.value: "Account Management Team",
        TicketCategory.OTHER.value: "Customer Service",
        "general_question": "Customer Service",  # Legacy support
    }

    URGENCY_ESCALATION = {
        "low": "Standard Queue",
        "medium": "Priority Queue",
        "high": "Escalation Team",
        "critical": "Immediate Attention Desk"
    }

    def __init__(self, raw_message: str):
        self.raw_message = raw_message
        self.message = self._parse_json(raw_message)

    def _parse_json(self, raw_message: str) -> dict:
        """Parse JSON message string."""
        try:
            return json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message router: {e}")
            return {}

    def route(self) -> dict:
        category = self.message.get("category", "general_question")
        urgency = self.message.get("urgency", "low")

        assigned_team = self.CATEGORY_ROUTING.get(category, "Customer Service")
        urgency_level = self.URGENCY_ESCALATION.get(urgency, "Standard Queue")

        return {
            "assigned_team": assigned_team,
            "urgency_level": urgency_level
        }

    def display_routing(self):
        routing_info = self.route()
        return f"📬 Routed to: {routing_info['assigned_team']} ⏱ Urgency Level: {routing_info['urgency_level']}"


# Example usage
if __name__ == "__main__":
    result = """
    {
      "category": "coverage_inquiry",
      "urgency": "low",
      "sentiment": "neutral",
      "confidence": 0.95,
      "key_information": [
        "Policy Number: HI123456789",
        "Policyholder Name: John Doe",
        "Inquiry about free annual health check-up eligibility and booking process"
      ],
      "suggested_action": "Verify eligibility for the annual health check-up and provide details on how to book it along with a list of network hospitals or clinics."
    }
    """

    router = MessageRouter(result)
    print(router.display_routing())
