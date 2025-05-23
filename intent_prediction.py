# --------------------------------------------------------------
# Customer Support Ticket Classification System
# --------------------------------------------------------------

import instructor
from pydantic import BaseModel, Field
from groq import Groq
from enum import Enum
from typing import List
from dotenv import load_dotenv

# Sample customer support tickets
ticket1 = """
I ordered a laptop from your store last week (Order #12345), but I received a tablet instead. 
This is unacceptable! I need the laptop for work urgently. Please resolve this immediately or I'll have to dispute the charge.
"""

ticket2 = """
I visited the hospital on April 5th for an emergency and was told my insurance was inactive.
This is shocking because I’ve been paying my premiums on time every month.
Now I’m being billed $2,300 out of pocket. I need this resolved ASAP — please confirm my coverage status and correct the hospital records.
"""

# Claim Denial
ticket3 = """
My claim for a knee MRI done on March 20th was denied stating it wasn't medically necessary.
However, it was prescribed by my orthopedic specialist. I need a detailed explanation for the denial and how I can appeal this decision.
"""

# Billing Error
ticket4 = """
I was charged twice for the same doctor visit on February 10th. The bill shows two identical charges, but I only had one appointment.
Please correct this billing error and refund the duplicate charge immediately.
"""

# Coverage Inquiry
ticket5 = """
I'm planning to have a minor outpatient surgery next month and need to confirm if it's covered under my current plan.
Can you please send me details of what's included in my benefits and any pre-authorization requirements?
"""

# Dependent Coverage Issue
ticket6 = """
I added my newborn to my policy in January, but the pediatrician’s office says there’s no record of coverage.
I’ve submitted the documents twice already. Can someone please verify the status and ensure my child is covered?
"""


# --------------------------------------------------------------
# Regular Completion using OpenAI (with drawbacks)
# --------------------------------------------------------------
load_dotenv()

client = Groq()

"""
Objective: Develop an AI-powered ticket classification system that:
- Accurately categorizes customer support tickets
- Assesses the urgency and sentiment of each ticket
- Extracts key information for quick resolution
- Provides confidence scores to flag uncertain cases for human review
Business impact:
- Reduce average response time by routing tickets to the right department
- Improve customer satisfaction by prioritizing urgent and negative sentiment tickets
- Increase efficiency by providing agents with key information upfront
- Optimize workforce allocation by automating routine classifications
"""

# --------------------------------------------------------------
# Step 2: Patch your LLM with instructor
# --------------------------------------------------------------

# Instructor makes it easy to get structured data like JSON from LLMs
# Enable instructor patches for Groq client
client = instructor.from_groq(client)

# --------------------------------------------------------------
# Step 3: Define Pydantic data models
# --------------------------------------------------------------

"""
This code defines a structured data model for classifying customer support tickets using Pydantic and Python's Enum class. 
It specifies categories, urgency levels, customer sentiments, and other relevant information as predefined options or constrained fields. 
This structure ensures data consistency, enables automatic validation, and facilitates easy integration with AI models and other parts of a support ticket system.
"""


class TicketCategory(str, Enum):
    ORDER_ISSUE = "claim_denial"
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_INQUIRY = "coverage_inquiry"
    TECHNICAL_SUPPORT = "dependent_coverage_issue"
    BILLING = "billing_issue"
    OTHER = "other"


class CustomerSentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"


class TicketUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketClassification(BaseModel):
    category: TicketCategory
    urgency: TicketUrgency
    sentiment: CustomerSentiment
    confidence: float = Field(ge=0, le=1, description="Confidence score for the classification")
    key_information: List[str] = Field(description="List of key points extracted from the ticket")
    suggested_action: str = Field(description="Brief suggestion for handling the ticket")


ticket_classification = TicketClassification(
    category=TicketCategory.ORDER_ISSUE,
    urgency=TicketUrgency.HIGH,
    sentiment=CustomerSentiment.ANGRY,
    confidence=0.9,
    key_information=["Order #12345", "Received tablet instead of laptop"],
    suggested_action="Contact customer to arrange laptop delivery"
)




# --------------------------------------------------------------
# Step 5: Optimize your prompts and experiment
# --------------------------------------------------------------
# To optimize:
# 1. Refine the system message to provide more context about your business
# 2. Experiment with different models (e.g., gpt-3.5-turbo vs gpt-4)
# 3. Fine-tune the model on your specific ticket data if available
# 4. Adjust the TicketClassification model based on business needs

SYSTEM_PROMPT = """
You are an AI assistant for a large health insurance customer support team. 
Your role is to analyze incoming customer support requests and provide structured information to help our team respond quickly and effectively.
Business Context:
- We handle thousands of requests daily across various categories (claim, accounts, products, technical issues, billing).
- Quick and accurate classification is crucial for customer satisfaction and operational efficiency.
- We prioritize based on urgency and customer sentiment.
Your tasks:
1. Categorize the requests into the most appropriate category.
2. Assess the urgency of the issue (low, medium, high, critical).
3. Determine the customer's sentiment.
4. Extract key information that would be helpful for our support team.
5. Suggest an initial action for handling the ticket.
6. Provide a confidence score for your classification.
Remember:
- Be objective and base your analysis solely on the information provided in the ticket.
- If you're unsure about any aspect, reflect that in your confidence score.
- For 'key_information', extract specific details like order numbers, product names, or account issues.
- The 'suggested_action' should be a brief, actionable step for our support team.
Analyze the following customer support requests and provide the requested information in the specified format.
"""


def classify_ticket(ticket_text: str) -> TicketClassification:
    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        response_model=TicketClassification,
        temperature=0,
        # max_retries=3,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": ticket_text}
        ]
    )
    return response


result1 = classify_ticket(ticket1)
result2 = classify_ticket(ticket2)
result3 = classify_ticket(ticket3)
result4 = classify_ticket(ticket4)
result5 = classify_ticket(ticket5)
result6 = classify_ticket(ticket6)

print(result1.model_dump_json(indent=2))
print(result2.model_dump_json(indent=2))
print(result3.model_dump_json(indent=2))
print(result4.model_dump_json(indent=2))
print(result5.model_dump_json(indent=2))
print(result6.model_dump_json(indent=2))