# --------------------------------------------------------------
# Customer Support Ticket Classification System
# --------------------------------------------------------------

import instructor
from pydantic import BaseModel, Field
from groq import Groq
from enum import Enum
from typing import List
from dotenv import load_dotenv
import chromadb

# Initialize ChromaDB client
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(path="my_vectordb")

collection_customer_interaction = chroma_client.get_or_create_collection(name="customer_interaction")
collection_customer_policies = chroma_client.get_or_create_collection(name="customer_policies")


# Dependent Coverage Issue
ticket7 = """
Hi Team,

I wanted to check if I'm eligible for the free annual health check-up mentioned in the policy.

Could you please let me know how to book it and if there’s a specific hospital or clinic I need to visit?

Thanks in advance for your help!

Best,
John Doe
"""

ticket8 = """
Hi Team,
I have not received any response to my previous email regarding the free annual health check-up mentioned in the policy.
I am really pissed off with the lack of communication from your side.
Can someone please get back to me with the details asap.

Thanks,
John Doe
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
- For 'key_information', extract specific details like order numbers, product names,account issues or brief from previous customer interactions.
- The 'suggested_action' should be a brief, actionable step for our support team.
Analyze the following customer support requests and provide the requested information in the specified format.
As additional context, you can use the customer interaction history and customer policies.
"""


def classify_ticket(ticket_text: str) -> TicketClassification:
    # Query ChromaDB for additional context
    results = collection_customer_interaction.query(
        query_texts=[ticket_text],
        n_results=1  # Retrieve top 3 relevant results
    )
    customer_interaction_context = " ".join([doc for sublist in results["documents"] for doc in sublist])

    print("customer_interaction_context",customer_interaction_context)

    results = collection_customer_policies.query(
        query_texts=[ticket_text],
        n_results=1  # Retrieve top 3 relevant results
    )
    customer_policies_context = " ".join([doc for sublist in results["documents"] for doc in sublist])

    print("customer_policies_context", customer_policies_context)

    additional_context = customer_interaction_context + customer_policies_context

    # Combine ticket text with additional context
    combined_input = f"{ticket_text}\n\nAdditional Context:\n{additional_context}"

    # Pass combined input to the classification model
    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        response_model=TicketClassification,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": combined_input}
        ]
    )
    return response


# result1 = classify_ticket(ticket1)
# result2 = classify_ticket(ticket2)
# result3 = classify_ticket(ticket3)
# result4 = classify_ticket(ticket4)
# result5 = classify_ticket(ticket5)
# result6 = classify_ticket(ticket6)
result7 = classify_ticket(ticket7)
# result8 = classify_ticket(ticket8)

# print(result1.model_dump_json(indent=2))
# print(result2.model_dump_json(indent=2))
# print(result3.model_dump_json(indent=2))
# print(result4.model_dump_json(indent=2))
# print(result5.model_dump_json(indent=2))
# print(result6.model_dump_json(indent=2))
print(result7.model_dump_json(indent=2))
# print(result8.model_dump_json(indent=2))

# if __name__ == "__main__":
#     print(classify_ticket(
#         "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
#     print(classify_ticket(
#         "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
#     print(classify_ticket("System reboot initiated by user 12345."))