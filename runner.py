# runner.py

from main import classify_and_get_cost

ticket_text = """
Charlie Davis :I’m scheduled to undergo a minor outpatient procedure next month and would like to confirm whether this is covered under my current Group Health Insurance Plan.
"""

classification, total_cost = classify_and_get_cost(ticket_text)

# Print or process as needed
print("Classification Result:")
print(classification.model_dump_json(indent=2))
print(f"Total Cost: ${total_cost:.6f}")
