from intent_prediction2 import classify_ticket
from main import classify_and_get_cost
import pandas as pd
from message_router import MessageRouter

def classify(logs):
    labels = []
    processing_cost = []
    label2 = []

    for source, log_msg in logs:
        label,cost = classify_log(source, log_msg)
        labels.append(label)
        processing_cost.append(cost)
        router = MessageRouter(label)
        label2.append(router.display_routing())
    return labels,label2,processing_cost

def classify_log(source, log_msg):
    classification, total_cost = classify_and_get_cost(log_msg)
    label = classification.model_dump_json(indent=2)
    return label,total_cost

def classify_csv(input_file):
    df = pd.read_csv(input_file, encoding='ISO-8859-1')

    # Get labels and routing info
    labels, routing_info ,processing_cost = classify(list(zip(df["source"], df["log_message"])))

    # Assign each list to its own column
    df["target_label"] = labels
    df["routing_info"] = routing_info
    df["processing_cost"] = processing_cost

    print(df)

    # Save the modified file
    output_file = "output.csv"
    df.to_csv(output_file, index=False)

    return output_file

if __name__ == '__main__':
    classify_csv("test.csv")