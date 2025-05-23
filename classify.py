from intent_prediction2 import classify_ticket
import pandas as pd

def classify(logs):
    labels = []
    for source, log_msg in logs:
        label = classify_log(source, log_msg)
        labels.append(label)
    return labels

def classify_log(source, log_msg):
    label = classify_ticket(log_msg)
    return label.model_dump_json(indent=2)

def classify_csv(input_file):
    df = pd.read_csv(input_file, encoding='ISO-8859-1')

    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

    print(df)

    # Save the modified file
    output_file = "output.csv"
    df.to_csv(output_file, index=False)

    return output_file

if __name__ == '__main__':
    classify_csv("test.csv")