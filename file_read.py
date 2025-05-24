# import pandas as pd
#
# # df = pd.read_csv('test.csv', encoding='utf-8', errors='ignore')
#
# df = pd.read_csv('test.csv', encoding='ISO-8859-1')
# print(df)

import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count the number of tokens in a given text for a specific model.

    Args:
        text (str): The input text to tokenize.
        model (str): The model name (default is "gpt-3.5-turbo").

    Returns:
        int: The number of tokens in the text.
    """
    # Load the tokenizer for the specified model
    encoding = tiktoken.encoding_for_model(model)
    # Encode the text and count the tokens
    return len(encoding.encode(text))

# Example usage
input_text = "Hello, how can I assist you today?"
output_text = "I can help you with coding, debugging, and more."

input_tokens = count_tokens(input_text)
output_tokens = count_tokens(output_text)

total_tokens = input_tokens + output_tokens

print(f"Input Tokens: {input_tokens}")
print(f"Output Tokens: {output_tokens}")
print(f"Total Tokens: {total_tokens}")
x=2000*10000
print(f"Output Tokens: {x} and cost: {x * 0.60 / 1000000:.8f} $")