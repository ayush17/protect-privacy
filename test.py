import torch
from transformers import AutoTokenizer, LlamaForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").to(device)
model.eval()

# Example query
sysPrompt = "system You are a helpful AI assistant for paraphrasing sentences. Only provide the paraphrased sentence"
sQuery = "user This is the worst restaurant ever! assistant"
query = sysPrompt + sQuery

# Tokenize query
query_inputs = tokenizer(query, return_tensors='pt').to(device)

# Generate response
with torch.no_grad():
    output = model.generate(
        input_ids=query_inputs['input_ids'],
        attention_mask=query_inputs['attention_mask'],
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        top_k=10
    )

# Decode and print the output
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Response:", response)
