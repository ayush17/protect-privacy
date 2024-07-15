from forward_tracer import ForwardTrace, ForwardTracer
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from sklearn.decomposition import PCA
from ICVLayer import ICVLayer, add_icv_layers, remove_icv_layers

device = "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").to("cuda")
model.eval()

# Example demonstration examples for dialogue safety
demo = [("Zero stars, I hate it.", "Five stars, I love it."),
                  ("it was terrible !", "it was awesome!"),
                  ("i did nt like it.", "i love it."),
                  ("i would call this the worse denny 's ever ", "i would call this the best denny 's ever "),
                  ("i would recommend find another place. This is the worst place in the world. I absolutely hate this place. Fuck this!", "i would recommend this place again!")]

def tokenize_demonstrations(demonstrations, tokenizer):
    return tokenize_each_demonstration(demonstrations, tokenizer)

def tokenize_each_demonstration(demonstration_list, tokenizer, dataset_name=None, prefix=None):
    special_characters = [
        "~", " ~", "~ ", "!", " !", "! ", "@", " @", "@ ", "#", " #", "# ",
        "$", " $", "$ ", "%", " %", "% ", "^", " ^", "^ ", "&", " &", "& ",
        "*", " *", "* ", "(", " (", "( ", ")", " )", ") ", "_", " *", "* ",
        "+", " +", "+ ", "`", " `", "` ", "-", " -", "- ", "=", " =", "= ",
        "{", " {", "{ ", "}", " }", "} ", "[", " [", "[ ", "]", " ]", "] ",
        "|", " |", "| ", "\\", " \\", "\\ ", ":", " :", ": ", ";", " ;", "; ",
        "\"", " \"", "\" ", "'", " '", "' ", "<", " <", "< ", ">", " >", "> ",
        ",", " ,", ", ", ".", " .", ". ", "?", " ?", "? ", "/", " /", "/ "
    ]

    def strip_special_characters(input_string):
        for char in special_characters:
            input_string = input_string.replace(char.strip(), '')
        return input_string.strip()

    tokenized_demonstration_list = []
    for demo in demonstration_list:
        if prefix is not None:
            input_text = prefix[0] + strip_special_characters(demo[0])
            output_text = prefix[1] + strip_special_characters(demo[1])
        else:
            input_text = strip_special_characters(demo[0])
            output_text = strip_special_characters(demo[1])

        input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']
        output_ids = tokenizer(output_text, return_tensors='pt')['input_ids']
        tokenized_demonstration_list.append((input_ids, output_ids))

    return tokenized_demonstration_list



tokenized_demos = tokenize_demonstrations(
            demo, tokenizer
            )

# Updated function to compute In-Context Vectors (ICVs)
# def compute_icv(model, tokenized_demos):
#     icvs = []
#     for input_ids, output_ids in tokenized_demos:
#         with torch.no_grad():
#             # Get hidden states for input and output
#             input_hidden_states = model(input_ids, output_hidden_states=True, return_dict=True)['hidden_states']
#             output_hidden_states = model(output_ids, output_hidden_states=True, return_dict=True)['hidden_states']
            
#             # Ensure the hidden states have the same length
#             min_len = min(input_hidden_states[-1].size(1), output_hidden_states[-1].size(1))
#             aligned_input_hidden = input_hidden_states[-1][:, :min_len, :]
#             aligned_output_hidden = output_hidden_states[-1][:, :min_len, :]
            
#             # Compute ICV (simple difference for demonstration purposes)
#             icv = aligned_output_hidden - aligned_input_hidden
#             icvs.append(icv.mean(dim=1))  # Averaging across tokens
#     return torch.stack(icvs)

def get_hiddenstates(model, inputs):
        h_all = []

        for example_id in range(len(inputs)):
            embeddings_for_all_styles= []
            for style_id in range(len(inputs[example_id])):
                forward_trace = ForwardTrace()
                context_manager = ForwardTracer(model, forward_trace)
                with context_manager:
                    _ = model(
                    input_ids=torch.tensor(inputs[example_id][style_id]['input_ids']).unsqueeze(0).cuda(), 
                    attention_mask = torch.tensor(inputs[example_id][style_id]['attention_mask']).unsqueeze(0).cuda(), 
                    output_attentions=False,
                    output_hidden_states=False
                    )
                    h = forward_trace.residual_stream.hidden
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,-1])
                embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))
        return h_all


def compute_icv(model, tokenized_demos, rank=1):
    hidden_states = get_hiddenstates(model, tokenized_demos) #each element, layer x len_tokens x dim
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []

    hidden_states_all = []

    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)
    neg_emb = torch.stack(neg_all).mean(0)
    pos_emb = torch.stack(pos_all).mean(0)

    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data) 
    direction = (pca.components_.sum(dim=0,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))#h_pca.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, (neg_emb).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
icvs, neg_emb = compute_icv(model, tokenized_demos)

# Updated hook function to handle tuple output
# def apply_icvs(model, icvs, lam=0.1):
#     def hook(module, input, output):
#         if isinstance(output, tuple):
#             output_tensor = output[0]  # Extract the tensor part of the tuple
#             modified_output = output_tensor + lam * icvs.mean(dim=0)
#             return (modified_output,) + output[1:]  # Repack the tuple
#         else:
#             return output + lam * icvs.mean(dim=0)
    
#     # Add hook to model layers (assuming a simple transformer structure)
#     for layer in model.model.layers:
#         layer.register_forward_hook(hook)

# Example query
sysPrompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a helpful AI assistant for paraphrasing sentences. Only provide the paraphrased sentence<|eot_id|>"

sQuery = "<|start_header_id|>user<|end_header_id|>This is the worst!<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

query = sysPrompt+sQuery
# Tokenize query
query_inputs = tokenizer(query, return_tensors='pt').to("cuda")

# Generate response
with torch.no_grad():
    output = model.generate(
        # input_ids=torch.tensor(query_inputs['input_ids']).unsqueeze(0).cuda(),
        # attention_mask=torch.tensor(query_inputs['attention_mask']).unsqueeze(0).cuda(),
        input_ids=query_inputs['input_ids'],
        attention_mask=query_inputs['attention_mask'],
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        top_k=10
    )

# Decode and print the output
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Original Response:", response)

# apply_icvs(model, icvs)
alpha = [1] * len(model.model.layers)  # Assuming a uniform alpha for simplicity
add_icv_layers(model, icvs, alpha)

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
