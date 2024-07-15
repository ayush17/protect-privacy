from forward_tracer import ForwardTrace, ForwardTracer
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import gc
import json
import os
import textwrap


from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# from common import setup_env, mk_parser

from ICVLayer import ICVLayer, add_icv_layers, remove_icv_layers

import numpy as np
import argparse
import os
import random

import torch

from utils.logger import tabular_pretty_print, fmt_floa



def tokenize_each_demonstration(demonstration_list, tokenizer, dataset_name=None, prefix = None):
    special_characters = [
        "~", " ~", "~ ", "!", " !", "! ", "@", " @", "@ ", "#", " #", "# ", 
        "$", " $", "$ ", "%", " %", "% ", "^", " ^", "^ ", "&", " &", "& ", 
        "*", " *", "* ", "(", " (", "( ", ")", " )", ") ", "_", " _", "_ ", 
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
    for exp_id in range(len(demonstration_list)):
        if prefix is not None:
            demonstration_list[exp_id] = (prefix[0] + strip_special_characters(demonstration_list[exp_id][0]), prefix[1] + strip_special_characters(demonstration_list[exp_id][1]))
        else:
            demonstration_list[exp_id] = (strip_special_characters(demonstration_list[exp_id][0]), strip_special_characters(demonstration_list[exp_id][1]))
        e_original = tokenizer(demonstration_list[exp_id][0]) 
        e_rewrite = tokenizer(demonstration_list[exp_id][1])
        tokenized_demonstration_list.append((e_original, e_rewrite)) 
    return tokenized_demonstration_list

class Args():
    dataset='demo'
    prompt_version='default'
    exemplar_method='random'
    num_k_shots=1
    model_type='falcon'
    model_size='7b'
    kv_iter= 15
    step_size=0.01
    momentum=0.9
    batch_size=32
    gpus=1
    in_8bit=True
    seed=0
    alpha=1.0
args=Args()

def setup_plain_seed(SEED):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
def setup_gpu(gpu_s):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_s)

def setup_seed(SEED):
    setup_plain_seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def setup_env(gpu_s, seed):
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    setup_gpu(gpu_s)
    setup_seed(seed)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").to(device)
model.eval()
torch.autograd.set_grad_enabled(False)

class RandomState:
    def __init__(self):
        self.random_mod_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_cpu_state = torch.get_rng_state()
        self.torch_gpu_states = [torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]

    def restore(self):
        random.setstate(self.random_mod_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_cpu_state)
        for d, state in enumerate(self.torch_gpu_states):
            torch.cuda.set_rng_state(state, d)

class RandomContext:
    """Save and restore state of PyTorch, NumPy, Python RNGs."""

    def __init__(self, seed=None):
        outside_state = RandomState()

        random.seed(seed)
        np.random.seed(seed)
        if seed is None:
            torch.manual_seed(random.randint(-sys.maxsize - 1, sys.maxsize))
        else:
            torch.manual_seed(seed)
        # torch.cuda.manual_seed_all is called by torch.manual_seed
        self.inside_state = RandomState()

        outside_state.restore()

        self._active = False

    def __enter__(self):
        if self._active:
            raise Exception("RandomContext can be active only once")

        self.outside_state = RandomState()
        self.inside_state.restore()
        self._active = True

    def __exit__(self, exception_type, exception_value, traceback):
        self.inside_state = RandomState()
        self.outside_state.restore()
        self.outside_state = None

        self._active = False

def set_seed(self, seed):
        self._rng_context = RandomContext(seed=seed)
demo = [("Zero stars, I hate it.", "Five stars, I love it."),
                  ("it was terrible !", "it was awesome!"),
                  ("i did nt like it.", "i love it."),
                  ("i would call this the worse denny 's ever ", "i would call this the best denny 's ever "),
                  ("i would recommend find another place. This is the worst place in the world. I absolutely hate this place. Fuck this!", "i would recommend this place again!")]
tokenized_demos = tokenize_each_demonstration(
            demo, tokenizer
            )
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
def obtain_icv(model, inputs, rank=1):
        hidden_states = get_hiddenstates(model, inputs) #each element, layer x len_tokens x dim
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
icv_safety, _ = obtain_icv(
        model, tokenize_each_demonstration(
            demo, tokenizer
            ), rank=1
        )
icv_safety = icv_safety[1:]
icvs_to_shift_safety = [icv_safety]
# Example query
sysPrompt = "system You are a helpful AI assistant for paraphrasing sentences. Only provide the paraphrased sentence"
sQuery = "user This is the worst restaurant ever! assistant"
query = sysPrompt + sQuery



# Tokenize query
query_inputs = tokenizer(query, return_tensors='pt').to(device)
lam = 0.12
add_icv_layers(model, torch.stack([icv_safety],dim=1).cuda(), [lam])
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
