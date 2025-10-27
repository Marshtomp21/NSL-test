import numpy as np
import torch
import time
import torch.nn as nn
import math
torch.set_printoptions(8)

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]

        Input: Tensor
        Output: Tensor
    """
    return nn.GELU()(x)


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    return nn.functional.softmax(x, dim=-1)


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input:
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])

    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)

    x_norm = (x - mean) / torch.sqrt(var + eps)

    return x_norm * g + b

def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer
        Input:
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']

    return x @ w + b


def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input:
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    return linear(gelu(linear(x, w_b1)), w_b2)


def attention(q, k, v, mask, kv_cache=None):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input:
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    if kv_cache is not None:
        k_cache, v_cache = kv_cache['k'], kv_cache['v']
        k = torch.cat([k_cache, k], dim=0)
        v = torch.cat([v_cache, v], dim=0)
        kv_cache['k'], kv_cache['v'] = k, v

    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    scores = torch.where(mask == 0, -torch.inf, scores)

    probs = softmax(scores)

    return probs @ v

def mha(x, attn, n_head, kv_cache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention

        Input:
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """
    qkv = x.chunk(3, dim=-1) # need to modify

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    n_seq_q = x.size(0)
    n_seq_kv_past = kv_cache[0]['k'].shape[0] if kv_cache is not None else 0
    n_seq_kv = n_seq_q + n_seq_kv_past

    if n_seq_q > 1:
        mask_01 = torch.tril(torch.ones(n_seq_q, n_seq_kv))
    else:
        mask_01 = torch.ones(1, n_seq_kv)

    additive_mask = (1.0 - mask_01) * -torch.inf

    out_heads = [attention(q, k, v, additive_mask, kv_cache=kv_cache[i] if kv_cache is not None else None) for i, (q, k, v) in enumerate(qkv_heads)]  # n_head * [n_seq, n_embd/n_head]
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, dim=-1) # need to modify

    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, block, n_head, kv_cache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']

    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head, kv_cache=kv_cache)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head, kv_cache=None):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings

    n_seq_past = kv_cache[0][0]['k'].shape[0] if kv_cache is not None else 0
    positions = range(n_seq_past, n_seq_past + len(inputs))
    x = wte[inputs] + wpe[positions]

    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for i, block in enumerate(blocks):
        cache = kv_cache[i] if kv_cache else None
        x = transformer_block(x, block, n_head=n_head, kv_cache=cache)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, hparams, n_tokens_to_generate):
    from tqdm import tqdm

    n_layer = hparams['n_layer']
    n_head = hparams['n_head']
    n_embd = hparams['n_embd']

    kv_cache = [[{'k': torch.empty(0, n_embd // n_head), 'v': torch.empty(0, n_embd // n_head)} for _ in range(n_head)] for _ in range(n_layer)]

    next_token = inputs[0]
    prompt_id = inputs[1:]

    for token_id in prompt_id:
        gpt2([next_token], params, n_head, kv_cache=kv_cache)
        next_token = token_id

    generated_ids = []
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2([next_token], params, n_head, kv_cache)
        next_token = np.argmax(logits[-1])
        generated_ids.append(int(next_token))

    return generated_ids

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):

    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding

        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.

    """
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams, n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)