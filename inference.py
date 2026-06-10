import torch

def prepare_input(cfg:dict, tokenizer, user_query:str, device):
    """
    Args:
        user_query (str): Input text prompt (e.g., "Once upon a time there was a little girl named Lily").

    Returns :
        x (torch.Tensor) : tensor containing the token ids of the user query as well as fully noisy sequence of size [1, context_length]
        update_mask (torch.Tensor) : bool tensor of indices that can be denoised
    ```
    >x

    ```
    """
    MASK_ID = tokenizer.mask_token_id

    context_length = cfg["context_length"]
    # we start with a completely masked sequence
    x = torch.full((1, context_length), MASK_ID, device=device)
    # we then append to it the user query
    prompt_text = f"<|user|>\n{user_query}.\n<|assistant|>\n<|end|>\n"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    length_prompt = len(prompt_ids)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0) # shape [1, length_prompt]

    # we then replace the first tokens of the noisy input with the prompt tokens, this will be tthe input at first step s
    x[:, :length_prompt] = prompt_ids
    
    # flag them as fixed tokens (won't be denoised during the inference)
    fixed_tokens = torch.zeros((1, context_length), dtype=torch.bool, device=device)
    fixed_tokens[:, :length_prompt] = True

    # will give the positions of the tokens to be changed, i.e those outside of the prompt text user
    update_mask = ~fixed_tokens

    return x, update_mask

def inference(model, x:torch.Tensor, update_mask:torch.Tensor, cfg:dict, tokenizer, device):
    model.eval()
    context_length = cfg["context_length"]
    length_prompt  = (~update_mask).int().argmax(dim=1).item()
    MASK_ID = tokenizer.mask_token_id
    for s in range(context_length, 0, -1): # reversed T, T-1, ...., 1, 0
        t = torch.tensor([s],dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x, timesteps=t)

        # add a top_k logic on top of predicted logits
        topk_vals, topk_idx = torch.topk(logits, k=50, dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        logits = filtered

        probs = torch.softmax(logits, dim=-1)   # [1, seq_length, vocab_size] as we don't look only at the last token to predict but the whole seq
        flat = probs.view(-1, probs.size(-1)) # [seq_length, vocab_size]

        sampled = torch.multinomial(flat, num_samples=1)  # (seq_length, 1)
        sampled = sampled.squeeze(-1).unsqueeze(0) # [1, seq_length]

        # get the probas associated with the tokens idx that were sampled
        confids = probs.gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)

        # update only the tokens outside of the prompt user
        x = torch.where(update_mask, sampled, x)
        '''
        at each prediction, we denoise all the masked tokens from the sampled var with the where method

        but we want to iteratively denoise subset of masked tokens and keep some MASKED, until at last step s=1, the tokens
        to be keepen as MASK token are zero : i.e nbr_tokens_kept_masked = 0, so next_ratio is zero

        '''
        next_ratio = (s - 1) / cfg["diffusion_steps"]
        nbr_tokens_kept_masked = int((cfg["context_length"] - length_prompt) * next_ratio)

        # ignore prompt tokens by forcing high prob there
        conf_for_rank = confids.clone()
        conf_for_rank[:, :length_prompt] = float("inf")

        # get the lowest idx token with low probabitlity to remask again
        _, keep_idx = torch.topk(
            conf_for_rank,
            k=nbr_tokens_kept_masked,
            dim=1,
            largest=False
        )
        "important, unlike in MaskGIT, when a token is predicted, it can be masked again  in future steps, so I don't filter them out"
        # replace those token idx with the MASK ID token
        x[0, keep_idx] = MASK_ID

    output_decoded = tokenizer.decode(x[0].tolist())
    return output_decoded