import torch
import torch.nn.functional as F
import itertools
from tqdm.auto import tqdm
from accelerate import Accelerator

@torch.no_grad()
def corrupt_with_mask(input_ids, attention_mask, t, mask_token_id, T, BOS_ID, EOS_ID, PAD_ID):
    """
    takes as input tensor of input_ids, and then return from it a noisy version by replacing some tokes with the MASK_ID, as well as the targets which is a tensor with ids of original tokens
    """
    batch_size, _ = input_ids.shape

    p = t / T # [batch_size, ]
    p = p.view(batch_size, 1) #[batch_size, 1]
    valid = attention_mask & (input_ids != BOS_ID) & (input_ids != EOS_ID) & (input_ids != PAD_ID)

    # Bernoulli sampling
    mask_positions = torch.bernoulli(p.expand_as(input_ids)).bool()
    mask_positions &= valid

    noisy = input_ids.clone()
    noisy[mask_positions] = mask_token_id

    targets = input_ids.clone()
    targets[~mask_positions] = -100

    return noisy, targets

def diffusion_loss(model, batch, T, MASK_ID):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_size = input_ids.size(0)
    t = torch.randint(1, T + 1, (batch_size,), device=input_ids.device)

    noisy_ids, targets = corrupt_with_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        t=t,
        mask_token_id=MASK_ID,
        T=T,
    )

    logits = model(noisy_ids, timesteps=t, attention_mask=attention_mask)  # [batch_size, seq_length, vocab_size]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), # will have [batch_size * seq_length, vocab_size]
        targets.view(-1), # to have shape [batch_size * seq_length, ]
        ignore_index=-100, # computes the loss only on the masked tokens
    )
    return loss

def train_model(model, cfg:dict, train_loader, train_steps,):

    accelerator = Accelerator(
    mixed_precision="bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16")
    device = accelerator.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    # without scheduler
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )

    GRAD_ACCUM = 1
    model.train()

    train_iter = itertools.cycle(train_loader) # we use cycle method to go over the iterator indefinitly
    running = []
    val_history = []
    T = cfg["diffusion_steps"]
    for step in tqdm(range(train_steps), disable=not accelerator.is_main_process):

        batch = next(train_iter)

        loss = diffusion_loss(model, batch, T=T)
        accelerator.backward(loss / GRAD_ACCUM)

        if (step + 1) % GRAD_ACCUM == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler update
            # scheduler.step()

            optimizer.zero_grad()

        running.append(loss.item())

        # if accelerator.is_main_process and step % 50 == 0:
        #     accelerator.print(
        #         f"step {step} | loss {sum(running[-50:]) / len(running[-50:]):.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        #     )

        if step % 500 == 0:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    # only run validation on the first 10 batches
                    if i == 40:
                        break
                    loss = diffusion_loss(model, batch, T=cfg["diffusion_steps"])
                    val_losses.append(accelerator.gather_for_metrics(loss).mean().item())

            mean_val_loss = sum(val_losses) / len(val_losses)
            val_history.append(mean_val_loss)
            accelerator.print(f"step {step} | val_loss {mean_val_loss:.4f}")
            model.train()
        # savie model checkpoints every 5000 steps
        # if accelerator.is_main_process and step > 0 and step % 5000 == 0:
        #       checkpoint = {
        #           "model": accelerator.unwrap_model(model).state_dict(),
        #           "optimizer": optimizer.state_dict(),
        #           "scheduler": scheduler.state_dict(),
        #           "step": step,
        #           "cfg": cfg,
        #       }

        #       torch.save(
        #           checkpoint,
        #           os.path.join("checkpoints", f"checkpoint_{step}.pt")
        #       )

        #       accelerator.print(f"Saved checkpoint at step {step}")