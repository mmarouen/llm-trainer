from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from logging import Logger
import torch
import time
import math
from transformers import get_scheduler
from tqdm.auto import tqdm
import bitsandbytes as bnb
## local imports
from llm_trainer.train import compute_grad_norm

def prepare_training(
        base_model: AutoModelForCausalLM,
        accelerator: Accelerator,
        logger: Logger,
        train_config: dict,
        train_set: DataLoader,
        valid_set: DataLoader = None
        ):

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in base_model.named_parameters() if p.requires_grad and "bias" not in n and "norm" not in n],
            "weight_decay": train_config['weight-decay'],
        },
        {
            "params": [p for n, p in base_model.named_parameters() if p.requires_grad and ("bias" in n or "norm" in n)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = None
    if train_config['optimizer-precision-bytes'] == 1:
        logger.warning('8bit Adam optimizer')
        optimizer = bnb.optim.AdamW8bit(
            optimizer_grouped_parameters,
            lr=train_config['learning-rate']
        )
    else:
        optimizer = bnb.optim.PagedAdamW(
            optimizer_grouped_parameters,
            lr=train_config['learning-rate']
        )

    total_batch_size = accelerator.num_processes * train_config['micro-batch-size'] * train_config['gradient-accumulation-steps']
    total_batch_size_tokens = total_batch_size * train_config['sequence-length']
    total_training_samples_epoch = len(train_set) * accelerator.num_processes * train_config['micro-batch-size']
    total_tokens_per_epoch = total_training_samples_epoch * train_config['sequence-length']
    all_params = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    if valid_set:
        model, optimizer, train_set, valid_set = accelerator.prepare(base_model, optimizer, train_set, valid_set)
    else:
        model, optimizer, train_set = accelerator.prepare(base_model, optimizer, train_set)

    logger.info(f"***** Start training *****")
    logger.info(f"  Num gpus = {accelerator.num_processes}")
    logger.info(f"  Total training samples per epoch = {total_training_samples_epoch:,}")
    logger.info(f"  Total tokens per epoch = {total_tokens_per_epoch:,}")
    logger.info(f"  Total optimization steps per device per epoch = {len(train_set) // train_config['gradient-accumulation-steps']}")
    logger.info(f"  Num Epochs = {train_config['n-epochs']}")
    logger.info(f"  Instantaneous batch size per device = {train_config['micro-batch-size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size} samples --> {total_batch_size_tokens:,} tokens")
    logger.info(f"  Gradient Accumulation steps = {train_config['gradient-accumulation-steps']}")
    logger.info(f"  Model trainable parameters {trainable:,d} || all params: {all_params:,d} || trainable%: {100 * trainable / all_params:.4f}")
    return model, optimizer, train_set, valid_set

def train(
        model: AutoModelForCausalLM,
        accelerator: Accelerator,
        logger: Logger,
        train_config,
        demo_mode: bool,
        verbose: bool,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader=None,
) -> AutoModelForCausalLM:

    model, optimizer, train_set, valid_set = prepare_training(
        base_model=model,
        accelerator=accelerator,
        logger=logger,
        train_config=train_config,
        train_set=train_dataloader,
        valid_set = valid_dataloader)
    n_epochs = train_config['n-epochs']
    
    progress_bar = tqdm(
        range(len(train_set) * n_epochs * accelerator.num_processes),
        disable=not accelerator.is_local_main_process
        )
    train_loss = 0.
    global_step = 0
    steps_acc = 0
    num_optimization_steps_per_epoch = len(train_set) // accelerator.gradient_accumulation_steps
    tokens_per_step = train_config['sequence-length'] * accelerator.num_processes * train_config['micro-batch-size']
    total_tokens_epoch = tokens_per_step * len(train_set)
    total_training_steps = n_epochs * num_optimization_steps_per_epoch
    warmup_steps = int(total_training_steps * train_config.get('warmup-ratio', 0.05))
    update_frequency = int(train_config['log-update-freq-ratio'] * num_optimization_steps_per_epoch) 
    n_steps_demo = int(train_config['n-steps-demo-ratio'] * num_optimization_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=train_config['scheduler-type'],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)
    for epoch in range(n_epochs):
        if verbose:
            logger.info(f"****** epoch {epoch}")
            logger.info(f"Loss update frequency {update_frequency} optimization steps {update_frequency * train_config['gradient-accumulation-steps']} micro steps")
        total_processed_tokens = 0
        accelerator.wait_for_everyone()
        model.train()
        step_start_time = time.time()
        epoch_start_time = time.time()

        for step, batch in enumerate(train_set):
            if demo_mode and global_step > n_steps_demo:
                logger.warning(f'DEMO MODE enabled: Stopping training after {n_steps_demo} steps')
                break
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                train_loss += loss.detach().float()
                steps_acc += 1
                accelerator.backward(loss)
                total_processed_tokens += tokens_per_step
                if verbose and step % 50 == 0 and step > 0:
                    ratio = total_processed_tokens * 100 / total_tokens_epoch
                    time_difference = time.time() - epoch_start_time
                    secs = time_difference % 60
                    hrs = int(time_difference // 3600)
                    mins = int((time_difference % 3600) // 60)
                    logger.info(f"Global step {global_step} / {num_optimization_steps_per_epoch:,}, \
    step {step} / {len(train_set):,}, \
    processed {total_processed_tokens:,} / {total_tokens_epoch:,} tokens ({ratio:.4f} %) \
    elapsed {hrs:02d}:{mins:02d}:{secs:05.2f}")


            if accelerator.sync_gradients:
                #grad_norm = compute_grad_norm(model)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

                if global_step % update_frequency == 0 and global_step > 0:
                    ### logging
                    elapsed = time.time() - step_start_time
                    tokens_per_sec = tokens_per_step * steps_acc / elapsed if elapsed > 0 else 0.0
                    step_start_time = time.time()
                    if accelerator.is_local_main_process:
                        progress_bar.update(update_frequency * accelerator.num_processes)
                        progress_bar.set_description(f"Epoch {epoch}/{n_epochs} train loss: {train_loss:.4f}")
                        print()

                    train_loss = accelerator.reduce(train_loss, reduction="sum")
                    train_loss = (train_loss / steps_acc).item() / accelerator.num_processes
                    perplexity = math.exp(min(train_loss, 20))
                    logger.info(f"  Epoch {epoch}, Step {step}: Ave Train Loss {train_loss:.4f}, global step throughput {tokens_per_sec:,.4f} perplexity {perplexity:.4f}")
                    accelerator.log(
                        {
                            "train/loss":          train_loss,
                            "train/perplexity":    perplexity,
                            #"train/grad_norm":     grad_norm,
                            "train/lr":            optimizer.param_groups[0]["lr"],
                            "perf/tokens_per_sec": tokens_per_sec,
                            "perf/gpu_mem_gb":     torch.cuda.memory_allocated() / 1e9,
                        },
                        step=global_step,
                    )

                    ### reset counters
                    train_loss = 0.
                    steps_acc = 0
    final_model = accelerator.unwrap_model(model)
    accelerator.end_training()
    return final_model