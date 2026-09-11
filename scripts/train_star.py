import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from star.utils.distributed import require_single_node


def _close_training_resources(order_sampler, accelerator, *, completed):
    if order_sampler is not None:
        order_sampler.close()
    if completed or accelerator.num_processes == 1:
        accelerator.end_training()


def _save_training_config(config, config_path, logger):
    from omegaconf import OmegaConf

    if config_path.exists():
        logger.info(f"Preserving existing config at {config_path}")
    else:
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")


def main():
    require_single_node()
    # Keep third-party imports inside the guarded entrypoint. Spawned UST
    # workers re-import this file as ``__mp_main__`` and must not initialize
    # Torch, TensorFlow, or CUDA before running the pure-Python worker target.
    import torch
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import set_seed
    from star.sampling.orders import ParallelOrderSampler
    from star.utils.logger import setup_logger
    from star.utils.training_assets import (
        DEFAULT_PRETOKENIZATION_FILENAME,
        DEFAULT_PRETOKENIZATION_REPO_ID,
        DEFAULT_TOKENIZER_FILENAME,
        DEFAULT_TOKENIZER_REPO_ID,
        resolve_training_asset,
    )
    from star.utils.train_utils import (
        ExactResumeState,
        auto_resume,
        build_exact_resume_manifest,
        create_dataloader,
        create_lr_scheduler,
        create_model_and_loss_module,
        create_optimizer,
        create_pretrained_tokenizer,
        get_config,
        save_checkpoint,
        set_training_dataloader_epoch,
        training_target_reached,
        train_one_epoch_generator,
    )

    config = get_config()
    data_seed = config.training.get("seed")
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    tracker = "wandb" if config.training.enable_wandb else None

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        dataloader_config=DataLoaderConfiguration(
            split_batches=False,
            use_seedable_sampler=True,
            data_seed=data_seed,
        ),
    )

    logger = setup_logger(
        name="STAR",
        log_level="INFO",
        output_file=f"{output_dir}/log{accelerator.process_index}.txt",
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        if config.training.enable_wandb:
            accelerator.init_trackers(
                config.experiment.project,
                init_kwargs={"wandb": {"name": config.experiment.name}},
            )
        config_path = Path(output_dir) / "config.yaml"
        _save_training_config(config, config_path, logger)

    # If passed along, set the training seed now.
    set_seed(data_seed, device_specific=True)

    config.model.vq_model.tokenizer_checkpoint = resolve_training_asset(
        config.model.vq_model.get("tokenizer_checkpoint"),
        repo_id=DEFAULT_TOKENIZER_REPO_ID,
        filename=DEFAULT_TOKENIZER_FILENAME,
        accelerator=accelerator,
    )
    config.dataset.params.pretokenized_jsonl = resolve_training_asset(
        config.dataset.params.get("pretokenized_jsonl"),
        repo_id=DEFAULT_PRETOKENIZATION_REPO_ID,
        filename=DEFAULT_PRETOKENIZATION_FILENAME,
        accelerator=accelerator,
    )

    # Get the MaskGIT-VQ tokenizer.
    tokenizer = create_pretrained_tokenizer(config, accelerator)

    model, loss_module = create_model_and_loss_module(config, logger, accelerator)

    optimizer = create_optimizer(config, logger, model)

    lr_scheduler = create_lr_scheduler(config, logger, accelerator, optimizer)

    train_dataloader = create_dataloader(config, logger)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )
    num_batches_per_epoch = len(train_dataloader)
    if num_batches_per_epoch == 0:
        raise ValueError(
            "The training dataloader is empty. Provide at least one full "
            "per-rank batch or reduce training.per_process_batch_size."
        )
    # Use the prepared loader's real rank-local length. The configured example
    # count can differ by one batch after drop-last and distributed sharding.
    num_update_steps_per_epoch = math.ceil(
        num_batches_per_epoch / config.training.gradient_accumulation_steps
    )

    # Afterwards we recalculate our number of training epochs.
    # Training is step-based; epochs are used for bookkeeping and to reuse the
    # same training loop with other datasets and loaders.
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(
        "  Instantaneous batch size per process = "
        f"{config.training.per_process_batch_size}"
    )
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_process_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    resume_manifest = build_exact_resume_manifest(
        config,
        accelerator,
        train_dataloader,
        data_seed=data_seed,
    )
    training_state = ExactResumeState(resume_manifest)
    accelerator.register_for_checkpointing(training_state)

    global_step, first_epoch, first_batch = auto_resume(
        config,
        logger,
        accelerator,
        training_state,
        num_batches_per_epoch,
    )

    skip_training = training_target_reached(
        global_step,
        config.training.max_train_steps,
        logger,
        accelerator,
    )
    order_sampler = None
    training_completed = False
    try:
        if not skip_training:
            num_order_workers = config.training.get("num_order_workers", 0)
            logger.info(
                "Creating rank-local UST sampler with "
                f"{num_order_workers} worker process(es)."
            )
            order_sampler = ParallelOrderSampler(
                num_workers=num_order_workers,
                base_seed=data_seed,
                stream_id=accelerator.process_index,
            )
            for current_epoch in range(first_epoch, num_train_epochs):
                accelerator.print(
                    f"Epoch {current_epoch}/{num_train_epochs-1} started."
                )
                set_training_dataloader_epoch(
                    train_dataloader,
                    current_epoch,
                    data_seed,
                )
                epoch_dataloader = train_dataloader
                if current_epoch == first_epoch and first_batch:
                    epoch_dataloader = accelerator.skip_first_batches(
                        train_dataloader,
                        first_batch,
                    )
                    # skip_first_batches creates a new DataLoaderShard whose
                    # default iteration is zero, so set its epoch explicitly too.
                    set_training_dataloader_epoch(
                        epoch_dataloader,
                        current_epoch,
                        data_seed,
                    )
                global_step = train_one_epoch_generator(
                    config,
                    logger,
                    accelerator,
                    model,
                    loss_module,
                    optimizer,
                    lr_scheduler,
                    epoch_dataloader,
                    tokenizer,
                    global_step,
                    order_sampler=order_sampler,
                    training_state=training_state,
                )
                first_batch = 0
                # Stop training if max steps is reached.
                if global_step >= config.training.max_train_steps:
                    accelerator.print(
                        "Finishing training: Global step is >= Max train steps: "
                        f"{global_step} >= {config.training.max_train_steps}"
                    )
                    break

        accelerator.wait_for_everyone()
        # Save checkpoint at the end of training.
        save_checkpoint(
            model,
            output_dir,
            accelerator,
            global_step,
            logger=logger,
            resume_state=training_state,
        )
        # Save the final trained checkpoint
        if accelerator.is_main_process:
            model = accelerator.unwrap_model(model)
            model.save_pretrained_weights(output_dir)
        training_completed = True
    finally:
        _close_training_resources(
            order_sampler,
            accelerator,
            completed=training_completed,
        )


if __name__ == "__main__":
    main()
