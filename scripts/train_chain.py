
import os
os.environ["NCCL_DEBUG"] = "WARN"
import sys
import yaml
import argparse
import logging
import torch
import transformers
import shutil
import json
import accelerate
from accelerate import Accelerator
import datetime
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
## local imports
from llm_trainer.train import get_nccl_cuda_env_variables, load_model, save_model, train
from llm_trainer.dataset import build_pretrain_dataloader

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False

parser.add_argument(
    "--train-step", type=str, default='domain-adaptation', help='Current training phase'
)
parser.add_argument(
    "--trace-run", type=str2bool, default=False, help='Triggers a tensorboard logger'
)
args = parser.parse_args()

## env variables
DEMO_MODE = os.environ.get("demo_mode", "TRUE").lower() == "true"
SEQUENCE_LENGTH = int(os.environ.get("sequence_length", "0"))
UPDATE_FREQ = float(os.environ.get("update_freq", "0"))
MICRO_BATCH_SIZE = int(os.environ.get("micro_batch_size", "0"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("gradient_accumulation_steps", "0"))
VERBOSITY = os.environ.get("verbose", "TRUE").lower() == "true"
OPTIMIZER_PRECISION = int(os.environ.get("optimizer_precision_bytes", "4"))

def main():


    config = {}
    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    # loggers
    transformers.utils.logging.set_verbosity_info()
    log_level = 'INFO'
    logger.setLevel(log_level)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('torch.distributed').setLevel(logging.WARNING)

    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
        )

    model_config = config['model']['input'] if args.train_step == 'domain-adaptation' else config['model']['pretrain']
    MODEL_BUCKET = model_config['gcp-bucket']
    MODEL_RELATIVE_PATH = model_config['relative-path']
    DATA_BUCKET = config['data']['bucket']
    DATA_RELATIVE_PATH = config['data']['pretrain-path'] if args.train_step == 'domain-adaptation' else config['data']['sft-path-train']
    DATA_EVAL_RELATIVE_PATH = None if args.train_step == 'domain-adaptation' else config['data']['sft-path-eval']
    # train config
    train_config = config['pretrain'] if args.train_step == 'domain-adaptation' else config['sft'] 

    ### update config
    if SEQUENCE_LENGTH:
        train_config['sequence-length'] = SEQUENCE_LENGTH
    if UPDATE_FREQ:
        train_config['log-update-freq-ratio'] = UPDATE_FREQ
    if MICRO_BATCH_SIZE:
        train_config['micro-batch-size'] = MICRO_BATCH_SIZE
    if GRADIENT_ACCUMULATION_STEPS:
        train_config['gradient-accumulation-steps'] = GRADIENT_ACCUMULATION_STEPS
    if OPTIMIZER_PRECISION:
        train_config['optimizer-precision-bytes'] = OPTIMIZER_PRECISION

    ### log job params
    sw_versions = f'torch: {torch.__version__}, accelerate {accelerate.__version__}, transformers {transformers.__version__}'
    logger.info(f'Software package versions:\n{sw_versions}')
    args_string = ' '.join(f'{k}={v}\n' for k, v in vars(args).items())
    logger.info(f'Input arguments:\n{args_string}')
    train_config_string = ' '.join(f'{k}={v}\n' for k, v in train_config.items())
    logger.info(f"Training configuration:\n{train_config_string}")

    ### init runtime
    local_output_dir = '/tmp/model_results'
    os.makedirs(local_output_dir, exist_ok=True)
    logger.info(f'Temp output folder :\n{local_output_dir}')
    use_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if use_ddp:
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logging_dir_ = './logs/profiler'
    os.makedirs(logging_dir_, exist_ok=True)
    tb_dir_ = './logs/tb_accelerate'
    os.makedirs(tb_dir_, exist_ok=True)

    accelerate_config = {
        'gradient_accumulation_steps': train_config['gradient-accumulation-steps'],
        'mixed_precision': "bf16" if torch.cuda.is_bf16_supported() else "fp16",
        'log_with': "wandb",
        #'project_dir': tb_dir_
    }

    profiler = None
    if args.trace_run:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6),
            on_trace_ready=tensorboard_trace_handler(logging_dir_),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            #with_flops=True,
        )
        profiler.start()

    nccl_envs = get_nccl_cuda_env_variables()
    if nccl_envs:
        logger.info("Defined NCCL/CUDA Environment Variables:")
        for key, value in nccl_envs.items():
            print(f"- {key}: {value}")
    else:
        logger.info("No NCCL/CUDA environment variables found.")

    accelerator = Accelerator(**accelerate_config)
    accelerator.init_trackers(
        project_name="llama3.2-3B domain adaptation",
        config=train_config,
    )
    train_dataloader = None
    eval_dataloader = None
    if args.train_step == 'domain-adaptation':
        train_dataloader = build_pretrain_dataloader(
            bucket=DATA_BUCKET,
            gcp_prefix=DATA_RELATIVE_PATH,
            logger=logger,
            chunk_size=train_config['sequence-length'],
            batch_size=train_config['micro-batch-size'],
            demo_mode=DEMO_MODE
        )
    elif args.train_step == 'sft':
        pass
    elif args.train_step == 'rlhf':
        pass
    else:
        raise ValueError("No such training mode")
    model = load_model(
        gcp_path=MODEL_RELATIVE_PATH,
        gcs_bucket=MODEL_BUCKET,
        logger=logger
    )
    model = train(
        model=model,
        accelerator=accelerator,
        logger=logger,
        train_config=train_config,
        demo_mode=DEMO_MODE,
        verbose=VERBOSITY,
        train_dataloader=train_dataloader,
        valid_dataloader=eval_dataloader
    )

    save_model(model)

if __name__ == "__main__":
    main()