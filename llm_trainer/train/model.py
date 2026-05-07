import os
import gcsfs
import torch
import tempfile
from logging import Logger
from transformers import AutoModelForCausalLM


def load_model(
    gcs_bucket: str,
    gcp_path: str,
    logger: Logger,
    compile_mode: str = "max-autotune-no-cudagraphs",
    torch_compile=True
) -> AutoModelForCausalLM:

    fs = gcsfs.GCSFileSystem()
    local_dir = tempfile.mkdtemp()
    gcs_url = f"{gcs_bucket}/{gcp_path}"
    fs.get(gcs_url, local_dir, recursive=True)
    logger.info(f'Copied LLM to {local_dir}: {os.listdir(local_dir)}')
    model = AutoModelForCausalLM.from_pretrained(
        f"{local_dir}/{gcp_path}",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    logger.info('Model loaded')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "determinism_check": "none",
        }
    )
    if torch_compile:
        model = torch.compile(model, mode=compile_mode)
    return model

def save_model(model):
    pass

