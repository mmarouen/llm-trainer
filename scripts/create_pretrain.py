import os
import shutil
from datetime import date
from dateutil.relativedelta import relativedelta
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import config
from llm_trainer.dataset.pretrain_processor import get_flat_ds, save_shard

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_SAMPLE = None
BATCH_SIZE = 1024
MAX_SEQ_SIZE = 64_000
BLOCK_SIZE = 2_048
OUTPUT_DIR  = "./tmp/shards"
GCS_BUCKET  = "gcp-ml-datastore"
GCS_PREFIX  = "hupd-shards"
PROJECT_ID = 'testcase-488513'

START_DATE = date(2014, 1, 1)
START_DATE = date(2015, 3, 1)
END_DATE = date(2018, 6, 30)
MONTH_SKIPS = 2
CACHE_CLEAR_EVERY = 4

start_date = START_DATE
index = 1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.getenv('hf_token'))
tokenizer.model_max_length = MAX_SEQ_SIZE
while start_date < END_DATE:
    end_date = start_date + relativedelta(months=MONTH_SKIPS)
    print('-------------------------------------')
    print(f'start {start_date} -> {end_date}')
    dataset_dict = load_dataset("HUPD/hupd",
        data_files="https://huggingface.co/datasets/HUPD/hupd/resolve/main/hupd_metadata_2022-02-22.feather", 
        name="all",
        icpr_label=None,
        trust_remote_code=True,
        force_extract=True,
        train_filing_start_date=str(start_date),
        train_filing_end_date=str(end_date),
        val_filing_start_date='2014-01-22',
        val_filing_end_date='2014-02-22',
    )
    flattened_ds = get_flat_ds(dataset_dict['train'], tokenizer=tokenizer, batch_size=BATCH_SIZE, n_samples=MAX_SAMPLE)
    flattened_ds.info.description += f'\nStart date {str(start_date)}, end date {str(end_date)}'
    print(flattened_ds.info.description)
    shard_prefix = f'shard_{start_date}_{end_date}'
    gcp_path = f'gs://{GCS_BUCKET}/{GCS_PREFIX}'
    save_shard(flattened_ds, shard_idx=shard_prefix, output_dir=OUTPUT_DIR, project_id=PROJECT_ID, gcp_folder=gcp_path)
    del dataset_dict, flattened_ds
    if index % CACHE_CLEAR_EVERY == 0:
        cache_dir = config.HF_DATASETS_CACHE
        print(f'Clearing cache at {cache_dir}')
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
    start_date = end_date
    index += 1

if __name__ == '__main__':
    pass
