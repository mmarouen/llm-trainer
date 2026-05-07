import re
import os
import numpy as np
import gcsfs
import itertools
from .prompts import MPT_ASSISTANT_PROMPT, MPT_SYS_PROMPT, MPT_USER_PROMPT


def _build_patent_text(batch, all_txt_cols):
    pattern = re.compile(r'^<SOH.+?EOH>')
    texts = []
    for row in zip(*(batch[col] for col in all_txt_cols)):
        parts = []
        for col_name, value in zip(all_txt_cols, row):
            cleaned = pattern.sub('', value.strip())
            parts.append(f"[{col_name.upper()}]\n{cleaned}")
        texts.append("\n".join(parts))
    
    return {"text": texts}

def _build_domain_adaptation_prompt(batch):
    out_prompts = []
    input_prompts = []
    for ipc, title, patent_text in zip(batch['ipc_label'], batch['title'], batch['text']):
        user_prompt = MPT_USER_PROMPT.format(ipc_code=ipc, title=title)
        assistant_prompt = MPT_ASSISTANT_PROMPT.format(patent_text=patent_text)
        input_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{MPT_SYS_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
        )
        generation_prompt = (
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_prompt}<|eot_id|><|end_of_text|>"
        )
        out_prompts.append(generation_prompt)
        input_prompts.append(input_prompt)
    return {
        'inputs': input_prompts,
        'outputs': out_prompts
    }

def get_prompts_ds(dataset, batch_size=512, n_samples=None):
    text_cols = ['abstract', 'claims', 'background', 'summary', 'description']
    to_remove = text_cols + ['decision', 'cpc_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id']
    all_txt_cols = ['title'] + text_cols

    if n_samples:
        dataset = dataset.take(n_samples)
    text_data = dataset.map(
        _build_patent_text,
        fn_kwargs={'all_txt_cols': all_txt_cols},
        remove_columns=to_remove,
        batch_size=batch_size,
        num_proc=4,
        desc='Build patent text',
        batched=True,
    )
    return text_data.map(
        _build_domain_adaptation_prompt,
        remove_columns=['title', 'ipc_label', 'text'],
        batch_size=batch_size,
        desc='Build domain prompt',
        num_proc=4,
        batched=True,
    )

def _tokenize_function(input_prompts, output_prompts, tokenizer):
    src_ids = tokenizer(input_prompts, add_special_tokens=False)["input_ids"]
    tgt_ids = tokenizer(output_prompts, add_special_tokens=False)["input_ids"]
    input_ids = src_ids + tgt_ids
    is_usable = True
    if len(input_ids) > tokenizer.model_max_length:
        is_usable = False
        return 0, 0, 0, is_usable
    labels = [-100] * len(src_ids) + tgt_ids            
    attention_mask = [1] * len(input_ids)
    return input_ids, labels, attention_mask, is_usable

def tokenize_function(data_row, tokenizer):
    all_input_ids = []
    all_labels = []
    all_attention_masks = []
    lengths = []
    for input_prompt, output_prompt in zip(data_row["inputs"], data_row["outputs"]):
        input_ids, labels, attention_mask, usable = _tokenize_function(
            input_prompts=input_prompt,
            output_prompts=output_prompt,
            tokenizer=tokenizer
        )
        if usable:
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)
            lengths.append(len(input_ids))

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        #"attention_mask": all_attention_masks,
        "lengths": lengths
    }

def _chunk_texts(batch: dict, block_size: int) -> dict:
    try:
        concatenated = {
            k: list(itertools.chain.from_iterable(batch[k]))
            for k in batch.keys()
        }
        total = len(concatenated["input_ids"])
        trimmed = (total // block_size) * block_size
        if trimmed == 0:
            return {k: [] for k in concatenated.keys()}

        result = {
            k: [v[i: i + block_size] for i in range(0, trimmed, block_size)]
            for k, v in concatenated.items()
        }
        return result
    except Exception as e:
        print(f"[group_texts] crash: {e}")
        raise

def get_tokenized_ds(dataset, tokenizer, batch_size=512, n_samples=None):
    prompts_ds = get_prompts_ds(dataset=dataset, batch_size=batch_size, n_samples=n_samples)
    tokenized_ds = prompts_ds.map(
        tokenize_function,
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=prompts_ds.column_names,
        batch_size=batch_size,
        desc='Tokenization',
        num_proc=4,
        batched=True,
    )
    total_tokens = sum(tokenized_ds["lengths"])
    tokenized_ds.remove_columns("lengths")
    tokenized_ds.info.description = f'Total tokens {total_tokens:,}'
    return tokenized_ds

def get_chunked_ds(tokenized_ds, batch_size=512, block_size=2048):
    return tokenized_ds.map(
        _chunk_texts,
        fn_kwargs={'block_size': block_size},
        remove_columns=tokenized_ds.column_names,
        batch_size=batch_size,
        desc=f'Grouping into blocks of {block_size} tokens',
        num_proc=10,
        batched=True,
    )

def get_flat_ds(dataset, tokenizer, batch_size=512, n_samples=None):
    def processing_fn(batch):
        concatenated = {}
        for key in ["input_ids", "labels"]:
            concatenated[key] = [item for sublist in batch[key] for item in sublist]
        return concatenated
    tokenized_ds = get_tokenized_ds(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        n_samples=n_samples
        )

    return tokenized_ds.map(
        processing_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=tokenized_ds.column_names,
        num_proc=os.cpu_count() - 2,
        desc="Flattening sequences"
    )

def save_to_gcs(arr, gcs_path, project=None):
    fs = gcsfs.GCSFileSystem(project=project)
    with fs.open(gcs_path, 'wb') as f:
        np.save(f, arr)

def save_shard(
        flat_dataset,
        shard_idx,
        output_dir,
        project_id,
        gcp_folder
        ):

    for col in ["input_ids", "labels"]:
        dtype = np.int32 if col != "attention_mask" else np.uint8
        data = np.array(flat_dataset[col], dtype=dtype)
        save_path = os.path.join(output_dir, f"{shard_idx}_{col}.npy")
        np.save(save_path, data)
        print(f"Data saved to {save_path} | n samples: {data.shape[0]:,}")
        #gcp_path = os.path.join(gcp_folder, f"{shard_idx}_{col}.npy")
        #save_to_gcs(data, gcs_path=gcp_path, project=project_id)
        #print(f'Data saved to {gcp_path} | n samples: ({data.shape[0]:,}, {data.shape[1]})')
        #os.remove()

