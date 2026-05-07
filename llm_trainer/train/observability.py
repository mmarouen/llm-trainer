import os

def get_nccl_cuda_env_variables():
    nccl_env_vars = {}
    for key, value in os.environ.items():
        if 'NCCL' in key or 'CUDA' in key:
            nccl_env_vars[key] = value
    return nccl_env_vars

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def compute_memory(
        hidden_state:int,
        vocab_size: int,
        layers: int,
        precision: float,
        n_params: int=None,
        seq_length: int=None,
        n_heads: int=None,
        batch_size: int=None
    ):
    """Computs the memory requirements to host an llm into memory
    Args:
        hidden_state (int): size of the hidden state
        vocab_size (int): llm vocabulary size
        layers (int): number of layers
        precision (int): floating point precision (fp32: 4., fp16: 2., 4bits: 0.5, ...)
        seq_leength (int): sequence length in tokens
        n_heads (int): number of attention heads
        batch_size (int): batch size in number of samples
    Returns
        n_params (int): number of parameters
        memory_size (float): memory size in Gb
        activation_memory (float): activation_memory size in Gb
    """
    Gb_to_byte = 1_024 ** 3
    FULL_PRECISION = 4.
    # N=h∗v+L∗(12∗h2+13∗h)+2∗h
    if not n_params:
        n_params = hidden_state * vocab_size + layers * (12 * hidden_state**2 + 13 * hidden_state) + 2 * hidden_state
    memory_size = n_params * precision / Gb_to_byte
    activation_memory_gb = None
    if seq_length and n_heads and batch_size:
        # https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=memory_for_activations
        # m_act_full_precision​=L⋅seq⋅bs⋅h⋅(34+h5⋅nheads​⋅seq / h​)
        num_elements = layers * seq_length * batch_size * hidden_state * (34 + 5 * n_heads * seq_length / hidden_state) / FULL_PRECISION
        activation_memory_gb = num_elements * precision / Gb_to_byte
    return n_params, memory_size, activation_memory_gb
