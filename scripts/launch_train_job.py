import os
import yaml
import argparse
import subprocess

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--compute', type=str, default='a100')
args = parser.parse_args()

if __name__ == "__main__":

    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    with open(os.path.join('config', '.env.yaml'), 'r') as file:
        project = yaml.safe_load(file)['project']

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    job_config_file_path = os.path.join(project_root, 'config', 'train-job.yaml')
    with open(job_config_file_path, 'r') as file:
        job_config_file = yaml.safe_load(file)
    image_config = config['image']
    job_id = '1st_generation_demo_train'
    machine_spec = job_config_file["workerPoolSpecs"][0]['machineSpec']
    machine_spec = {}
    machine_spec['machineType'] = config['compute'][args.compute]['machine_type']
    machine_spec['acceleratorType'] = config['compute'][args.compute]['accelerator_type'].upper().replace("-", "_")
    machine_spec['acceleratorCount'] = config['compute'][args.compute]['n_accelerators']
    job_config_file["workerPoolSpecs"][0]['machineSpec'] = machine_spec
    wandb_api_key = os.environ.get('WANDB_API_KEY')

    scheduling = {}
    scheduling["strategy"] = "STANDARD" # STANDARD, SPOT
    job_config_file["scheduling"] = scheduling
    env_variables = []
    env_variables.append({'name': 'job_id', 'value': str(job_id)})
    env_variables.append({'name': 'demo_mode', 'value': str(True)})
    env_variables.append({'name': 'micro_batch_size', 'value': str(8)})
    env_variables.append({'name': 'sequence_length', 'value': str(3968)})
    env_variables.append({'name': 'gradient_accumulation_steps', 'value': str(16)})
    env_variables.append({'name': 'WANDB_API_KEY', 'value': wandb_api_key})
    env_variables.append({'name': 'verbose', 'value': str(True)})
    env_variables.append({'name': 'optimizer_precision_bytes', 'value': str(1)})
    job_config_file["workerPoolSpecs"][0]['containerSpec']['env'] = env_variables

    tag = f'{image_config["region"]}-docker.pkg.dev/{project["id"]}/{image_config["repository-name"]}/{image_config["image-name"]}:latest'
    job_config_file["workerPoolSpecs"][0]['containerSpec']['imageUri'] = tag

    with open(job_config_file_path, 'w+') as ff:
        yaml.dump(job_config_file, ff)

    shell_string = f'gcloud ai custom-jobs create \
--region europe-west4 \
--display-name={job_id} \
--config={job_config_file_path}'

    try:
        subprocess.run(shell_string, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in {__file__}: {e}")
        exit(1)
