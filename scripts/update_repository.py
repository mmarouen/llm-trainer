import os
import yaml
import time
import argparse
import subprocess

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--step', type=str, default='domain-adaptation')

args = parser.parse_args()

if __name__ == "__main__":

    with open(os.path.join('config', 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    with open(os.path.join('config', '.env.yaml'), 'r') as file:
        project = yaml.safe_load(file)['project']

    image_config = config['image']
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = f'{image_config["region"]}-docker.pkg.dev/{project["id"]}/{image_config["repository-name"]}/{image_config["image-name"]}:latest'
    config_file = os.path.join(project_root, 'config', 'cloudbuild.yaml')

    start_time = time.time()
    shell_string = f'gcloud builds submit {project_root} \
--substitutions=_TASK={args.step},_TAG={tag},\
_DOCKERFILE={image_config["dockerfile"]},_PROJECT_ID={project["id"]} \
--config {config_file}'

    try:
        subprocess.run(shell_string, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in {__file__}: {e}")
        exit(1)
    time_diff = time.time() - start_time
    print(f'Repository update successful after {time_diff:.4f}')