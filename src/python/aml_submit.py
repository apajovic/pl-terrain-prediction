# aml_submit.py
# Script to submit training jobs to Azure ML (AML)
import argparse
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment

def submit_to_aml(config_path, script_path):
    ws = Workspace.from_config()
    env = Environment.from_conda_specification(name='project-env', file_path='environment.yml')
    src = ScriptRunConfig(source_directory='.', script=script_path, arguments=['--config', config_path], environment=env)
    exp = Experiment(workspace=ws, name='model-training')
    run = exp.submit(src)
    print(f'Submitted to AML. Run ID: {run.id}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--script', type=str, default='train.py')
    args = parser.parse_args()
    submit_to_aml(args.config, args.script)

if __name__ == '__main__':
    main()
