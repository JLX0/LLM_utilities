import subprocess


def get_conda_packages(environment_name) :
    """Get a list of conda packages for the specified environment."""
    bash_script = f"""#!/bin/bash

    unset PYTHONPATH
    eval "$(conda shell.bash hook)"

    conda activate {environment_name}

    conda list | awk '{{if (NR > 3) print $1}}'
    """
    try :
        # Run the Bash script and capture the output
        conda_output = subprocess.check_output(["bash" , "-c" , bash_script] , text=True)
        conda_packages = conda_output.splitlines()
        return conda_packages
    except subprocess.CalledProcessError as e :
        print(f"Error fetching conda packages: {e}")
        return []


def get_pip_packages(environment_name) :
    """Get a list of pip packages for the specified environment."""
    bash_script = f"""#!/bin/bash

    unset PYTHONPATH
    eval "$(conda shell.bash hook)"

    conda activate {environment_name}

    pip list --format=freeze | awk -F '==' '{{print $1}}'
    """
    try :
        # Run the Bash script and capture the output
        pip_output = subprocess.check_output(["bash" , "-c" , bash_script] , text=True)
        pip_packages = pip_output.splitlines()
        return pip_packages
    except subprocess.CalledProcessError as e :
        print(f"Error fetching pip packages: {e}")
        return []


def load_packages(environment_name) :
    """
    Collect conda and pip packages for the specified environment and return them as a dictionary.

    Args:
        environment_name (str): Name of the conda environment.

    Returns:
        dict: A dictionary containing the environment name, conda packages, and pip packages.
    """
    # Collect conda and pip packages
    conda_packages = get_conda_packages(environment_name)
    pip_packages = get_pip_packages(environment_name)

    # Organize into a dictionary
    packages_dict = {
        "environment_name" : environment_name ,  # Add environment_name to the dictionary
        "conda" : conda_packages ,
        "pip" : pip_packages
        }

    return packages_dict