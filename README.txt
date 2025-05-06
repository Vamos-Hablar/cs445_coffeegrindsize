To set up your environment, I suggest the following:

# 1. (assuming you have conda installed) Create fresh Conda environment with the appropriate python version

conda create -n coffee_env python=3.7.12 -c conda-forge
conda activate coffee_env

# 2. Activate venv
.\project_venv\Scripts\activate

# 3. Adding additional packages: if/when you pip install new packages for your work, be sure to update requirements.txt accordingly and include changes to the project_venv folder in your commit

You can then just run the automated analysis version via
python auto_analyzer.py
