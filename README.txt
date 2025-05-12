https://github.com/Vamos-Hablar/cs445_coffeegrindsize/tree/master

To set up your environment, I suggest the following:

# 1. (assuming you have conda installed) Create fresh Conda environment with the appropriate python version
>conda create -n coffee_env python=3.7.12 -c conda-forge
>conda activate coffee_env

# 2. update director to project folder
>cd cs445_coffeegrindsize

>>>(DISREGARD VENV INSTRUCTIONS, THIS WAS NOT PORTABLE TO DIFFERENT MACHINES)<<<
# 3. from your cloned repo folder, pip install the requirements.txt folder to your new conda environment 
 .\cs445_coffeegrindsize> pip install -r requirements.txt


# 4. Running the project
>python auto_analyze.py

# 5. Adding additional packages: if/when you pip install new packages to the venv for your work, be sure to update requirements.txt accordingly and include changes to the project_venv folder in your commit
