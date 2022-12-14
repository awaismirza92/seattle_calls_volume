# Prediction of Call Volume for Seattle Fire Department

The repository contains a notebook named `call_volume.ipynb` for predicting number of calls arriving at Seattle Fire Department in a given hour at a given date. The notebook contains links to datasets which have been employed, data exploration plots, features extraction section, modeling fitting and evaluation sections.


## Installation

The notebook has been developed on Google Colab, but it should run on a local environment with the following installation steps:

1. Install Anaconda distribution of Python from here: https://www.anaconda.com/products/distribution

2. Create a [new environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) having python 3.8:
`conda create -n seattle python=3.8`

3. Activate the environment:
`conda activate seattle`

4. Clone the repository:
`git clone https://github.com/awaismirza92/seattle_calls_volume.git`

5. Change directory: `cd seattle_calls_volume`

6. Install the required packages:
`pip install -r requirements.txt`

7. Open the notebook: `jupyter-lab call_volume.ipynb`

## Script file
The associated srcipt file (`call_volume.py`) containing essentially the same code as in the notebook is present in `script` folder. If run in the local environment, the notebook downloads the datasets in the repo's home folder. For this reason, the script should be run from inside the `script` folder, so that it utilizes the datasets already downloaded (if notebook was run before) in the home folder. Otherwise, it would download new copies of datasets in the folder one level above from where it is run.

## Plots
Two plots `actual_vs_predicted_week.png` and `actual_vs_predicted_month.png` are the outcome of the model fitting.
