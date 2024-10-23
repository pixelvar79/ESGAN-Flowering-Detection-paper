Example implementation and set up in Windows 11 OS

  https://docs.conda.io/projects/miniconda/en/latest/
  
  miniconda will be enough for creating venv and dependencies
  
  During the installation, you might be prompted to add Anaconda to the system PATH. If not, and if you encounter issues, you can add it         manually:
  
  check "Add Anaconda to my PATH environment variable" during installation.
  
  On Linux you may need to add the following line to your shell profile file (e.g., .bashrc or .zshrc):

Build local Conda virtual environment and dependencies
```
  conda create -n gan python=3.9   

  conda activate gan

  conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

  python -m pip install tensorflow==2.10

  pip install -r requirements.txt       
  
```
This was the last version of Tensorflow in native Windows that supports GPU, otherwise it should be implemented in WSL Linux node or native Linux OS.

# My Project

This is a description on how py files are strcutured for implementing the manuscript analysis.

## Screenshot
![App Screenshot](./workflow.png)
