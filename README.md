# Turboshaft-Engine-Fault-Detection

This project was developed as a team in collaboration with Eneko Barbadillo and Nora Ibarguren during the *"Data Analytics for Industry"* subject. 

The project focuses on **analyzing and modeling helicopter engine data to enable early fault detection in key components**. Helicopter engines consist of three main parts: the compressor (where ambient air enters), the combustion chamber (where air is mixed with fuel), and the turbine (which converts fuel into mechanical energy). Early fault detection is critical in the aerospace sector, as any failure can have severe consequences.

An interactive interface was also developed using Streamlit to explore results and predictions from the implemented models.

# Project Structure

This project consists of a main notebook (*Entrega_final.ipynb*), which details all the decisions made throughout the analysis and modeling process, as well as the conclusions obtained after evaluating the different approaches and techniques applied.

For better organization and code cleanliness, the project has been divided into four main modules, each with a specific function:

- *modelos/*: Contains the different architectures of the implemented models, each developed in a separate Python file.
- *paginas/*: Includes the different sections (tabs) used in the Streamlit application, facilitating navigation and modularity of the interface.
- *utils/*: Groups utility functions used during the project development, aiming to avoid code duplication and make the system more scalable and maintainable.
- *utils.py*: General utilities for the project.
- *utils_modelos.py*: Auxiliary functions specific to the models.
- *plots/*: Contains functions dedicated to plotting and visualization, keeping visualization code separate and reusable.

This modular structure ensures more organized, reusable, and scalable development, making it easier to understand and extend the project in the future.

# Dependencies / Libraries
Various Python libraries were used for this project to ensure correct execution. To facilitate installation and reproducibility, a *requirements.txt* file has been provided with all required dependencies.
It is recommended to create a virtual environment before running the project to keep dependencies isolated and avoid conflicts with other environments. Follow these steps:
1. Create a virtual environment
> python3 -m venv env_name

2. Activate the virtual environment
> source env_name/bin/activate   # macOS or Linux
> env_name\Scripts\activate      # Windows


3. Install required dependencies
> pip install -r requirements.txt

4. Run the notebook (to save the models in BentoML)
  
5. Run the BentoML server
> bentoml serve service.py:TurboshaftService --reload

6. Run Streamlit
> streamlit run app_streamlit.py --server.fileWatcherType none
