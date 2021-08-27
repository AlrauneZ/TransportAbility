# Overview

This project provides all python scripts to reproduce the results of the 
paper "A Probabilistic Formulation of the Diffusion Coefficient in Porous Media 
as Function of Porosity" by Alraune Zech and Matthijs de Winter

It provides the class implementation of the upscaling workflows, both numerical
and theoretical upscaling. It further provides simulation results of upscaling
workflows presented in the manuscript and python scripts to reproduce all figures
based on the input and upscaling data. 


## Structure

The project is organized as follows:

- `README.md` - description of the project
- `LICENSE` - the default license is MIT
- `data/` - folder containing data:
  + `FCC_2-1_por_ta_data_d2_r2.csv` - observational data from at resolution r = 2
  + remaining filesa are results of upscaling workflows 
- `results/` - folder containing plots and a folder with example data for upscaling workflow
- `src/` - folder containing the Python scripts of the project:
  + `00_run_upscaling.py` - run an upscaling workflow   
  + `01_pdf_porosity.py` - reproducing Figure 1 of the manuscript
  + `02_Scatter_TA_Data.py` - reproducing Figure 2 of the manuscript
  + `03_Normality_Histogram.py` - reproducing Figure 3 of the manuscript
  + `04_stats_TA.py` - reproducing Figure 4 of the manuscript
  + `05_Scatter_TA_eff_2D.py` - reproducing Figure 6a of the manuscript
  + `06_pdf_marginal_TA_por.py` - reproducing Figure 6b+c of the manuscript
  + `07_ens_stats_evolution.py` - reproducing Figure 7 of the manuscript
  + `08_cloud_TA_pdf.py` - reproducing Figure 8 of the manuscript
  + `Distributions.py` - containg classes for specifying porosity distribution and
  a class for analysing connected transport ability data distributed over a range of porosity values
  + `TA_Simulation.py` - containing class for numerical upscaling work flow to generate 
  ensemble of networks consisting and calcuting network properties and the class on 
  calculating the transport ability through the network flow simulation
  + `TA_Upscaling.py` - containing class which combines numerical and theoretical upscaling


## Python environment

To make the example reproducible, we provide the following files:
- `requirements.txt` - requirements for [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files) to install all needed packages


## Contact

You can contact us via <a.zech@uu.nl>.


## License

MIT © 2021
