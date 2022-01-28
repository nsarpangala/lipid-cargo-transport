# lipid-cargo-transport
Simulation of transport of lipid cargoes by multiple kinesin motors. Description of the model is in https://www.biorxiv.org/content/10.1101/2021.06.10.447989v3

To setup required directories run

bash setup.bash

Make sure you install all the packages listed in requirements.txt

Prepare a input file with required parameters in inputs/ directory

To run main simulations

python src/simulations/main_script_cargo_rotation.py input_filename.txt 2 num_cores

(num_cores is the number of cores you want to use to run these simulations)

Analysis files can be found in src/analysis/

plot_class_parquet.py has core set of analysis scripts like finding motor force arrays for force distribution, motor off rate,  cargo run length etc.


Developed in the Gopinathan Lab UC Merced

For more information on this model and help with running simulations and analysis, feel free to contact 
Ajay Gopinathan (agopinathan@ucmerced) and Niranjan Sarpangala (nsarpangala@ucmerced.edu)
