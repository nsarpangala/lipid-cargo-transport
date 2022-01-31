# Brownian Dynamics model of the transport of lipid cargoes by multiple kinesin motors.


This is a model of transport of vesicles by teams of kinesin motors. 

Please the details of the model and analysis in our [pre-print](https://www.biorxiv.org/content/10.1101/2021.06.10.447989v3)


### Instructions to help you run the code:


- **Initial set-up**

  ```
  git clone git@github.com:nsarpangala/lipid-cargo-transport.git lipid_cargo_transport_v2
  cd lipid_cargo_transport_v2
  bash setup.bash
  conda create --name <env_name> --file requirements.txt
  conda activate <env_name>
  ```

- **Create an input file with desired parameters for simulation in the `inputs/` directory.**
   Please use the sample file given in the `inputs/` directory as a template. 

- **Run cargo transport simulations**

  `python src/simulations/main_script_cargo_rotation.py input_filename.txt 2 num_cores`

  * num_cores is the number of cores you want to use to run these simulations
  * Choose appropriate main_script file in `src/simulation/` depending on whether you want to run simulations with or without cargo rotation.
  * When the simulation is complete, the data set with cargo center of mass, motor anchor positions etc should be stored in `~/data/3dtransport/<simname>`

- **Analyse the data from simulations**
  Please find some useful analysis scripts that compute the metrics like cargo runlength, average number of bound motors, motor off-rate etc in `src/analysis/`
  `src/analysis/plot_class_parquet.py` has the core set of analysis methods.


This computational model was developed in the [Gopinathan Lab, UC Merced](http://gopinathanlab.ucmerced.edu/)

For more information on this model, help with running simulations and analysis, and issues feel free to contact 
Ajay Gopinathan (agopinathan at ucmerced dot edu) and Niranjan Sarpangala (nsarpangala at ucmerced dot edu)

NS acknowledges helpful tips from Dr. David Quint and @Carreau on writing the code.
