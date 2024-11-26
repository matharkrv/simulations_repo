**Simulation repository**

Contents: 
1. The simulation itself (main_simulation)
2. The simulation methods (simulation_methods)
4. Form factor methods (form_factor_methods)
5. Example simulation run (run_simulation_analysis)

For a detailed explanation on how the simulation works, refer to the powerpoint file.

Using the example in the simulation file you can try to run whichever form factor you want, given that it is formatted as function(q, Rg). For more complex form factors which include more than 1 parameter, 
you should wrap them in an external function so that they fit into the function(q, Rg) format. 

Reading and analysing the simulation data can be done as seen in the example simulation run file (run_simulation_analysis).
