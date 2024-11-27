**Simulation repository**

Contents: 
1. The simulation itself (main_simulation)
2. The simulation methods (simulation_methods)
4. Form factor methods (form_factor_methods)
5. Example simulation run (run_simulation_analysis)

For a detailed explanation on how the simulation works, refer to the powerpoint file.

To run the simulation, you need to make sure you are within
```ruby
if __name__ == "__main__":
```
A callable example of the simulation can be done by running
```ruby
if __name__ == "__main__":
    new_simulation.example()
```

Running the simulation with custom parameters can be done using 
```ruby
if __name__ == "__main__":
    new_simulation.IqMC_2D(arguments)
```

An example of how these arguments look like can be seen in *run_simulation_analysis* and in the example() method of *new_simulation*. 

Note that the scattering function for the simulation includes polydispersity. This means that it must be formatted as function(q, param), where param is your polydisperse parameter. If the function includes more than one parameter (e.g, elliptical scattering), you must wrap it in a way that only one parameter is given to the function:
```ruby
def new_function(x, param_a):
    param_b = number
    return old_function(x, param_a, param_b)
```
On default, the polydisperse parameter is the radius of gyration Rg. 

In addition to the form factor, you must also account to include to the simulation the set of possible Rg (or polydisperse parameter) values and their distribution. 
In this way, if you want to simulate a sample with no polydispersity, you can give it a very sharp distribution function, or only include one possible Rg value. In the example the default distribution of Rgs is a gaussian distribution around Rg=4nm and a relative variance of 0.05.




