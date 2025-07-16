# VariationalMLP.jl

**VariationalMLP.jl** is a Julia module that implements variational inference for multilayer perceptrons (MLPs).  
It is designed for research and experimentation with Bayesian Neural Networks (BNNs).  

This guide covers installation instructions. Please see the tutorials folder for some basic tutorials. 

---

## 📥 Installation

To use this module, you will need [Julia](https://julialang.org/downloads/) installed (version **1.11 or higher** is recommended).

### 1. Install Julia

- Download Julia from the official site: https://julialang.org/downloads/
- Follow platform-specific instructions to install it.
- Optionally, install [VS Code](https://code.visualstudio.com/) and the Julia extension for a better development environment.

### 2. Install Jupyter Notebook

- Open terminal and run

```bash
pip install notebook
```

- Verify installation

```bash
jupyter notebook
```


### 2. Set up a Julia environment

- Navigate to the project directory, where **VariationalMLP.jl** is saved.

```bash
cd "/full/path/to/VariationalMLP"
```


- Start Julia with the project environment run:

```bash
julia --project=.
```

- Inside Julia REPL 

```bash
using Pkg

# Instantiate packages listed in Project.toml
Pkg.instantiate()

# Add IJulia to connect Julia to Jupyter
Pkg.add("IJulia")

```

- Create new Jupyter kernel tied to the project environment

```bash
using IJulia
installkernel("Julia VariationalMLP", env=Dict("JULIA_PROJECT" => pwd()))
```

- Launch Jupyter directly from Julia

```bash
using IJulia
installkernel("Julia VariationalMLP", env=Dict("JULIA_PROJECT" => pwd()))
```

- Start a new notebook and select the correct environment


### Import modules

- Include the path where the files are and run inside a cell

```bash
include(path)

using .VariationalMLP
```

- Similarly you can include supporting files

```bash
include(path)

using .MCUtils.jl
```

