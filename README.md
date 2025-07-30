# ForgeDreamer
## Introduction

This repository contains the code to run two different training tasks. Each task requires a separate Conda environment to manage its dependencies. Please follow the instructions below to set up the environments and run the training scripts.

## Prerequisites

Before you begin, ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (Anaconda or Miniconda) installed on your system.

## Installation and Setup

Follow these steps to clone the repository, create the Conda environments, and install the required dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

### 2. Create and Set Up the First Environment (`env1`)

This environment is for running the first training task.

```bash
# Run from your anaconda/miniconda prompt or terminal
# Create a new conda environment named env1 with python 3.9
conda create -n env1 python=3.9 -y

# Activate the newly created environment
conda activate env1

# In the env1 environment, install the first requirements file using pip
pip install -r requirements1.txt
```
> **Note**: `requirements1.txt` should contain all the Python packages needed to run the first training script.

### 3. Create and Set Up the Second Environment (`env2`)

This environment is for running the second training task.

```bash
# Make sure you have deactivated the previous environment, or just create it directly
# Create a new conda environment named env2 with python 3.10
conda create -n env2 python=3.10 -y

# Activate the newly created environment
conda activate env2

# In the env2 environment, install the second requirements file using pip
pip install -r requirements2.txt
```
> **Note**: `requirements2.txt` should contain all the Python packages needed to run the second training script.

## How to Run the Training

Make sure you have activated the correct Conda environment for each task.

### 1. Run the First Training Script

```bash
# Activate the first environment
conda activate env1

# If necessary, first grant execution permissions to the script
# chmod +x train_script_1.sh

# Run the first training script
./train_script_1.sh
```

### 2. Run the Second Training Script

```bash
# Activate the second environment
conda activate env2

# If necessary, first grant execution permissions to the script
# chmod +x train_script_2.sh

# Run the second training script
./train_script_2.sh
```

## Project Structure

```
.
├── train_script_1.sh   # First training script
├── train_script_2.sh   # Second training script
├── requirements1.txt   # Dependencies for env1
├── requirements2.txt   # Dependencies for env2
├── src/                  # Directory for main source code (optional)
└── README.md             # Project README file
```

## Contributing

Contributions, issues, and pull requests are welcome.

## License

This project is licensed under the [MIT](LICENSE) License.
