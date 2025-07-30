# ForgeDreamer
## Example
<h3 align="center">Generated 3D Models Showcase</h3>

<table>
  <!-- 第一行 -->
  <tr>
    <td align="center">
      <b>Bearing</b><br>
      <img src="[ForgeDreamer/Example/bearing.gif](https://github.com/Junhaocai27/ForgeDreamer/blob/main/Example/bearing.gif)" width="260">
    </td>
    <td align="center">
      <b>Gasket</b><br>
      <img src="ForgeDreamer/Example/gasket.gif" width="260">
    </td>
    <td align="center">
      <b>Green LED</b><br>
      <img src="ForgeDreamer/Example/green_LED.gif" width="260">
    </td>
  </tr>
  
  <!-- 第二行 -->
  <tr>
    <td align="center">
      <b>Red LED</b><br>
      <img src="ForgeDreamer/Example/red_LED.gif" width="260">
    </td>
    <td align="center">
      <b>Nail</b><br>
      <img src="ForgeDreamer/Example/nail.gif" width="260">
    </td>
    <td align="center">
      <b>Nut</b><br>
      <img src="ForgeDreamer/Example/nut.gif" width="260">
    </td>
  </tr>
  
  <!-- 第三行 -->
  <tr>
    <td align="center">
      <b>Resistor</b><br>
      <img src="ForgeDreamer/Example/resistor.gif" width="260">
    </td>
    <td align="center">
      <b>Screw</b><br>
      <img src="ForgeDreamer/Example/screw.gif" width="260">
    </td>
    <td align="center">
      <b>Nut and Screw</b><br>
      <img src="ForgeDreamer/Example/nut_screw.gif" width="260">
    </td>
  </tr>
</table>

## Introduction

This repository contains the code to run two different training tasks. Each task requires a separate Conda environment to manage its dependencies. Please follow the instructions below to set up the environments and run the training scripts.

## Installation and Setup

Follow these steps to clone the repository, create the Conda environments, and install the required dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

### 2. Create and Set Up the First Environment (`LoRA_Distillation`)

This environment is for running the Lora Distillation task.

```bash
# Run from your anaconda/miniconda prompt or terminal
# Create a new conda environment named lora_distillation with python 3.10
conda create -n lora_distillation python=3.10 -y

# Activate the newly created environment
conda activate lora_distillation

# In the env1 environment, install the first requirements file using pip
pip install -r requirements_disLoRA.txt
```

### 3. Create and Set Up the Second Environment (`ForgeDreamer`)

This environment is for running the second training task.

```bash
# Make sure you have deactivated the previous environment, or just create it directly
# Create a new conda environment named ForgeDreamer with python 3.10
conda create -n ForgeDreamer python=3.10 -y

# Activate the newly created environment
conda activate ForgeDreamer

# In the env2 environment, install the second requirements file using pip
pip install -r requirements_t23d.txt
```

## How to Run the Training

Make sure you have activated the correct Conda environment for each task.

### 1. Run the First Training Script

```bash
# Activate the first environment
conda activate lora_distillation

# Run the first training script
sh ditill.sh
```

### 2. Run the Second Training Script

```bash
# Activate the second environment
conda activate ForgeDreamer

# Run the second training script
sh train.sh
```
