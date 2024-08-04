PROJECT OVERVIEW
Introduction
This project focuses on comparing the performance of training a neural network model on different hardware accelerators: GPU, TPU, and CPU. The analysis includes training the model on each type of hardware, evaluating the training performance, and visualizing the results.

Directory Structure
.git: Contains Git version control data.
.gitattributes: Git attributes file.
charts: Directory for storing generated charts and visualizations.
main.py: The main script to run the project.
src: Source code directory containing modules for various tasks.
Key Components
main.py
The central script that coordinates the following tasks:

Hardware Detection:

Checks for available GPUs.
Model Training and Evaluation:

Trains a neural network model on GPU.
Trains a neural network model on TPU.
Trains a neural network model on CPU.
Saves the training accuracy plots for each hardware type.
src Directory
Contains the following modules:

train.py: Contains functions to train the model on GPU, TPU, and CPU.
utils.py: Contains utility functions such as saving plots.

Conclusion
This project provides a comprehensive comparison of training a neural network model on different hardware accelerators: GPU, TPU, and CPU. It includes detailed visualizations of the training performance for each hardware type. ​