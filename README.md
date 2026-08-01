# **Multilayer Perceptron (MLP) from Scratch**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## **Overview**
An implementation of a Multilayer Perceptron (MLP) neural network built entirely from scratch in Python. 

The main goal of this project is educational: to truly understand how artificial intelligence works **under the hood**. Instead of relying on high-level frameworks like TensorFlow or PyTorch, this project implements the core mathematical foundations of Deep Learning, including linear algebra, matrix multiplication, and calculus-based optimization.

##  **Features**
*   **Fully custom architecture:** Configurable number of hidden layers and neurons.
*   **Forward Propagation:** Implemented using pure matrix operations.
*   **Backpropagation:** Custom implementation of Gradient Descent to update weights and biases.
*   **Activation Functions:** Built-in mathematical functions and their derivatives.
*   **No Black Boxes:** Only built-in Python tools and NumPy were used for matrix manipulation.

##  **Mathematical Foundations Applied**
To build this network from the ground up, I applied several mathematical concepts directly into code:
*   **Linear Algebra:** Vectors, matrices, and dot products for weighting inputs.
*   **Calculus:** Partial derivatives and the chain rule for backpropagation.
*   **Optimization:** Minimizing the loss function through iterative gradient steps.

## **How to Run the Project**

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Ryan-Fidelis/mlp-from-scratch.git](https://github.com/Ryan-Fidelis/mlp-from-scratch.git)
   ```
2. **Navigate to the directory**
   ```bash
   cd mlp-from-scratch
   ```
3. **Install dependencies**
Make sure you have Python installed. You only need NumPy to run the core network.
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
4. **Run the model**
   ```bash
   python mlp_8_8_1.py
   ```
📈 **Results and Learnings**

This project leverages the NASA Kepler dataset from Kaggle. The goal was to select the most relevant features (columns) to train a simple neural network, aiming to mimic the analysis performed by the Kepler Space Telescope. 

The MLP architecture consists of 3 layers with an 8-8-1 structure, utilizing the Sigmoid activation function for binary classification. By building this entire network from scratch without relying on frameworks like PyTorch or TensorFlow, I gained a deep understanding of how these libraries function behind the scenes, enabling me to use them much more effectively in future projects.

👨‍💻 **Author**

Developed by Ryan Fidelis - AI Technology Student.
Transitioning from industrial mechanics to software development, bringing a strong diagnostic and problem-solving mindset to code.
