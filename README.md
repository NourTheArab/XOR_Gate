# XOR Gate - Neural Network

This project implements a simple feedforward neural network from scratch using NumPy to solve the classic XOR problem.

## File Structure

```
XOR_Gate
├── xor.txt                  # XOR dataset in tab-separated format
├── Lab_D_Implementation.py # Full neural network code (training, evaluation, visualization)
├── loss_plot.png            # Plot of training loss over epochs
├── network_diagram.png      # Neural network architecture diagram with weights & biases
├── Implementing_and_Visualizing_a_Neural_Network_to_Solve_the_XOR_Problem.pdf
```

## What’s Implemented

- **Neural Network Initialization**: Two-layer network with customizable size
- **Sigmoid Activation**: Used for both hidden and output layers
- **Data Loading**: Reads logic gate data from tab-separated `.txt` files
- **Forward Propagation**: Computes activations at hidden and output layers
- **Backpropagation**: Calculates gradients using binary cross-entropy loss
- **Training Loop**: Performs multiple epochs of training with parameter updates
- **Loss Tracking**: Records loss during training and plots it
- **Visualization**:
  - Line plot of loss vs. epochs (`loss_plot.png`)
  - Schematic of the neural network with weights and biases (`network_diagram.png`)
- **Output Display**: Final predictions are printed and exported as a DataFrame

## Results

- The model successfully learns XOR with a final loss < 0.01.
- Outputs closely match expected values: [0, 1, 1, 0].

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`
- `pandas`

Install dependencies with:

```bash
pip install numpy matplotlib pandas
```

## Write-Up

See the accompanying PDF for a full explanation of:
- Algorithm design
- Network structure
- Visual output interpretation
- Learning outcomes

**`Implementing_and_Visualizing_a_Neural_Network_to_Solve_the_XOR_Problem.pdf`**

#### Side Note

Added or, nor, nand, and in case someone wants to use it for a different gate.