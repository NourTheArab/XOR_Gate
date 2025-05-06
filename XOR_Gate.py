import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def init_weights_biases(num_input_nodes, num_hidden_nodes, num_output_nodes):
    return {
        "hidden_weights": np.random.randn(num_hidden_nodes, num_input_nodes),
        "hidden_biases": np.zeros((num_hidden_nodes, 1)),
        "output_weights": np.random.randn(num_output_nodes, num_hidden_nodes),
        "output_biases": np.zeros((num_output_nodes, 1))
    }

def read_file_to_array(file_name):
    with open(file_name, 'r') as f:
        lines = f.read().strip().split('\n')
    
    header = lines[0].split('\t')
    data = [list(map(float, line.split('\t'))) for line in lines[1:]]
    
    data = np.array(data).T
    features = data[:-1, :]
    labels = data[-1:, :]
    headers = np.array(header, ndmin=2).T
    
    return features, labels, headers

def forward_propagate(features, weights_biases_dict):
    hidden_inputs = np.dot(weights_biases_dict["hidden_weights"], features) + weights_biases_dict["hidden_biases"]
    hidden_outputs = sigmoid(hidden_inputs)
    output_inputs = np.dot(weights_biases_dict["output_weights"], hidden_outputs) + weights_biases_dict["output_biases"]
    output_outputs = sigmoid(output_inputs)
    return {"hidden_layer_outputs": hidden_outputs, "output_layer_outputs": output_outputs}

def find_loss(output_layer_outputs, labels):
    num_examples = labels.shape[1]
    loss = (-1 / num_examples) * np.sum(
        np.multiply(labels, np.log(output_layer_outputs)) +
        np.multiply(1 - labels, np.log(1 - output_layer_outputs))
    )
    return loss

def backprop(feature_array, labels, output_vals, weights_biases_dict):
    num_examples = labels.shape[1]
    hidden_outputs = output_vals["hidden_layer_outputs"]
    output_outputs = output_vals["output_layer_outputs"]
    output_weights = weights_biases_dict["output_weights"]

    raw_error = output_outputs - labels
    output_weights_gradient = np.dot(raw_error, hidden_outputs.T) / num_examples
    output_bias_gradient = np.sum(raw_error, axis=1, keepdims=True) / num_examples

    blame = np.dot(output_weights.T, raw_error)
    hidden_error = blame * (hidden_outputs * (1 - hidden_outputs))
    hidden_weights_gradient = np.dot(hidden_error, feature_array.T) / num_examples
    hidden_bias_gradient = np.sum(hidden_error, axis=1, keepdims=True) / num_examples

    return {
        "hidden_weights_gradient": hidden_weights_gradient,
        "hidden_bias_gradient": hidden_bias_gradient,
        "output_weights_gradient": output_weights_gradient,
        "output_bias_gradient": output_bias_gradient
    }

def update_weights_biases(params, gradients, learning_rate):
    return {
        "hidden_weights": params["hidden_weights"] - learning_rate * gradients["hidden_weights_gradient"],
        "hidden_biases": params["hidden_biases"] - learning_rate * gradients["hidden_bias_gradient"],
        "output_weights": params["output_weights"] - learning_rate * gradients["output_weights_gradient"],
        "output_biases": params["output_biases"] - learning_rate * gradients["output_bias_gradient"]
    }

def model_file(file_name, num_inputs, num_hiddens, num_outputs, epochs, learning_rate):
    features, labels, _ = read_file_to_array(file_name)
    params = init_weights_biases(num_inputs, num_hiddens, num_outputs)
    losses = []

    for epoch in range(epochs):
        output_vals = forward_propagate(features, params)
        loss = find_loss(output_vals["output_layer_outputs"], labels)
        losses.append(loss)
        gradients = backprop(features, labels, output_vals, params)
        params = update_weights_biases(params, gradients, learning_rate)
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss:.5f}")

    return params, losses

def plot_loss_curve(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()

def plot_neural_net(weights, output_weights, hidden_biases=None, output_biases=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis('off')

    layers = {
        "input": [(0, 0.6), (0, 0.3)],
        "hidden": [(1.5, 0.7), (1.5, 0.2)],
        "output": [(3, 0.45)]
    }

    def draw_connection(x1, y1, x2, y2, weight, ax):
        color = 'blue' if weight >= 0 else 'red'
        linewidth = min(4, 0.5 + abs(weight) / 2)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.7)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, f"{weight:.2f}", fontsize=8, color=color)

    for layer, positions in layers.items():
        for i, (x, y) in enumerate(positions):
            circle = plt.Circle((x, y), 0.07, fill=True, color='lightgray', edgecolor='black')
            ax.add_patch(circle)
            if layer == "input":
                ax.text(x - 0.2, y, f"{'x' if i==0 else 'y'}", fontsize=13, va='center', fontweight='bold')
            elif layer == "hidden":
                bias = hidden_biases[i][0] if hidden_biases is not None else 0
                ax.text(x, y, f"h{i+1}\nb={bias:.2f}", fontsize=9, ha='center', va='center')
            elif layer == "output":
                bias = output_biases[0][0] if output_biases is not None else 0
                ax.text(x + 0.2, y, f"out\nb={bias:.2f}", fontsize=10, va='center', ha='left')

    for i, (x1, y1) in enumerate(layers["input"]):
        for j, (x2, y2) in enumerate(layers["hidden"]):
            draw_connection(x1, y1, x2, y2, weights[j, i], ax)

    for j, (x2, y2) in enumerate(layers["hidden"]):
        x3, y3 = layers["output"][0]
        draw_connection(x2, y2, x3, y3, output_weights[0, j], ax)

    plt.title("Neural Network Architecture (with Weights & Biases)", fontsize=14)
    plt.tight_layout()
    plt.savefig("network_diagram.png")
    plt.show()

file_name = "xor.txt"
params, losses = model_file(file_name, 2, 2, 1, 5000, 0.3)
features, labels, _ = read_file_to_array(file_name)
final_outputs = forward_propagate(features, params)
final_predictions = final_outputs["output_layer_outputs"]

plot_loss_curve(losses)
df = pd.DataFrame(final_predictions.T, columns=["Output"])
print("\nFinal Predictions (XOR):")
print(df)

plot_neural_net(
    params["hidden_weights"],
    params["output_weights"],
    params["hidden_biases"],
    params["output_biases"]
)