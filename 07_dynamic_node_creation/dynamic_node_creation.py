import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import collections

# --- Dynamic Network Definition ---

class DynamicNet(nn.Module):
    """
    A neural network that can dynamically add nodes to its hidden layer.
    """
    def __init__(self, input_size, output_size, initial_hidden_size=1):
        """
        Initializes the network.
        Args:
            input_size (int): The number of input features.
            output_size (int): The number of output features.
            initial_hidden_size (int): The starting number of nodes in the hidden layer.
        """
        super(DynamicNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = initial_hidden_size

        # Define the layers
        self.hidden_layer = nn.Linear(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # Sigmoid activation function, as used in the paper (logistic function)
        self.activation = nn.Sigmoid()

        # Initialize weights as per the paper's description
        self.initialize_weights()

    def forward(self, x):
        """
        Performs the forward pass.
        """
        x = self.activation(self.hidden_layer(x))
        x = self.activation(self.output_layer(x))
        return x

    def initialize_weights(self):
        """
        Initializes weights to small random values, as described in the paper.
        The paper used a range of -0.1666 to +0.1666. We'll use a similar uniform distribution.
        """
        init_range = 0.1666
        for param in self.parameters():
             if param.data.ndimension() >= 2: # Check if it's a weight matrix
                nn.init.uniform_(param.data, -init_range, init_range)
             else: # It's a bias vector
                nn.init.zeros_(param.data)


    def add_hidden_node(self):
        """
        Adds a single node to the hidden layer and reinitializes the optimizer.
        This is the core of the "Dynamic Node Creation" method.
        """
        print(f"\n--- Adding a new hidden node. New size: {self.hidden_size + 1} ---\n")
        
        # 1. Store old weights and biases
        old_hidden_weights = self.hidden_layer.weight.data
        old_hidden_biases = self.hidden_layer.bias.data
        old_output_weights = self.output_layer.weight.data
        
        # 2. Increment hidden layer size
        self.hidden_size += 1
        
        # 3. Create new, larger layers
        new_hidden_layer = nn.Linear(self.input_size, self.hidden_size)
        new_output_layer = nn.Linear(self.hidden_size, self.output_size)

        # 4. Initialize new layers with small random weights first
        init_range = 0.1666
        nn.init.uniform_(new_hidden_layer.weight.data, -init_range, init_range)
        nn.init.zeros_(new_hidden_layer.bias.data)
        nn.init.uniform_(new_output_layer.weight.data, -init_range, init_range)
        nn.init.zeros_(new_output_layer.bias.data)
        
        # 5. Copy old weights and biases into the new layers
        # For the hidden layer, we add a new row for the new neuron
        new_hidden_layer.weight.data[:-1, :] = old_hidden_weights
        new_hidden_layer.bias.data[:-1] = old_hidden_biases
        
        # For the output layer, we add a new column for the new connection
        new_output_layer.weight.data[:, :-1] = old_output_weights

        # 6. Replace the old layers with the new ones
        self.hidden_layer = new_hidden_layer
        self.output_layer = new_output_layer


# --- Training Logic ---

def train_dnc(model, X, y, max_epochs=20000):
    """
    Trains the DynamicNet using the Dynamic Node Creation algorithm.
    
    Args:
        model (DynamicNet): The network instance to train.
        X (torch.Tensor): The input data.
        y (torch.Tensor): The target data.
        max_epochs (int): The maximum number of training epochs.
    """
    # Parameters from the paper (Table I & Implementation section)
    learning_rate = 0.5
    momentum = 0.9
    w = 1000          # Window width for checking error slope
    delta_T = 0.05    # Trigger slope for adding a node
    C_a = 0.001       # Desired average squared error cutoff
    C_m = 0.01        # Desired maximum squared error cutoff

    # Initialize training variables
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss(reduction='none') # 'none' to calculate max error manually
    
    t0 = 0  # Time (epoch) when the last node was added
    error_history = collections.deque(maxlen=w + 1)
    a_t0 = 1.0 # Initial error when the "last" node was added (start of training)
    
    node_growth_enabled = True
    start_time = time.time()

    print("--- Starting DNC Training ---")
    print(f"Initial hidden nodes: {model.hidden_size}")
    print(f"Parameters: LR={learning_rate}, Momentum={momentum}, Window(w)={w}, Trigger(ΔT)={delta_T}")
    print(f"Stop Conditions: AvgError(Ca)<={C_a}, MaxError(Cm)<={C_m}\n")

    for t in range(1, max_epochs + 1):
        # --- Standard Training Step ---
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        
        # Calculate loss
        loss_per_pattern = criterion(outputs, y)
        total_loss = loss_per_pattern.mean()
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # --- DNC Logic ---
        # Calculate error metrics as per the paper (Table I)
        # a_t: Average squared error per output node
        a_t = total_loss.item()
        # m_t: Maximum squared error on any output node in any pattern
        m_t = loss_per_pattern.max().item()
        
        error_history.append(a_t)

        # Check stopping condition (Equation 4)
        if a_t <= C_a and m_t <= C_m:
            if node_growth_enabled:
                print(f"--- Node growth disabled at epoch {t} ---")
                print(f"Reason: Target accuracy reached (AvgError={a_t:.5f}, MaxError={m_t:.5f})")
                node_growth_enabled = False
            # Allow for fine-tuning after growth is disabled
            if a_t < 0.0001: # A stricter condition to finally stop
                 break


        # Check node creation condition (Equations 2 & 3)
        if node_growth_enabled and (t - t0 >= w):
            # We need at least 'w' samples in our history to calculate the slope
            if len(error_history) > w:
                a_t_minus_w = error_history[0] # The error from w steps ago
                
                # Equation 2: (a_t-w - a_t) / a_t0 < delta_T
                # The paper's formula seems to have a typo (a_t - a_t-w).
                # The error drop (a_t-w - a_t) should be positive.
                error_drop = a_t_minus_w - a_t
                slope = error_drop / a_t0 if a_t0 > 0 else float('inf')

                if slope < delta_T:
                    model.add_hidden_node()
                    # Reset optimizer with new parameters
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
                    # Reset DNC tracking variables
                    t0 = t
                    a_t0 = a_t
                    error_history.clear()

        if t % 1000 == 0:
            print(f"Epoch {t:5d}/{max_epochs} | "
                  f"Hidden Nodes: {model.hidden_size} | "
                  f"Avg Sq Error: {a_t:.6f} | "
                  f"Max Sq Error: {m_t:.6f}")

    end_time = time.time()
    print("\n--- Training Finished ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Final epoch: {t}")
    print(f"Final hidden nodes: {model.hidden_size}")
    print(f"Final Avg Sq Error: {a_t:.6f}")
    print(f"Final Max Sq Error: {m_t:.6f}")


# --- Main Execution ---
if __name__ == '__main__':
    # Set up the PAR2 (XOR) problem
    # Inputs: 2, Outputs: 1, Patterns: 4
    X_train = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
    y_train = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

    # Initialize the model, starting with 1 hidden node as per the paper's method
    dnc_net = DynamicNet(input_size=2, output_size=1, initial_hidden_size=1)

    # Run the training
    train_dnc(dnc_net, X_train, y_train)

    # --- Verification ---
    print("\n--- Verifying final network performance ---")
    dnc_net.eval()
    with torch.no_grad():
        predictions = dnc_net(X_train)
        for i in range(len(X_train)):
            print(f"Input: {X_train[i].numpy()} -> "
                  f"Target: {y_train[i].numpy()[0]:.1f}, "
                  f"Prediction: {predictions[i].numpy()[0]:.4f}")

