import pygame
import numpy as np
import random

class Brain:
    def __init__(self, layer_sizes, activation_functions, output_activations):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.activation_functions = activation_functions
        self.output_activations = output_activations

        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.random.randn(layer_sizes[i + 1])
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def activate(self, x, func):
        if func == 'sigmoid':
            return self.sigmoid(x)
        elif func == 'tanh':
            return self.tanh(x)
        elif func == 'relu':
            return self.relu(x)

    def forward(self, inputs):
        activations = [inputs]
        for i in range(len(self.weights) - 1):
            inputs = self.activate(np.dot(inputs, self.weights[i]) + self.biases[i], self.activation_functions[i])
            activations.append(inputs)

        # Handle the final layer separately for different output activations
        final_layer_input = np.dot(inputs, self.weights[-1]) + self.biases[-1]
        outputs = np.zeros_like(final_layer_input)
        for j in range(len(outputs)):
            outputs[j] = self.activate(final_layer_input[j], self.output_activations[j])
        activations.append(outputs)
        
        return activations

    def decide_action(self, inputs):
        activations = self.forward(inputs)
        outputs = activations[-1]
        actions = {
            'thrust_level': outputs[0],  # sigmoid
            'brake_level': outputs[1],   # sigmoid
            'turning': outputs[2]        # tanh
        }
        return actions, activations

def draw_text(win, text, pos, color,size=16):
    font = pygame.font.SysFont('arial', size)
    text_surface = font.render(text, True, color)
    win.blit(text_surface, pos)

def main():
    # Initialize Pygame
    pygame.init()

    # Set up display
    width, height = 1200, 800
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Neural Network Simulator")

    weight_label_N = 10

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    # Define the neural network structure and activation functions
    N = 5  # Number of raycasts
    input_size = 4 + N  # posX, posY, velX, velY, and raycast distances
    output_size = 3  # thrust level, brake level, turning
    layer_sizes = [input_size, 8, 6, output_size]
    activation_functions = ['relu', 'tanh', 'relu']
    output_activations = ['sigmoid', 'sigmoid', 'tanh']  # last layer activation functions
    brain = Brain(layer_sizes, activation_functions, output_activations)

    # Main loop
    running = True
    while running:
        win.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False

        # Example input: posX, posY, velX, velY, and N raycast distances
        inputs = np.random.rand(input_size)
        actions, activations = brain.decide_action(inputs)

        # Display neurons and connections (simplified visualization)
        neuron_positions = []

        for layer_index, layer_size in enumerate(layer_sizes):
            layer_positions = [
                (100 + layer_index * 250, 100 + i * (height - 200) // (layer_size - 1) if layer_size > 1 else height // 2)
                for i in range(layer_size)
            ]
            neuron_positions.append(layer_positions)

        for layer_index in range(len(neuron_positions) - 1):
            for i, start_pos in enumerate(neuron_positions[layer_index]):
                for j, end_pos in enumerate(neuron_positions[layer_index + 1]):
                    weight = brain.weights[layer_index][i, j]
                    color = BLUE if weight >= 0 else RED
                    pygame.draw.line(win, BLACK, start_pos, end_pos, 1)
                    mid_pos = (((weight_label_N-1)*start_pos[0] + end_pos[0]) // weight_label_N, ((weight_label_N-1)*start_pos[1] + end_pos[1]) // weight_label_N)
                    draw_text(win, f'{weight:.2f}', (mid_pos[0] - 10, mid_pos[1] - 10), color,size=8)

        for layer_index, layer in enumerate(neuron_positions):
            for i, pos in enumerate(layer):
                activation = activations[layer_index][i]
                color = BLUE if activation >= 0 else RED
                pygame.draw.circle(win, color, pos, 20)
                pygame.draw.circle(win, BLACK, pos, 20, 2)  # Draw the outline
                draw_text(win, f'{activation:.2f}', (pos[0] - 10, pos[1] - 10), WHITE, size=12)

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
