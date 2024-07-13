import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Neural Network Simulator")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = random.random()
        self.output = 0

    def activate(self, inputs):
        # Simple linear activation followed by a sigmoid function
        z = np.dot(self.weights, inputs) + self.bias
        self.output = 1 / (1 + np.exp(-z))
        return self.output

class Brain:
    def __init__(self, Nin, Nout):
        self.Nin = Nin
        self.Nout = Nout
        self.inputs = [0] * Nin
        self.outputs = [0] * Nout
        self.neurons = []

        for i in range(Nout):
            new_output_neuron = Neuron(Nin)
            self.neurons.append(new_output_neuron)

    def forward(self, inputs):
        self.inputs = inputs
        for i, neuron in enumerate(self.neurons):
            self.outputs[i] = neuron.activate(inputs)
        return self.outputs

# Create a brain with 3 inputs and 2 outputs
brain = Brain(3, 2)

# Main loop
running = True
while running:
    win.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Example input
    inputs = [random.random() for _ in range(3)]
    outputs = brain.forward(inputs)

    # Display neurons and connections (simplified visualization)
    input_neuron_pos = [(100, 100 + i * 100) for i in range(brain.Nin)]
    output_neuron_pos = [(600, 150 + i * 100) for i in range(brain.Nout)]

    for pos in input_neuron_pos:
        pygame.draw.circle(win, BLACK, pos, 20)

    for pos in output_neuron_pos:
        pygame.draw.circle(win, BLACK, pos, 20)

    for i, inp_pos in enumerate(input_neuron_pos):
        for j, out_pos in enumerate(output_neuron_pos):
            pygame.draw.line(win, BLACK, inp_pos, out_pos, 1)

    pygame.display.update()

pygame.quit()
