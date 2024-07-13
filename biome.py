import numpy as np
import pygame
import sys
import os

# Parameters
A, B, C = 0.3, 0.2, 0.1
w, k, j = 2 * np.pi / 800, 2 * np.pi / 800, 2 * np.pi / 800  # Adjusted to ensure periodicity
nx = 1000  # Number of spatial points
X_min, X_max = -400, 400  # X range
dx = (X_max - X_min) / nx

# Define the terrain function Z(X) with periodic boundary conditions
X = np.linspace(X_min, X_max, nx)
Z = A * np.cos(w * X) + B * np.sin(k * X) + C * np.sin(j * X)

# Initialize Pygame
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Terrain Visualization")
clock = pygame.time.Clock()

def draw_terrain(Z):
    screen.fill((173, 216, 230))  # Light cyan background
    terrain_points = []

    # Calculate scaling factors
    min_Z, max_Z = np.min(Z), np.max(Z)
    height_scale = WINDOW_HEIGHT / 2 * (max_Z - min_Z)
    width_scale = WINDOW_WIDTH / nx

    for i in range(nx):
        terrain_height = WINDOW_HEIGHT - int((Z[i] - min_Z) * height_scale) - 100
        screen_x = int(i * width_scale)
        
        terrain_points.append((screen_x, terrain_height))
    
    terrain_points.append((WINDOW_WIDTH, WINDOW_HEIGHT))
    terrain_points.append((0, WINDOW_HEIGHT))
    
    pygame.draw.polygon(screen, (139, 69, 19), terrain_points)  # Brown terrain

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_s:
                screenshot_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'terrain_screenshot.png')
                pygame.image.save(screen, screenshot_path)
                print(f"Screenshot saved to {screenshot_path}")

    # Render the terrain
    draw_terrain(Z)
    pygame.display.flip()

    # Control the frame rate
    clock.tick(30)  # Set frame rate

pygame.quit()
sys.exit()
