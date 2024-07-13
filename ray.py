import pygame
import numpy as np
from agent import Agent
import time

# Initialize Pygame
pygame.init()

# Screen dimensions
TOTAL_WIDTH, TOTAL_HEIGHT = 2000, 1500
VIEW_WIDTH, VIEW_HEIGHT = 1200, 800
screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
pygame.display.set_caption('Raycasting Agents')

# Define areas
MAIN_SCENE_WIDTH = int(VIEW_WIDTH * 0.75)
MAIN_SCENE_HEIGHT = int(VIEW_HEIGHT * 0.75)

# Create surfaces
main_scene = pygame.Surface((MAIN_SCENE_WIDTH, MAIN_SCENE_HEIGHT))
gui_area = pygame.Surface((VIEW_WIDTH - MAIN_SCENE_WIDTH, VIEW_HEIGHT))
minimap = pygame.Surface((MAIN_SCENE_WIDTH, VIEW_HEIGHT - MAIN_SCENE_HEIGHT))

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (139, 0, 0)
GREEN = (0, 100, 0)
BLUE = (0, 0, 139)
YELLOW = (255, 255, 0)
BROWN = (139, 69, 19)
LIGHT_CYAN = (224, 255, 255)

# Colors for agents
AGENT_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
]

# Walls for S-shaped labyrinth (each wall is defined by a start and end point with color)
walls = [
    ((100, 100), (1900, 100), RED),       # Top wall
    ((1900, 100), (1900, 1400), RED),     # Right wall
    ((1900, 1400), (100, 1400), RED),     # Bottom wall
    ((100, 1400), (100, 100), RED),       # Left wall
    
    # S-shaped walls
    ((100, 400), (1600, 400), GREEN),
    ((1600, 400), (1600, 1100), GREEN),
    ((1600, 1100), (600, 1100), GREEN),
    ((600, 1100), (600, 600), GREEN),
    ((600, 600), (1300, 600), GREEN),
    ((1300, 600), (1300, 900), GREEN),
    ((1300, 900), (300, 900), GREEN),
    ((300, 900), (300, 1200), GREEN)
]

# Agent settings
start_position = [200, 200]
agents = [Agent(position=start_position, walls=walls, color=color) for color in AGENT_COLORS]

def spawn_new_generation(best_agent):
    new_agents = []
    for color in AGENT_COLORS:
        new_agents.append(Agent(position=start_position, walls=walls, color=color))
    if best_agent:
        new_agents[0] = Agent(position=start_position, walls=walls, color=best_agent.color)
        new_agents[0].brain = best_agent.brain
    return new_agents

# Camera settings
camera_pos = np.array([0, 0])
camera_speed = 20
dragging = False
drag_start_pos = None
drag_camera_start_pos = None

running = True
clock = pygame.time.Clock()
generation_time = 10  # Time per generation in seconds
start_time = time.time()

while running:
    current_time = time.time()
    if current_time - start_time >= generation_time:
        # Find the best agent based on fitness
        best_agent = max(agents, key=lambda agent: agent.fitness)
        # Spawn a new generation of agents
        agents = spawn_new_generation(best_agent)
        start_time = current_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                dragging = True
                drag_start_pos = pygame.mouse.get_pos()
                drag_camera_start_pos = camera_pos.copy()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                mouse_pos = pygame.mouse.get_pos()
                drag_offset = np.array(mouse_pos) - np.array(drag_start_pos)
                camera_pos = drag_camera_start_pos - drag_offset

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        camera_pos[1] -= camera_speed
    if keys[pygame.K_DOWN]:
        camera_pos[1] += camera_speed
    if keys[pygame.K_LEFT]:
        camera_pos[0] -= camera_speed
    if keys[pygame.K_RIGHT]:
        camera_pos[0] += camera_speed

    # Automated neural network-based movement for all agents
    for agent in agents:
        agent.neural_move()
        agent.update_lifespan()

    # Clear surfaces
    main_scene.fill(BROWN)  # Fill the entire scene with brown
    
    # Draw the white rectangle inside the largest box
    inner_rect = pygame.Rect(
        100 - camera_pos[0], 100 - camera_pos[1], 
        1800, 1300
    )
    pygame.draw.rect(main_scene, WHITE, inner_rect)

    # Draw the very light cyan square in the middle
    square_size = 200
    square_center = np.array([TOTAL_WIDTH / 2, TOTAL_HEIGHT / 2])
    square_rect = pygame.Rect(
        square_center[0] - square_size / 2 - camera_pos[0],
        square_center[1] - square_size / 2 - camera_pos[1],
        square_size,
        square_size
    )
    pygame.draw.rect(main_scene, LIGHT_CYAN, square_rect)

    # Draw walls on main_scene with camera offset
    for wall_start, wall_end, color in walls:
        adjusted_wall_start = np.array(wall_start) - camera_pos
        adjusted_wall_end = np.array(wall_end) - camera_pos
        pygame.draw.line(main_scene, color, adjusted_wall_start, adjusted_wall_end, 4)

    # Draw all agents and their rays
    for agent in agents:
        agent.draw(main_scene, camera_pos)

    # Draw borders
    pygame.draw.rect(main_scene, BLACK, main_scene.get_rect(), 4)
    pygame.draw.rect(gui_area, BLACK, gui_area.get_rect(), 4)
    pygame.draw.rect(minimap, BLACK, minimap.get_rect(), 4)

    # Update main screen
    screen.fill(WHITE)
    screen.blit(main_scene, (0, 0))
    screen.blit(gui_area, (MAIN_SCENE_WIDTH, 0))
    screen.blit(minimap, (0, MAIN_SCENE_HEIGHT))
    
    # Draw border around the entire canvas
    pygame.draw.rect(screen, BLACK, screen.get_rect(), 4)

    pygame.display.flip()
    clock.tick(30)  # Control the speed of the agents' movement

pygame.quit()
