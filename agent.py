import numpy as np
import pygame
import time
from brain import Brain

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (139, 0, 0)
GREEN = (0, 100, 0)
BLUE = (0, 0, 139)
YELLOW = (255, 255, 0)
BROWN = (139, 69, 19)
LIGHT_CYAN = (224, 255, 255)

def whiten_color(color,n=4):
    res = ((color[0]+255*(n-1))/n,(color[1]+255*(n-1))/n,(color[2]+255*(n-1))/n)
    return res

class Agent:
    def __init__(self, position, walls, color, angle=0, num_rays=7, ray_length=600, move_speed=5, rotate_speed=np.pi/36):
        self.position = np.array(position, dtype=float)
        self.angle = angle
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.ray_angles = np.linspace(-np.pi/4, np.pi/4, num_rays)
        self.move_speed = move_speed
        self.rotate_speed = rotate_speed
        self.walls = walls
        self.color = color
        self.trail_color = tuple(min(255, c + 60) for c in color)
        self.alive = True
        self.start_time = time.time()
        self.lifespan = 0
        self.max_lifespan = 10  # Maximum lifespan in seconds
        self.collision_color = BLACK
        self.velocity = np.array([0, 0], dtype=float)
        self.fitness = 0
        self.visited_positions = set()
        self.trail = []
        self.cell_size = 300  # Using 300x300 pixel squares

        # Initialize the brain
        input_size = num_rays + 5  # posX, posY, velX, velY, angle, and raycast distances
        output_size = 3  # thrust level, brake level, turning
        layer_sizes = [input_size, 8, 6, output_size]
        activation_functions = ['relu', 'tanh', 'relu']
        output_activations = ['sigmoid', 'sigmoid', 'tanh']  # last layer activation functions
        self.brain = Brain(layer_sizes, activation_functions, output_activations)
        self.surface = pygame.Surface((60, 60), pygame.SRCALPHA)

    def move_forward(self, thrust_level):
        if self.alive:
            self.velocity = thrust_level * self.move_speed * np.array([np.cos(self.angle), np.sin(self.angle)])
            new_position = self.position + self.velocity
            if not self.check_collision(new_position):
                self.position = new_position
                self.update_fitness_and_trail(new_position)
            else:
                self.die()

    def move_backward(self, brake_level):
        if self.alive:
            self.velocity = -brake_level * self.move_speed * np.array([np.cos(self.angle), np.sin(self.angle)])
            new_position = self.position + self.velocity
            if not self.check_collision(new_position):
                self.position = new_position
                self.update_fitness_and_trail(new_position)
            else:
                self.die()

    def rotate_left(self, turning):
        if self.alive:
            self.angle -= turning * self.rotate_speed

    def rotate_right(self, turning):
        if self.alive:
            self.angle += turning * self.rotate_speed

    def check_collision(self, new_position):
        for wall_start, wall_end, color in self.walls:
            if self.line_intersect_circle(wall_start, wall_end, new_position, 10):  # Assuming agent radius is 10
                self.lifespan = time.time() - self.start_time
                self.collision_color = color
                return True
        return False

    def die(self):
        self.alive = False

    def update_fitness_and_trail(self, new_position):
        # Discretize the position to the nearest cell
        cell_x = int(new_position[0] // self.cell_size)
        cell_y = int(new_position[1] // self.cell_size)
        cell_position = (cell_x, cell_y)
        if cell_position not in self.visited_positions:
            self.visited_positions.add(cell_position)
            self.fitness += 1
        self.trail.append(self.position.copy())

    def draw(self, screen, camera_pos):
        if self.alive:
            # Draw agent
            self.draw_triangle(screen, self.position - camera_pos, self.angle)
            # Draw fitness value beside the agent
            fitness_position = self.position - camera_pos + np.array([30, -30])
            self.draw_text(screen, f'Fitness: {self.fitness}', fitness_position, self.color)
            # Draw trail
            self.draw_trail(screen, camera_pos)
            # Cast and draw rays
            for angle in self.ray_angles:
                ray_dir = np.array([np.cos(self.angle + angle), np.sin(self.angle + angle)])
                ray_end = self.position + self.ray_length * ray_dir
                closest_intersection, min_dist, wall_color = self.find_closest_intersection(ray_dir)
                if closest_intersection is not None:
                    self.draw_ray(screen, self.position - camera_pos, closest_intersection - camera_pos, whiten_color(wall_color))
                    self.draw_text(screen, f'{min_dist:.0f}', closest_intersection - camera_pos, BLACK if wall_color != BLACK else WHITE)
                else:
                    self.draw_ray(screen, self.position - camera_pos, ray_end - camera_pos, RED)
                    self.draw_text(screen, f'{self.ray_length:.0f}', ray_end - camera_pos, RED)
        else:
            # Draw lifespan text offset by velocity direction
            text_position = self.position - camera_pos - self.velocity * 2
            self.draw_text(screen, f'Lifespan: {self.lifespan:.0f} seconds', text_position, self.collision_color)
            # Draw the trail even when the agent is dead
            self.draw_trail(screen, camera_pos)

    def draw_triangle(self, screen, pos, angle, size=20):
        half_size = size / 2
        points = [
            pos + size * np.array([np.cos(angle), np.sin(angle)]),
            pos + half_size * np.array([np.cos(angle + 2*np.pi/3), np.sin(angle + 2*np.pi/3)]),
            pos + half_size * np.array([np.cos(angle - 2*np.pi/3), np.sin(angle - 2*np.pi/3)])
        ]
        self.surface.fill((0, 0, 0, 0))
        pygame.draw.polygon(self.surface, (*self.color, 38), [(p[0] + size, p[1] + size) for p in points])
        screen.blit(self.surface, (pos[0] - size, pos[1] - size))

    def draw_ray(self, screen, start, end, color=RED):
        pygame.draw.line(screen, (*color, 38), start, end, 2)

    def draw_text(self, screen, text, position, color=BLACK):
        font = pygame.font.SysFont(None, 12)
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, position)

    def draw_trail(self, screen, camera_pos):
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                start_pos = self.trail[i] - camera_pos
                end_pos = self.trail[i + 1] - camera_pos
                pygame.draw.line(screen, (*self.trail_color, 38), start_pos, end_pos, 2)

    def find_closest_intersection(self, ray_dir):
        closest_intersection = None
        min_dist = float('inf')
        wall_color = RED  # Default color

        for wall_start, wall_end, color in self.walls:
            intersection = self.ray_intersect(self.position, ray_dir * self.ray_length, wall_start, wall_end)
            if intersection is not None:
                dist = np.linalg.norm(intersection - self.position)
                if dist < min_dist:
                    min_dist = dist
                    closest_intersection = intersection
                    wall_color = color

        return closest_intersection, min_dist, wall_color

    def ray_intersect(self, ray_origin, ray_dir, wall_start, wall_end):
        x1, y1 = wall_start
        x2, y2 = wall_end
        x3, y3 = ray_origin
        x4, y4 = ray_origin + ray_dir

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and u > 0:
            intersection = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
            return intersection
        else:
            return None

    def line_intersect_circle(self, line_start, line_end, circle_center, circle_radius):
        # Check if the line segment intersects with the circle
        ax, ay = line_start
        bx, by = line_end
        cx, cy = circle_center
        lab = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        dx = (bx - ax) / lab
        dy = (by - ay) / lab
        t = dx * (cx - ax) + dy * (cy - ay)
        ex = t * dx + ax
        ey = t * dy + ay
        lec = np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
        return lec <= circle_radius and min(ax, bx) <= ex <= max(ax, bx) and min(ay, by) <= ey <= max(ay, by)

    def get_inputs(self):
        inputs = []
        for angle in self.ray_angles:
            ray_dir = np.array([np.cos(self.angle + angle), np.sin(self.angle + angle)])
            closest_intersection, min_dist, _ = self.find_closest_intersection(ray_dir)
            inputs.append(min_dist if closest_intersection is not None else self.ray_length)
        inputs.extend([self.position[0], self.position[1], self.velocity[0], self.velocity[1], self.angle])
        return np.array(inputs)

    def neural_move(self):
        if self.alive:
            inputs = self.get_inputs()
            actions, _ = self.brain.decide_action(inputs)
            self.move_forward(actions['thrust_level'])
            self.move_backward(actions['brake_level'])
            if actions['turning'] < 0:
                self.rotate_left(-actions['turning'])
            else:
                self.rotate_right(actions['turning'])

    def update_lifespan(self):
        if self.alive and (time.time() - self.start_time) > self.max_lifespan:
            self.die()
