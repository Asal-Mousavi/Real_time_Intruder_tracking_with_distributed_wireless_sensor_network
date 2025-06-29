import sys
import pygame
import numpy as np
import heapq
import time
import random

# Constants
CELL_SIZE = 60
MARGIN = 2
FPS = 2
SENSOR_SCORE = 1000
RADIUS_SCORE = 10
DEBUG_VISIBILITY = False

# Color Dictionary
COLORS = {
    'black': (0, 0, 0),
    'gray': (200, 200, 200),
    'light_gray': (160, 160, 160),
    'purple': (160, 32, 240),
    'yellow': (255, 255, 0),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (173, 216, 230)
}


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points


def is_visible(grid, sensor, target):
    for x, y in bresenham_line(sensor[0], sensor[1], target[0], target[1]):
        if grid[x][y] == 'W':
            return False
    return True


def build_visibility_matrix(game, thief_pos):
    matrix = np.full((len(game.sensors), len(thief_pos)), -1)
    for i, sensor in enumerate(game.sensors):
        for j, tpos in enumerate(thief_pos):
            dist = abs(sensor[0] - tpos[0]) + abs(sensor[1] - tpos[1])
            if dist <= game.SENSOR_RADIUS and is_visible(game.map, sensor, tpos):
                matrix[i][j] = 0
    if DEBUG_VISIBILITY:
        print("\nVisibility Matrix (0 = visible, -1 = blocked):")
        print(matrix)
    return matrix


class Game:
    def __init__(self, num_sensors, num_thieves, num_walls, sensor_radius, k_sensors):
        self.SENSOR_RADIUS = sensor_radius
        self.K_SENSORS = k_sensors
        self.map = self.generate_map(10, 10, num_sensors, num_thieves, num_walls, sensor_radius)
        self.sensors = self.get_positions('S')
        self.walls = self.get_positions('W')
        self.goal = self.get_positions('G')[0]
        self.thieves = [Thief(tid, pos, self.goal) for tid, pos in enumerate(self.get_positions('T'))]
        self.communication_matrix = self.build_communication_matrix(self.sensors)
        self.score_map = self.build_score_map(10, 10)

    def get_positions(self, target):
        return [(i, j) for i in range(len(self.map)) for j in range(len(self.map[0])) if self.map[i][j] == target]

    def build_score_map(self, rows, cols):
        score = np.ones((rows, cols))

        for sx, sy in self.sensors:
            score[sx][sy] = SENSOR_SCORE

            for dx in range(-self.SENSOR_RADIUS, self.SENSOR_RADIUS + 1):
                for dy in range(-self.SENSOR_RADIUS, self.SENSOR_RADIUS + 1):
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < rows and 0 <= ny < cols and self.map[nx][ny] not in ['S', 'W']:
                        dist = abs(dx) + abs(dy)
                        if dist <= self.SENSOR_RADIUS and is_visible(self.map, (sx, sy), (nx, ny)):
                            score[nx][ny] += (RADIUS_SCORE - dist)

        print("\nScore Map:")
        print(score)
        return score

    def build_communication_matrix(self, sensors):
        n = len(sensors)
        matrix = np.zeros((n, n), dtype=int)
        for i, (x1, y1) in enumerate(sensors):
            for j, (x2, y2) in enumerate(sensors):
                if i != j:
                    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if dist <= self.SENSOR_RADIUS:
                        matrix[i][j] = 1
        print("\nCommunication Matrix:")
        print(matrix)
        return matrix

    def generate_map(self, rows, cols, num_sensors, num_thieves, num_walls, sensor_radius, min_comm_pairs=2):
        grid = [['.' for _ in range(cols)] for _ in range(rows)]
        occupied = set()

        def random_empty_cell():
            while True:
                x, y = random.randint(0, rows - 1), random.randint(0, cols - 1)
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    return (x, y)

        def place_sensors():
            sensors = []
            for _ in range(100):
                sensors.clear()
                occupied.clear()

                local_sensors = [random_empty_cell() for _ in range(num_sensors)]
                comm_matrix = self.build_communication_matrix(local_sensors)
                connected_pairs = sum(
                    comm_matrix[i][j] for i in range(len(comm_matrix)) for j in range(i + 1, len(comm_matrix))
                )

                if connected_pairs >= min_comm_pairs:
                    sensors.extend(local_sensors)
                    for x, y in sensors:
                        grid[x][y] = 'S'
                        occupied.add((x, y))
                    return

        def place_walls():
            for _ in range(num_walls):
                x, y = random_empty_cell()
                grid[x][y] = 'W'

        def place_goal():
            gx, gy = random_empty_cell()
            grid[gx][gy] = 'G'

        def place_thieves():
            edge_cells = [(i, j) for i in range(rows) for j in range(cols)
                          if i == 0 or i == rows - 1 or j == 0 or j == cols - 1]
            random.shuffle(edge_cells)
            thieves_placed = 0
            for x, y in edge_cells:
                if (x, y) not in occupied:
                    grid[x][y] = 'T'
                    occupied.add((x, y))
                    thieves_placed += 1
                if thieves_placed >= num_thieves:
                    break

        place_sensors()
        place_walls()
        place_goal()
        place_thieves()

        print("\nGenerated Game Map:")
        for row in grid:
            print(' '.join(row))

        return grid


class Thief:
    def __init__(self, id, start, goal):
        self.id = id
        self.start = start
        self.goal = goal
        self.path = []
        self.frozen = False
        self.seen_by = set()

    def find_path(self, game):
        self.path = astar(game.map, self.start, self.goal, game.score_map)
        print(f"Thief {self.id} path: {self.path}")


def astar(grid, start, goal, score_map):
    rows, cols = len(grid), len(grid[0])
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 'W':
                tentative = g_score[current] + score_map[nx][ny]
                if tentative < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = tentative
                    heapq.heappush(open_set, (tentative + heuristic((nx, ny), goal), (nx, ny)))
                    came_from[(nx, ny)] = current
    return []


def run_game(game):
    pygame.init()
    screen = pygame.display.set_mode((len(game.map[0]) * (CELL_SIZE + 2), len(game.map) * (CELL_SIZE + 2)))
    pygame.display.set_caption("Sensor-Thief Tracker")
    clock = pygame.time.Clock()

    spawn_delay = 2
    base_time = time.time()
    thief_start_times = [base_time + i * spawn_delay for i in range(len(game.thieves))]
    step_indices = [0] * len(game.thieves)

    for thief in game.thieves:
        thief.find_path(game)

    def handle_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def update_thieves(current_time, vis_matrix):
        for t_id, thief in enumerate(game.thieves):
            if current_time < thief_start_times[t_id]:
                continue

            if thief.frozen:
                draw_thief(screen, thief.path[step_indices[t_id] - 1], COLORS['blue'], game)
                continue

            if step_indices[t_id] >= len(thief.path):
                continue

            pos = thief.path[step_indices[t_id]]
            draw_thief(screen, pos, COLORS['red'], game)
            step_indices[t_id] += 1

            seen_by = {i for i in range(len(game.sensors)) if vis_matrix[i][t_id] == 0}
            thief.seen_by.update(seen_by)

            for sid in seen_by:
                sx, sy = game.sensors[sid]
                pygame.draw.line(screen, COLORS['red'],
                                 (sy * (CELL_SIZE + 2) + CELL_SIZE // 2, sx * (CELL_SIZE + 2) + CELL_SIZE // 2),
                                 (pos[1] * (CELL_SIZE + 2) + CELL_SIZE // 2, pos[0] * (CELL_SIZE + 2) + CELL_SIZE // 2),
                                 2)

            if len(thief.seen_by) >= game.K_SENSORS:
                thief.frozen = True
                print(f"Thief {t_id} frozen at {pos} (seen by {len(thief.seen_by)} sensors)")

    running = True
    while running:
        screen.fill(COLORS['gray'])
        running = handle_events()

        draw_map(screen, game)

        thief_positions = [t.path[min(step_indices[i], len(t.path) - 1)] for i, t in enumerate(game.thieves)]
        vis_matrix = build_visibility_matrix(game, thief_positions)
        current_time = time.time()

        update_thieves(current_time, vis_matrix)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def draw_map(screen, game):
    for i in range(len(game.map)):
        for j in range(len(game.map[0])):
            rect = pygame.Rect(j * (CELL_SIZE + 2), i * (CELL_SIZE + 2), CELL_SIZE, CELL_SIZE)
            color = COLORS['light_gray'] if game.map[i][j] == 'W' else COLORS['black']
            pygame.draw.rect(screen, color, rect)
            if game.map[i][j] == 'S':
                pygame.draw.circle(screen, COLORS['purple'], rect.center, CELL_SIZE // 3)
            elif game.map[i][j] == 'G':
                pygame.draw.rect(screen, COLORS['yellow'], rect.inflate(-4, -4))


def draw_thief(screen, pos, color, game):
    cx = pos[1] * (CELL_SIZE + 2) + CELL_SIZE // 2
    cy = pos[0] * (CELL_SIZE + 2) + CELL_SIZE // 2
    pygame.draw.circle(screen, color, (cx, cy), CELL_SIZE // 3)


def get_user_input():
    pygame.init()
    font = pygame.font.Font(None, 32)
    header_font = pygame.font.Font(None, 40)
    screen = pygame.display.set_mode((520, 380))
    pygame.display.set_caption("Game Setup")

    params = ["Sensor Radius", "K Sensors to Freeze", "Number of Sensors", "Number of Thieves", "Number of Walls"]
    values = [""] * len(params)
    active = [False] * len(params)
    input_boxes = [pygame.Rect(260, 60 + i * 50, 160, 36) for i in range(len(params))]
    submit_button = pygame.Rect(180, 330, 160, 36)

    while True:
        screen.fill((25, 25, 35))

        title = header_font.render("Sensor-Thief Tracker Setup", True, COLORS['yellow'])
        screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 15))

        # Inputs
        for i, box in enumerate(input_boxes):
            pygame.draw.rect(screen, COLORS['light_gray'], box, 2)
            txt = font.render(params[i], True, COLORS['gray'])
            screen.blit(txt, (40, box.y + 5))
            val = font.render(values[i], True, COLORS['green'])
            screen.blit(val, (box.x + 5, box.y + 5))

        # Button
        pygame.draw.rect(screen, COLORS['purple'], submit_button)
        btn_txt = font.render("Start Game", True, COLORS['yellow'])
        screen.blit(btn_txt, (submit_button.x + 25, submit_button.y + 5))

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if submit_button.collidepoint(event.pos):
                    try:
                        return list(map(int, values))
                    except ValueError:
                        pass
                active = [box.collidepoint(event.pos) for box in input_boxes]
            elif event.type == pygame.KEYDOWN:
                for i in range(len(values)):
                    if active[i]:
                        if event.key == pygame.K_BACKSPACE:
                            values[i] = values[i][:-1]
                        elif event.unicode.isdigit():
                            values[i] += event.unicode

        pygame.display.flip()



if __name__ == "__main__":
    SENSOR_RADIUS, K_SENSORS, num_sensors, num_thieves, num_walls = get_user_input()
    game = Game(num_sensors, num_thieves, num_walls, SENSOR_RADIUS, K_SENSORS)
    run_game(game)
