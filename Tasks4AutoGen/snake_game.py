"""
AutoGen Code:
    # pip install ag2

    from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

    # Load LLM config from an env variable or a file
    config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
    assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
    user_proxy= UserProxyAgent("user_proxy", code_execution_config={'work_dir': 'coding'})
    user_proxy.initiate_chat(assistant, message="Write a snake game in python.")
    # This initiates an automated chat between two agents to solve the task

The Question:
    Write a snake game in python.

The Answer:
"""

import random
import time

# Import necessary modules
import pygame

# Initialize Pygame
pygame.init()

# Define game constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
BLOCK_SIZE = 20  # Size of each block of the snake
FPS = 10  # Frames per second

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Create game window
game_window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")


# Define Snake class
class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.color = GREEN

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = ((cur[0] + (x * BLOCK_SIZE)), (cur[1] + (y * BLOCK_SIZE)))
        if new in self.positions[2:]:
            self.reset()
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()

    def reset(self):
        self.length = 1
        self.positions = [(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BLACK, r, 1)

    def collide(self, food_pos):
        if self.get_head_position() == food_pos:
            self.length += 1
            return True
        else:
            return False


# Define Food class
class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (
            random.randint(0, (SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
            random.randint(0, (SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
        )

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, BLACK, r, 1)


# Define main function
def main():
    snake = Snake()
    food = Food()

    clock = pygame.time.Clock()

    # Game loop
    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                elif event.key == pygame.K_LEFT:
                    snake.turn((-1, 0))
                elif event.key == pygame.K_RIGHT:
                    snake.turn((1, 0))
                elif event.key == pygame.K_UP:
                    snake.turn((0, -1))
                elif event.key == pygame.K_DOWN:
                    snake.turn((0, 1))

        snake.move()
        if snake.collide(food.position):
            food.randomize_position()
        if (
            snake.get_head_position()[0] < 0
            or snake.get_head_position()[0] > SCREEN_WIDTH - BLOCK_SIZE
            or snake.get_head_position()[1] < 0
            or snake.get_head_position()[1] > SCREEN_HEIGHT - BLOCK_SIZE
        ):
            snake.reset()

        # Draw game screen
        game_window.fill(WHITE)
        snake.draw(game_window)
        food.draw(game_window)
        pygame.display.update()


# Run the game
main()
