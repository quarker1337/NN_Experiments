import pygame
import Tetris.game as game
from Tetris.game import max_score,create_grid,get_shape,convert_shape_format,draw_window,draw_next_shape,draw_text_middle,check_lost,valid_space, clear_rows, update_score
import neat
import os
import pickle
import random
import multiprocessing

pygame.font.init()

# GLOBALS VARS
s_width = 800
s_height = 700
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 20 height per block
block_size = 30

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height

win = pygame.display.set_mode((s_width, s_height))

class TetrisGame:
    def __init__(self):
        pass
    def test_ai(self, genome, config):
        pass
    def train_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        last_score = max_score()
        locked_positions = {}
        grid = create_grid(locked_positions)

        change_piece = False
        run = True
        current_piece = get_shape()
        next_piece = get_shape()

        clock = pygame.time.Clock()
        fall_time = 0
        # fall_speed = 0.27 for normal view
        fall_speed = 0.01
        level_time = 0
        score = 0
        piece_counter = 0

        while run:
            pygame.init()
            grid = create_grid(locked_positions)
            fall_time += clock.get_rawtime()
            clock.tick()

            if level_time / 1000 > 5:
                level_time = 0
                if level_time > 0.12:
                    level_time -= 0.005

            if fall_time / 1000 > fall_speed:
                fall_time = 0
                current_piece.y += 1
                if not (valid_space(current_piece, grid)) and current_piece.y > 0:
                    current_piece.y -= 1
                    change_piece = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.display.quit()
                    quit()

            # Use the neural network to determine the movement of the current piece

            # Create the Custom Grid Float to feed as input to the NN
            flattened_list = [item for sublist in grid for item in sublist]
            binary_strings = [tuple_to_binary_string(t) for t in flattened_list]
            binary_representation = "".join(binary_strings)

            grid_result = float(binary_representation)

            # Current Piece Float Generator
            binary_strings = [string_to_binary_string(s) for sublist in current_piece.shape for s in sublist]
            binary_representation = "".join(binary_strings)
            piece_shape_result = float(binary_representation)

            # Current Piece Float Generator
            binary_strings2 = [string_to_binary_string(s) for sublist in next_piece.shape for s in sublist]
            binary_representation2 = "".join(binary_strings2)
            next_shape_result = float(binary_representation2)

            new_grid = [int(sum(T)>0) for T in flattened_list]
            inputs = new_grid + current_piece.vector + next_piece.vector

            #
            # 
            # inputs = (grid_result, piece_shape_result, next_shape_result, score)
            output = net.activate(inputs)
            if output[0] > 1:
                current_piece.x -= 1
                if not (valid_space(current_piece, grid)):
                    current_piece.x += 1
            if output[1] > 1:
                current_piece.x += 1
                if not (valid_space(current_piece, grid)):
                    current_piece.x -= 1
            if output[2] > 1:
                current_piece.y += 1
                if not (valid_space(current_piece, grid)):
                    current_piece.y -= 1
            if output[3] > 1:
                current_piece.rotation += 1
                if not (valid_space(current_piece, grid)):
                    current_piece.rotation -= current_piece.rotation - 1 % len(current_piece.shape)

            shape_pos = convert_shape_format(current_piece)

            for i in range(len(shape_pos)):
                try:
                    x, y = shape_pos[i]
                    if y > -1:
                        if x < 0 or x >= 9:
                            continue
                        grid[y][x] = current_piece.color
                        continue
                except IndexError:
                    # ignore the error and continue
                    continue

            if change_piece:
                for pos in shape_pos:
                    p = (pos[0], pos[1])
                    locked_positions[p] = current_piece.color
                current_piece = next_piece
                next_piece = get_shape()
                change_piece = False
                score += clear_rows(grid, locked_positions) * 10
                piece_counter += 1

            draw_window(win, grid, score, last_score)
            draw_next_shape(next_piece, win)
            pygame.display.update()
            if check_lost(locked_positions):
                draw_text_middle(win, "YOU LOST!", 80, (255, 255, 255))
                pygame.display.update()
                pygame.time.delay(150)
                run = False
                update_score(score)
                self.calculate_fitness(genome, score, piece_counter)
    #pygame.display.quit()
    def calculate_fitness(self, genome, score, piece_counter):
        genome.fitness = (score*100) + piece_counter
        #print(f"Debug: Genome Fitness in calculate_fitness: {genome.fitness}")



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        game = TetrisGame()
        game.train_ai(genome, config)


def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-149')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 150)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

def test_ai(config):
    game2 = TetrisGame()

def tuple_to_binary_string(t):
    return "".join(str(x) for x in t)

def string_to_binary_string(s):
    return "".join("1" if c == "0" else "0" for c in s)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)