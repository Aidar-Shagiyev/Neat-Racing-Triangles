"""Run the training races."""

import race
import neat
import os

race = race.Game()

local_dir = os.path.dirname(".")
config_path = os.path.join(local_dir, "config.ini")
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)
p = neat.Population(config)
stats = neat.StatisticsReporter()
p.add_reporter(stats)

winner = p.run(race.run, 100)

# Visualize
#
# node_names = {-1: 'Rel X', -2: 'Rel Y', -3: 'Rel Prev X', -4: 'Rel Prev Y',
#               -5: 'CP angle', 0: 'Rel Next X', 1: 'Rel Next Y', 2: 'Thrust'}
# neat.visualize.draw_net(config, winner, True, node_names=node_names,
#                         show_disabled=False)
# neat.visualize.plot_stats(stats, ylog=False, view=True)
# neat.visualize.plot_species(stats, view=True)
