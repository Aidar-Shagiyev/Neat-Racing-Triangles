# Neat-Racing-Triangles
Use reinforcement learning (specifically [NEAT](https://neat-python.readthedocs.io/en/latest/)) to teach little triangles how to race. 

The triangles can control their rotation speed and thrust. Their goal is to complete two laps (which consist of purple numbered checkpoints) as fast as possible. The triangles that do not pass any checkpoint for a long time die (freeze and become red). The triangles that complete two laps also freeze, but are colored cyan.

During development I focused more on building fun stuff myself and being pythonic rather than optimizing for execution time (hence the silly Vector class and no numpy whatsoever).

## Getting Started
### Prerequisites
  * Python 3.4+
  * neat-python 0.92+

### Running the race
To run the simulation just execute run.py:
```
git clone https://github.com/Aidar-Shagiyev/Neat-Racing-Triangles
cd Neat-Racing-Triangles/game
python run.py
```
This will result in something like this:
![](race.gif)
