"""Implementation of the race."""

import core
import neat
import random
import tkinter


class Checkpoint:
    """A circle which represents checkpoint."""

    CHECK_RADIUS = 600
    BORDER_COLOR = "magenta"
    FONT = "Courier 18"
    ACTIVE_COLOR = "lime"
    MIN_CHECKPOINTS = 3
    MAX_CHECKPOINTS = 6
    CHECKPOINTS_PADDING = 600

    @classmethod
    def create_checkpoints(cls, canvas, width, height, norm):
        """Create a list which contains random number of Checkpoint objects,
        between min_checkpoints and max_checkpoints, located at random
        positions, not intersecting and padded.
        """
        checkpoints = []
        padding_r = cls.CHECKPOINTS_PADDING + cls.CHECK_RADIUS
        quads = [
            (x, y, x + width / 2 - padding_r, y + height / 2 - padding_r)
            for x in [padding_r, width / 2]
            for y in [padding_r, height / 2]
        ]
        for i in range(random.randint(cls.MIN_CHECKPOINTS, cls.MAX_CHECKPOINTS)):
            if quads:
                quad = random.choice(quads)
                min_x, min_y, max_x, max_y = quad
                quads.remove(quad)
            else:
                min_x, min_y = padding_r, padding_r
                max_x = width - padding_r
                max_y = height - padding_r
            while True:
                x = random.randint(min_x, max_x)
                y = random.randint(min_y, max_y)
                new_pos = core.Vector(x, y)
                for point in checkpoints:
                    if point.cross(new_pos, padding_r):
                        break
                else:
                    break
            checkpoint = cls(canvas, norm, i, new_pos)
            checkpoints.append(checkpoint)
        return checkpoints

    def __init__(self, canvas, norm, index, pos=core.Vector(0, 0)):
        self.canvas = canvas
        self.pos = pos
        self.norm = norm
        self.index = index
        self.body = None

    def draw(self, color=None):
        """Create the circle with the checkpoint's index on the canvas."""
        if color is None:
            color = self.BORDER_COLOR
        v1 = self.pos - core.Vector(self.CHECK_RADIUS, self.CHECK_RADIUS)
        v2 = self.pos + core.Vector(self.CHECK_RADIUS, self.CHECK_RADIUS)
        v1 /= self.norm
        v2 /= self.norm
        self.body = [self.canvas.create_oval(v1.x, v1.y, v2.x, v2.y, outline=color)]
        self.body += [
            self.canvas.create_text(
                self.pos.x / self.norm,
                self.pos.y / self.norm,
                text=str(self.index),
                justify=tkinter.CENTER,
                font=self.FONT,
                fill=color,
            )
        ]

    def hide(self):
        """Delete the circle from the canvas."""
        for item in self.body:
            self.canvas.delete(item)

    def cross(self, pos, r):
        """Test if the checkpoint intersects a circle with radius r and center
        at pos.
        """
        d = self.pos - pos
        return d.length <= r + self.CHECK_RADIUS

    def activate(self):
        """Turn the checkpoint into ACTIVE_COLOR."""
        self.hide()
        self.draw(self.ACTIVE_COLOR)

    def deactivate(self):
        """Turn the checkpoint back to the BORDER_COLOR."""
        self.hide()
        self.draw()


class Rocket(core.Vehicle):
    TIMEOUT = 100
    MAX_THRUST = 100  # Applied before movement
    FRICTION = 0.15  # Friction force is rounded up; subtracted after movement.
    POD_RADIUS = 400  # For collisions, not implemented.
    MAX_YAW = 18
    ALIVE_COLOR = "lime"
    DEAD_COLOR = "red"
    FINISHED_COLOR = "cyan"

    @classmethod
    def create_rockets(
        cls, canvas, norm, pos, dir_, genomes, config, min_dost_to_next_checkpoint
    ):
        rockets = []
        for _, genome in genomes:
            rocket = Rocket(
                canvas, norm, genome, config, min_dost_to_next_checkpoint, pos, dir_
            )
            rockets.append(rocket)
        return rockets

    def __init__(
        self,
        canvas,
        norm,
        genome,
        config,
        min_dist_to_next_checkpoint,
        pos=core.Vector(0, 0),
        direction=core.Vector(0, -1),
    ):
        self.canvas = canvas
        self.norm = norm
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.timeout = self.TIMEOUT
        self.next_checkpoint_i = 1
        self.lap = 0
        self.steps = 0
        self.passed_checkpoints = 0
        self.min_dist_to_next_checkpoint = min_dist_to_next_checkpoint
        self.body = None
        self.color = self.__class__.ALIVE_COLOR
        super(self.__class__, self).__init__(
            pos, direction, self.FRICTION, self.MAX_THRUST, self.MAX_YAW
        )

    def draw(self):
        """Create a triangle on the canvas which represents the rocket."""
        dir_ = self.direction.normalized
        vec1 = self.POD_RADIUS * dir_
        vec2 = self.POD_RADIUS * dir_.rotated(135)
        vec3 = self.POD_RADIUS * dir_.rotated(-135)
        v1 = self.pos + vec1
        v2 = self.pos + vec2
        v3 = self.pos + vec3
        v1 /= self.norm
        v2 /= self.norm
        v3 /= self.norm
        self.body = self.canvas.create_polygon(
            v1.x, v1.y, v2.x, v2.y, v3.x, v3.y, fill=self.color
        )

    def hide(self):
        """Delete the circle from the canvas."""
        self.canvas.delete(self.body)

    def move(self, x, y, thrust):
        """Move the vehicle, using desired force, reduce timeout, redraw it and
        draw the path.
        """
        super(self.__class__, self).move(x, y, thrust)
        self.timeout -= 1
        self.hide()
        self.draw()

    def brain(self, next_checkpoint, width, height):
        """Calculate and return the desired force."""
        rel_x = (next_checkpoint.pos.x - self.pos.x) / width
        rel_y = (next_checkpoint.pos.y - self.pos.y) / height
        rel_prev_x = (self.pos.x - self.prev_pos.x) / width
        rel_prev_y = (self.pos.y - self.prev_pos.y) / height
        next_checkpoint_angle = self.direction.angle(next_checkpoint.pos - self.pos)
        rel_next_checkpoint_angle = next_checkpoint_angle / 180
        input_ = [rel_x, rel_y, rel_prev_x, rel_prev_y, rel_next_checkpoint_angle]
        output = self.net.activate(input_)
        rel_out_x, rel_out_y, rel_thrust = output
        out_x = rel_out_x * width + self.pos.x
        out_y = rel_out_y * height + self.pos.y
        thrust = rel_thrust * self.MAX_THRUST
        return int(round(out_x)), int(round(out_y)), int(round(thrust))

    def calc_fitness(
        self,
        max_laps,
        checkpoints_per_lap,
        dist_to_next_checkpoint=None,
        track_length=None,
    ):
        """Calculate rocket.genome.fitness -- fraction of the completed course,
        adding the inverse of the steps.
        """
        self.genome.fitness = self.passed_checkpoints / checkpoints_per_lap / max_laps
        if self.lap >= max_laps:
            self.genome.fitness += track_length / self.steps
        else:
            self.genome.fitness += (
                self.min_dist_to_next_checkpoint
                / dist_to_next_checkpoint
                / max_laps
                / checkpoints_per_lap
            )


class Game(core.RootWindow):
    """A window of the race game."""

    TITLE = "Race"
    BG = "black"
    FG = "lime"
    FONT = "Courier 18"
    Y_OFFSET = 60
    WIDTH = 19200
    HEIGHT = 10800
    NORMALIZE = 10  # Coefficient to normalize canvas's size
    DEFAULT_FPS = 1000
    TOGGLED_FPS = 30
    MAX_LAPS = 2

    def __init__(self):
        super(self.__class__, self).__init__(
            self.TITLE, self.BG, self.FG, self.FONT, self.Y_OFFSET
        )
        self.fps = self.DEFAULT_FPS
        canvas_width = int(self.WIDTH / self.NORMALIZE)
        canvas_height = int(self.HEIGHT / self.NORMALIZE)
        self.create_canvas(canvas_width, canvas_height, 0, 0)
        fps_button_text = "FPS: {}".format(self.DEFAULT_FPS)
        self.create_button(fps_button_text, self._toggle_fps, "fps", 1, 0)
        self.generation = 1
        self.checkpoints = None
        self.rockets = None
        self.config = None
        self.track_lengths = []

    def _toggle_fps(self):
        """Toggle fps between DEFAULT_FPS and TOGGLED_FPS, update the text on
        the corresponding button.
        """
        if self.fps == self.DEFAULT_FPS:
            self.fps = self.TOGGLED_FPS
        else:
            self.fps = self.DEFAULT_FPS
        self.buttons["fps"]["text"] = "FPS: {}".format(self.fps)

    def _reset(self, genomes, config):
        """Clear canvas, create checkpoints and rockets."""
        self.canvas.delete(tkinter.ALL)
        self.checkpoints = Checkpoint.create_checkpoints(
            self.canvas, self.WIDTH, self.HEIGHT, self.NORMALIZE
        )
        for checkpoint in self.checkpoints:
            checkpoint.draw()
        self.track_lengths = self._calc_track_lengths()
        dir_ = self.checkpoints[1].pos - self.checkpoints[0].pos
        self.rockets = Rocket.create_rockets(
            self.canvas,
            self.NORMALIZE,
            self.checkpoints[0].pos,
            dir_,
            genomes,
            config,
            self.track_lengths[0],
        )
        for rocket in self.rockets:
            rocket.draw()
        generation_text = "Generation: {}".format(self.generation)
        self.canvas.create_text(
            10,
            10,
            text=generation_text,
            anchor=tkinter.NW,
            font=self.FONT,
            fill=self.FG,
        )

    def _calc_track_lengths(self):
        track_lengths = []
        prev_checkpoint = self.checkpoints[-1]
        for checkpoint in self.checkpoints:
            track_lengths.insert(-1, (checkpoint.pos - prev_checkpoint.pos).length)
            prev_checkpoint = checkpoint
        return track_lengths

    def _step(self):
        """Check if rockets reached their checkpoints and completed MAX_LAPS,
        update next_checkpoints and move rockets.
        """
        for rocket in self.rockets[:]:
            rocket.steps += 1
            next_checkpoint = self.checkpoints[rocket.next_checkpoint_i]
            if next_checkpoint.cross(rocket.pos, 0):
                rocket.passed_checkpoints += 1
                rocket.min_dist_to_next_checkpoint = self.track_lengths[
                    rocket.next_checkpoint_i
                ]
                if rocket.next_checkpoint_i == 0:
                    rocket.lap += 1
                    if rocket.lap >= self.MAX_LAPS:
                        rocket.calc_fitness(
                            self.MAX_LAPS,
                            len(self.checkpoints),
                            track_length=sum(self.track_lengths),
                        )
                        self.rockets.remove(rocket)
                        rocket.color = Rocket.FINISHED_COLOR
                rocket.timeout = rocket.TIMEOUT
                rocket.next_checkpoint_i += 1
                rocket.next_checkpoint_i %= len(self.checkpoints)
                next_checkpoint = self.checkpoints[rocket.next_checkpoint_i]
            elif rocket.timeout <= 0:
                rocket.calc_fitness(
                    self.MAX_LAPS,
                    len(self.checkpoints),
                    self.track_lengths[rocket.next_checkpoint_i - 1],
                )
                self.rockets.remove(rocket)
                rocket.color = Rocket.DEAD_COLOR
            else:
                rocket.min_dist_to_next_checkpoint = min(
                    rocket.min_dist_to_next_checkpoint,
                    (rocket.pos - next_checkpoint.pos).length,
                )
            x, y, thrust = rocket.brain(next_checkpoint, self.WIDTH, self.HEIGHT)
            rocket.move(x, y, thrust)
        self.root.update()

    def run(self, genomes, config):
        """Reset (clear canvas, create checkpoints, rockets) and run the game.
        """
        self._reset(genomes, config)
        self.generation += 1
        self.center_window()
        while self.rockets:
            step_size_ns = 1_000_000_000 // self.fps
            core.exec_for_time(step_size_ns, self._step)
