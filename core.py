"""Core classes and functions for the game."""

import math
import tkinter
import time


def exec_for_time(time_of_exec, func, *args):
    """Execute the callable func with arguments *args for time_of_exec (in ns)."""
    t0 = time.time_ns()
    func(*args)
    t1 = time.time_ns()
    diff = time_of_exec - (t1 - t0)
    time.sleep(max(0, diff) / 1000000000)


class Vector:
    """Two dimensional vector."""

    @staticmethod
    def distance(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx ** 2 + dy ** 2)

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __str__(self):
        return "Vector({}, {})".format(self.x, self.y)

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            x_mul = self.x * other.x
            y_mul = self.y * other.y
            return x_mul + y_mul
        else:
            x = self.x * other
            y = self.y * other
            return self.__class__(x, y)

    def __rmul__(self, number):
        return self * number

    def __truediv__(self, number):
        x = self.x / number
        y = self.y / number
        return self.__class__(x, y)

    def __add__(self, other_vector):
        x = self.x + other_vector.x
        y = self.y + other_vector.y
        return self.__class__(x, y)

    def __sub__(self, other_vector):
        return self + (-1 * other_vector)

    def int_up(self):
        """Round self coordinates up and convert to int."""
        self.x = math.ceil(self.x)
        self.y = math.ceil(self.y)

    def int_round(self):
        """Round self coordinates and convert to int."""
        self.x = int(round(self.x))
        self.y = int(round(self.y))

    def rotated(self, angle):
        """Return Vector, which is rotated clockwise by angle (in degrees)."""
        angle = math.radians(angle)
        new_x_axis = Vector(math.cos(angle), -math.sin(angle))
        new_y_axis = Vector(math.sin(angle), math.cos(angle))
        new_x = self.x * new_x_axis.x + self.y * new_y_axis.x
        new_y = self.x * new_x_axis.y + self.y * new_y_axis.y
        return Vector(new_x, new_y)

    def angle(self, other_vector):
        """Measure the angle between self and another Vector."""
        if self.length == 0 or other_vector.length == 0:
            return 0
        cos = (self * other_vector) / self.length / other_vector.length
        cos = min(cos, 1)
        cos = max(cos, -1)
        angle = math.acos(cos)
        angle = math.degrees(angle)
        return angle

    @property
    def length(self):
        return self.distance(0, 0, self.x, self.y)

    @property
    def normalized(self):
        """Return Vector with the same direction as self, but with the length of 1."""
        return self / self.length


class RootWindow:
    """Wrapper for tkinter.Tk object."""

    def __init__(self, title, bg, fg, font, y_offset):
        self.root = tkinter.Tk()
        self.root.title(title)
        self.root["bg"] = bg
        self.bg = bg
        self.fg = fg
        self.font = font
        self.y_offset = y_offset
        self.canvas = None
        self.buttons = {}

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        width = int(self.canvas["width"]) + 4
        x = int((screen_width / 2) - (width / 2))
        self.root.geometry("+{}+{}".format(x, self.y_offset))

    def create_canvas(self, width, height, row=0, col=0, rowspan=1, colspan=1):
        """Create tkinter.Canvas object and put it in the window."""
        self.canvas = tkinter.Canvas(self.root, width=width, height=height, bg=self.bg)
        self.canvas.grid(
            row=row, column=col, rowspan=rowspan, columnspan=colspan, sticky="nsew"
        )

    def create_button(
        self, text, command, command_name, row, col, rowspan=1, colspan=1
    ):
        """Create tkinter.Button object and put it in the window."""
        button = tkinter.Button(
            self.root,
            text=text,
            font=self.font,
            command=command,
            bg=self.bg,
            fg=self.fg,
        )
        button.grid(
            row=row, column=col, rowspan=rowspan, columnspan=colspan, sticky="nsew"
        )
        self.buttons[command_name] = button


class Vehicle:
    """A class to implement simple vehicle's physics."""

    def __init__(
        self,
        pos=Vector(0, 0),
        direction=Vector(0, -1),
        friction_c=0.0,
        max_thrust=float("inf"),
        max_yaw=float("inf"),
    ):
        self.pos = pos
        self.vel = Vector(0, 0)
        self.acc = Vector(0, 0)
        self.direction = direction
        self.friction_c = friction_c
        self.max_thrust = max_thrust
        self.max_yaw = max_yaw
        self.prev_pos = pos

    def _apply_force(self, x, y, thrust):
        """Apply thrust in the direction (x, y) taking into account limitations
        of max_thrust and max_yaw.
        """
        thrust = min(thrust, self.max_thrust)
        thrust = max(0, thrust)
        dir_vec = Vector(x, y) - self.pos
        angle = dir_vec.angle(self.direction)
        if angle > self.max_yaw:
            if self.direction.rotated(angle).angle(dir_vec) < 0.01:
                self.direction = self.direction.rotated(self.max_yaw)
            else:
                self.direction = self.direction.rotated(-self.max_yaw)
        force = self.direction.normalized * thrust
        force.int_round()
        self.acc += force

    def move(self, x, y, thrust):
        """Apply desirable force, move the vehicle and apply friction."""
        self._apply_force(x, y, thrust)
        self.vel += self.acc
        self.prev_pos = self.pos
        self.pos += self.vel
        friction = -1 * (self.friction_c * self.vel)
        friction.int_up()
        self.vel += friction
        self.acc = Vector(0, 0)
