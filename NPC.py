import taichi as ti
import taichi.math as tm

import os
import sys

# Get the parent directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_directory)

from utils.utils import vec2

@ti.dataclass
class NPC:
    # phycial attribute
    pos: vec2
    vel: vec2
    acc: vec2
    interaction_zone: ti.f32
    movable: ti.int8
    attached_agent_id: ti.i32




