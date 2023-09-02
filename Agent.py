import taichi as ti
import taichi.math as tm
import numpy as np

import os
import sys

# Get the parent directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_directory)

from utils.utils import vec2


@ti.dataclass
class TrailAgent:
    # basic attributes
    from_node_ind: ti.i8
    edge_indx: ti.i8
    to_node_ind: ti.i8
    pos: vec2
    vel: ti.f32
    acc: ti.f32
    size: ti.f32
    mass: ti.f32

    # physical properties constrant
    max_acc: ti.f32
    max_spd: ti.f32

    # charge
    charge: ti.f32


@ti.dataclass
class FreeAgent:
    # basic attributes
    pos: vec2
    vel: vec2
    acc: vec2
    size: ti.f32
    mass: ti.f32

    # physical properties constrant
    max_acc: ti.f32
    max_spd: ti.f32
    
    # charge
    charge: ti.f32


@ti.data_oriented
class SwarmAgents:
    def __init__(self,
                 free_n,
                 trail_n: int = 0,
                 num_envs: int = 1,
                 pos=None,
                 vel=None,
                 acc=None,
                 max_acc=None,
                 max_spd=None,
                 alignment_size=0.1,
                 separation_size=0.2,
                 cohesion_size=0.3,
                 front_angle=None,
                 front_size=None,
                 grid_n: int = 10,
                 simulation_size: float = 1.0,
                 focus_number: int=6):
        self.free_n = free_n
        self.trail_n = trail_n
        self.num_envs = num_envs
        self.grid_n = grid_n
        self.simulation_size = simulation_size # simulation size: [0,simulation_size]*[0,simulation_size]
        self.focus_number = focus_number
        # initialize the free agents swarm
        self.free_agents = FreeAgent.field(shape=(self.free_n, self.num_envs))
        self.free_affect_number = ti.field(dtype=ti.i32,
                                           shape=(self.n, self.num_envs, 3),
                                           name="record of num of interaction agents") 
        
        self.init_field(self.free_agents.pos, pos)
        self.init_field(self.free_agents.vel, vel)
        self.init_field(self.free_agents.acc, acc)
        self.init_field(self.free_agents.size, 0.0)
        self.init_field(self.free_agents.mass, 1.0)
        self.init_field(self.free_agents.max_acc, max_acc)
        self.init_field(self.free_agents.max_spd, max_spd)
        self.init_field(self.free_agents.alignment_size, alignment_size)
        self.init_field(self.free_agents.separation_size, separation_size)
        self.init_field(self.free_agents.cohesion_size, cohesion_size)
        self.init_field(self.free_agents.front_angle, front_angle)
        self.init_field(self.free_agents.front_size, front_size)

        # initialize the trail agents swarm
        self.trail_agents = TrailAgent.field(shape=(self.trail_n, self.num_envs))
        self.trail_affect_number = ti.field(dtype=ti.i32,
                                           shape=(self.n, self.num_envs, 3),
                                           name="record of num of interaction agents")

    def init_field(self, property_field, property_value):
        if property_value is not None:
            if isinstance(property_value, np.ndarray):
                property_field.from_numpy(property_value)
            else:
                property_field.from_numpy(
                    np.full(fill_value=property_value, dtype=np.float32, shape=(self.n, self.num_envs)))
        else:
            property_field.fill(0.0)

