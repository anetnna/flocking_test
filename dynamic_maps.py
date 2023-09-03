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

from utils.utils import vec2, interpolation_all


@ti.data_oriented
class Dynamic_maps:
    def __init__(self,
                 grid_n,
                 max_size,
                 env_num):
        self.grid_n = grid_n # n columns * n rows
        self.max_size = max_size
        self.grid_length = self.max_size / self.grid_n
        self.env_num = env_num
        self.field_map = ti.field(dtype=ti.f32,
                                  shape=(self.grid_n, self.grid_n, self.env_num),
                                  name="dynamics field maps")

        self.bound_map = ti.field(dtype=ti.f32,
                                  shape=(self.grid_n, self.grid_n, self.env_num),
                                  name="dynamics field maps")
    
    @ti.kernel
    def get_boung_map(self, input_maps: ti.template()):
        interpolation_all(self.bound_map, input_maps, bilinear=1)
    
    @ti.kernel
    def dynamic_rule(self, h: float):
        """ descript the dynamic rules, not implemented here, please implement in the child class
            
            Args:
                h: float, describing the time interval
            Comments:
                
        """
        pass


# @ti.dataclass
# class Diffusion:
#     q_s: ti.f32 # release strength
#     U: ti.f32 # average wind speed
#     D: ti.f32 # isotropic diffusivity
#     psi: ti.f32 # wind direction
#     tau: ti.f32 # particle lifetime

#     @ti.func
#     def consentration(self, p, p_s, delta_t): # p: sensor position   p_s: plume source position   delta_t: duration time
#         length = ti.length(p-p_s)
#         lambda_symbol = ti.sqrt(self.D * self.tau/(1 + (self.U**2)*self.tau/(4*self.D)))
#         delta_y = -(p[0]-p_s[0]) * tm.sin(self.psi) + (p[1]-p_s[1]) * tm.cos(self.psi)
#         m = self.q_s/(4 * tm.pi * self.D * length) * tm.exp((-delta_y * self.U / (2 * self.D))+(-length * delta_t / lambda_symbol))
#         # m: mean gas concentration at p
#         return m
