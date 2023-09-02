import taichi as ti
import taichi.math as tm

import numpy as np
import yaml
import json

import os
import sys

# Get the parent directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_directory)

from utils.utils import vec2


@ti.dataclass
class Node:
    idx: ti.i8       # node index (should be the same with index in Trails.nodes)
    pos: vec2        # node position
    attribute: ti.i8 # node attribute
    # linked_edge_index: ti.Vector(10, ti.i32) # indexes of linked edges to node


@ti.dataclass
class Edge:
    s_node_idx: ti.i8   # start node index
    e_node_idx: ti.i8   # end node index
    s_pos: vec2         # start node position
    e_pos: vec2         # end node position
    s_con: vec2         # start control point position (related to s_pos)
    e_con: vec2         # end control point position (related to e_pos)
    line_state: ti.f32  # 0: beizer line; 1: direct line.
    line_length: ti.f32 # length of edge line

    @ti.func
    def get_t_pos(self, t):
        bezier_pos = (1 - t)**3 * self.s_pos + 3 * (1 - t)**2 * t * self.s_con + 3 * (1 - t) * t**2 * self.e_con + t**3 * self.e_pos
        line_pos = (self.e_pos - self.s_pos) * t + self.s_pos
        return bezier_pos * self.line_state + line_pos * (1 - self.line_state)

    @ti.func
    def get_line_length(self, point_count=10):
        direct_line_length = tm.length(self.e_pos - self.s_pos)
        bezier_length = self.get_bezier_length(point_count)
        return direct_line_length * self.line_state + bezier_length * (1 - self.line_state)

    @ti.func
    def get_bezier_length(self, point_count=10):
        length = 0.0
        last_point = self.get_t_pos(0.0)
        for i in range(1, point_count+1):
            t = (i*1.0) / point_count
            point = self.get_t_pos(t)
            length += tm.length(point - last_point)
            last_point = point
        return length


@ti.data_oriented
class Trails:
    def __init__(self, nodes_num, edges_num, env_num, max_size=1, max_edges_2_node=10):
        self.nodes_num = nodes_num
        self.edges_num = edges_num
        self.env_num = env_num
        self.max_edges_2_node = max_edges_2_node
        self.max_size = max_size
        self.color = 0xa0a0a0
        self.a_matrix = ti.field(dtype=ti.i8,
                                 shape=(self.nodes_num, self.nodes_num, self.env_num),
                                 name="adjent matrix")
        self.nodes = Node.field(shape=(self.nodes_num, 
                                       self.env_num))
        self.edges = Edge.field(shape=(self.edges_num, 
                                       self.env_num))
        self.nodes_2_edges_s = ti.field(dtype=ti.i8,
                                        shape=(self.nodes_num, self.max_edges_2_node, self.env_num),
                                        name="index of egdes on nodes (out)")
        self.nodes_2_edges_e = ti.field(dtype=ti.i8,
                                        shape=(self.nodes_num, self.max_edges_2_node, self.env_num),
                                        name="index of egdes on nodes (in)")
    
    @ti.kernel
    def generate_trails(self, 
                        node_pos_vector: ti.template(), 
                        nodes_attribute: ti.template(),
                        edges_ind_s: ti.template(),
                        edges_ind_e: ti.template(),
                        edges_pos_s: ti.template(),
                        edges_pos_e: ti.template(),
                        edges_con_s: ti.template(),
                        edges_con_e: ti.template(),
                        edges_line_state: ti.template()): 
        for i, j in ti.ndrange(self.nodes_num, self.env_num):
            self.nodes[i, j].idx = i
            self.nodes[i, j].pos = vec2(node_pos_vector[i, j, 0], node_pos_vector[i, j, 1])
            self.nodes[i, j].attribute = nodes_attribute[i, j]
        
        for i, j, k in ti.ndrange(self.edges_num, self.env_num, 2):
            self.edges[i, j].s_pos[k] = edges_pos_s[i, j, k]
            self.edges[i, j].e_pos[k] = edges_pos_e[i, j, k]
            self.edges[i, j].s_con[k] = edges_con_s[i, j, k] + edges_pos_s[i, j, k]
            self.edges[i, j].e_con[k] = edges_con_e[i, j, k] + edges_pos_e[i, j, k]     
        
        for i, j in ti.ndrange(self.edges_num, self.env_num):
            self.edges[i, j].s_node_idx = edges_ind_s[i, j]
            self.edges[i, j].e_node_idx = edges_ind_e[i, j]
            self.edges[i, j].line_state = edges_line_state[i, j]
            self.edges[i, j].line_length = self.edges[i, j].get_line_length(1)
            self.a_matrix[self.edges[i, j].s_node_idx, self.edges[i, j].e_node_idx, j] = 1

        for i, j in ti.ndrange(self.nodes_num, self.env_num):
            node_2_edge_s_ind = 0
            node_2_edge_e_ind = 0
            for k in range(self.edges_num):
                is_start = (self.edges[k, j].s_node_idx == i)
                self.nodes_2_edges_s[i, node_2_edge_s_ind, j] = k * (is_start) - (1 - is_start)
                is_end =  (self.edges[k, j].e_node_idx == i)
                self.nodes_2_edges_e[i, node_2_edge_e_ind, j] = k * (is_end) - (1 - is_end)

        # check whether the index of egde start and end is the same with pos of edge start and end
        # for i, j in ti.ndrange(self.edges_num, self.env_num):
        #     diff_pos_s = (self.nodes[edges_ind_s[i, j], j].pos[0] - edges_pos_s[i, j, 0],
        #                   self.nodes[edges_ind_s[i, j], j].pos[1] - edges_pos_s[i, j, 1])
        #     diff_pos_e = (self.nodes[edges_ind_e[i, j], j].pos[0] - edges_pos_e[i, j, 0],
        #                   self.nodes[edges_ind_e[i, j], j].pos[1] - edges_pos_e[i, j, 1])
    
    def warp_data(self):
        all_data = {
            "envs": [],
            "max_size": self.max_size
        }
        for j in range(self.env_num):
            data = {
                "nodes": [],
                "edges": []
            }
            for i in range(self.nodes_num):
                node_data = {
                    "idx": self.nodes[i, j].idx,
                    "pos": [self.nodes[i, j].pos[0], self.nodes[i, j].pos[1]],
                    "attribute": self.nodes[i, j].attribute
                }
                data["nodes"].append(node_data)

            for i in range(self.edges_num):
                edge_data = {
                    "s_node_idx": self.edges[i, j].s_node_idx,
                    "e_node_idx": self.edges[i, j].e_node_idx,
                    "s_pos": [self.edges[i, j].s_pos[0], self.edges[i, j].s_pos[1]],
                    "e_pos": [self.edges[i, j].e_pos[0], self.edges[i, j].e_pos[1]],
                    "s_con": [self.edges[i, j].s_con[0], self.edges[i, j].s_con[1]],
                    "e_con": [self.edges[i, j].e_con[0], self.edges[i, j].e_con[1]],
                    "line_state": self.edges[i, j].line_state
                }
                data["edges"].append(edge_data)
            
            all_data["envs"].append(data)
        return all_data
    
    def extract_data(self, dict_input):
        node_pos_vector = ti.field(dtype=ti.f32, shape=(self.nodes_num, self.env_num, 2))
        nodes_attribute = ti.field(dtype=ti.i8, shape=(self.nodes_num, self.env_num))

        edges_ind_s = ti.field(dtype=ti.i8, shape=(self.edges_num, self.env_num))
        edges_ind_e = ti.field(dtype=ti.i8, shape=(self.edges_num, self.env_num))
        
        edges_pos_s = ti.field(dtype=ti.f32, shape=(self.edges_num, self.env_num, 2))
        edges_pos_e = ti.field(dtype=ti.f32, shape=(self.edges_num, self.env_num, 2))
        edges_con_s = ti.field(dtype=ti.f32, shape=(self.edges_num, self.env_num, 2))
        edges_con_e = ti.field(dtype=ti.f32, shape=(self.edges_num, self.env_num, 2))
        edges_line_state = ti.field(dtype=ti.f32, shape=(self.edges_num, self.env_num))

        self.max_size = dict_input["max_size"]

        for i in range(self.env_num):
            for j in range(self.nodes_num):
                node_data = dict_input["envs"][i]["nodes"][j]
                node_pos_vector[j, i, 0] = node_data["pos"][0]
                node_pos_vector[j, i, 1] = node_data["pos"][1]
                nodes_attribute[j, i] = node_data["attribute"]
        
        for i in range(self.env_num):
            for j in range(self.edges_num):
                edge_data = dict_input["envs"][i]["edges"][j]
                edges_ind_s[j, i] = edge_data["s_node_idx"]
                edges_ind_e[j, i] = edge_data["e_node_idx"]
                edges_line_state[j, i] = edge_data["line_state"]
                for k in range(2):
                    edges_pos_s[j, i, k] = edge_data["s_pos"][k]
                    edges_pos_e[j, i, k] = edge_data["e_pos"][k]
                    edges_con_s[j, i, k] = edge_data["s_con"][k]
                    edges_con_e[j, i, k] = edge_data["e_con"][k]

        self.generate_trails(node_pos_vector, nodes_attribute, edges_ind_s, edges_ind_e,
                             edges_pos_s, edges_pos_e, edges_con_s, edges_con_e, edges_line_state)
            
    def save_to_yaml(self, file_name): 
        all_data = self.warp_data()
        yaml_file_name = file_name + ".yaml"
        with open(yaml_file_name, 'w') as yaml_file:
            yaml.dump(all_data, yaml_file)
    
    def save_to_json(self, file_name): 
        all_data = self.warp_data()
        json_file_name = file_name + ".json"
        with open(json_file_name, 'w') as json_file:
            json.dump(all_data, json_file,
                      indent=2, sort_keys=True)

    def load_from_yaml(self, file_name):
        with open(file_name, 'r') as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.extract_data(data)
        
    def load_from_json(self, file_name):
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
        self.extract_data(data)  

    def render(self, gui, env_idx=0):
        centers = self.nodes.pos.to_numpy()[:, env_idx, :]
        centers /= self.max_size
        gui.circles(centers, color=self.color, radius=6)

        # suppose that all the edges are direct lines
        s_pos = self.edges.s_pos.to_numpy()[:, env_idx] / self.max_size
        e_pos = self.edges.e_pos.to_numpy()[:, env_idx] / self.max_size
        gui.lines(s_pos, e_pos, color=self.color)

        # if not, please use this code
        # for i in range(self.edges_num):
        #     if self.edges[i, env_idx].line_state == 0:  # Bezier line
        #         t_values = np.linspace(0, 1, num=30)
        #         points = np.array([self.edges[i, env_idx].get_t_pos(t) for t in t_values])
        #         # gui.circles(points, color=0x000000, radius=3)
        #         for k in range(len(points) - 1):
        #             gui.line(points[k], points[k + 1], color=self.color)
        #     else:  # Direct line
        #         s_pos = self.edges[i, env_idx].s_pos / self.max_size
        #         e_pos = self.edges[i, env_idx].e_pos / self.max_size
        #         gui.line(s_pos, e_pos, color=self.color)


@ti.data_oriented
class Static_maps:
    def __init__(self, 
                 grid_n,
                 nodes_num,
                 edges_num,
                 env_num,
                 static_maps_settings_file,
                 trails_settings_file,
                 img_size=540):
        self.grid_n = grid_n # n columns * n rows
        self.grid_length = 1.0
        self.max_size = self.grid_n * self.grid_length
        self.env_num = env_num
        self.static_maps_settings = static_maps_settings_file
        self.trails_settings = trails_settings_file
        self.img_size = img_size
        
        self.field_map = ti.field(dtype=ti.u8,
                                  shape=(self.grid_n, self.grid_n, self.env_num),
                                  name="static field maps")
        self.canvas = ti.field(dtype=ti.u8, shape=(self.grid_n, self.grid_n))

        self.width = self.grid_n * self.grid_length
        self.height = self.grid_n * self.grid_length
        
        self.trails = Trails(nodes_num, edges_num, env_num) 

        self.generate_field()      
    
    def generate_field(self):
        if self.static_maps_settings[-4:] == "json":
            with open(self.static_maps_settings, 'r') as json_file:
                data = json.load(json_file)
        elif self.static_maps_settings[-4:] == "yaml":
            with open(self.static_maps_settings, 'r') as yaml_file:
                data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        if data is None:
            raise Exception
        else:
            self.grid_n = data["grid_n"]
            self.max_size = data["max_size"]
            self.grid_length = self.max_size / self.grid_n
            data_maps = data["maps"]
            for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.env_num):
                self.field_map[i, j, k] = data_maps[i][j][k]

        if self.trails_settings[-4:] == "json":
            self.trails.load_from_json(self.trails_settings)
        elif self.trails_settings[-4:] == "yaml":
            self.trails.load_from_yaml(self.trails_settings)
        
    
    @ti.kernel
    def interact_with_borad(self):
        pass

    @ti.kernel
    def plot_maps_2_canvas(self, env_idx: int):
        for x, y in ti.ndrange(self.grid_n, self.grid_n):
            # self.canvas[x, y] = self.field_map[x, y, env_idx] / 255
            field_x = x * self.grid_n // self.img_size
            field_y = y * self.grid_n // self.img_size
            self.canvas[x, y] = self.field_map[field_x, field_y, env_idx]


    def render(self, gui, env_idx=0):
        # window_width, window_height = gui.res
        self.plot_maps_2_canvas(env_idx)
        # gui.contour(self.canvas, normalize=True)
        gui.set_image(self.canvas.to_numpy())
        self.trails.render(gui, env_idx)


if __name__ == "__main__":
    filename_maps = './scenario_template/test_env/maps_data.json'
    filename_trail = './scenario_template/test_env/ware_house_test.json'
    # test_maps(filename_maps, filename_trail)
    # test_load_from_file(filename_trail)    
    # test_save_to_yaml()
    pass
    
