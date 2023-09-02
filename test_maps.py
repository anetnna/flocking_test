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

from static_maps import Trails, Static_maps

def test_save_to_yaml():
    ti.init(ti.cpu, debug=True)
    nodes_num = 16
    edges_num = 48
    env_num = 2
    max_size = 10
    test_trails = Trails(nodes_num=nodes_num, edges_num=edges_num, env_num=env_num, max_size=max_size)
    node_pos_vector_tmp = ti.field(shape=(nodes_num, env_num, 2), dtype=ti.f32)
    nodes_attribute = ti.field(shape=(nodes_num, env_num), dtype=ti.i8)
    x_tmp, y_tmp = 0.125*max_size, 0.125*max_size
    node_idx = 0
    for i in range(nodes_num):
        for k in range(env_num):
            nodes_attribute[i, k] = 0
    for node_idx in range(nodes_num):
        # pos_tmp = vec2(x_tmp, y_tmp)
        for k in range(env_num):
            node_pos_vector_tmp[node_idx, k, 0] = x_tmp
            node_pos_vector_tmp[node_idx, k, 1] = y_tmp
        x_tmp += 0.25 * max_size
        if x_tmp > 0.95 * max_size:
            x_tmp = 0.125 * max_size
            y_tmp += 0.25 * max_size
        node_idx += 1
    # print(f"DEBUG INFO: original node_pos_vector: {node_pos_vector_tmp.to_numpy()}")
    a_mtraix_tmp = ti.field(shape=(nodes_num, nodes_num, env_num), dtype=ti.i8)
    for i in range(nodes_num):
        for j in range(nodes_num):
            if (j == i + 1) and not (i%4 == 3):
                a_mtraix_tmp[i, j, 0] = 1
                a_mtraix_tmp[i, j, 1] = 1
            if j == i + 4:
                a_mtraix_tmp[i, j, 0] = 1
                a_mtraix_tmp[i, j, 1] = 1
            if (j == i - 1)  and not (j%4 == 3):
                a_mtraix_tmp[i, j, 0] = 1
                a_mtraix_tmp[i, j, 1] = 1
            if j == i - 4:
                a_mtraix_tmp[i, j, 0] = 1
                a_mtraix_tmp[i, j, 1] = 1
    
    print(a_mtraix_tmp.to_numpy()[:, :, 0])
    edge_idx_list = [0] * env_num
    edge_ind_s_tmp = ti.field(shape=(edges_num, env_num), dtype=ti.i8)
    edge_ind_e_tmp = ti.field(shape=(edges_num, env_num), dtype=ti.i8)
    edge_pos_s_tmp = ti.field(shape=(edges_num, env_num, 2), dtype=ti.f32)
    edge_pos_e_tmp = ti.field(shape=(edges_num, env_num, 2), dtype=ti.f32)
    edge_con_s_tmp = ti.field(shape=(edges_num, env_num, 2), dtype=ti.f32)
    edge_con_e_tmp = ti.field(shape=(edges_num, env_num, 2), dtype=ti.f32)
    edges_line_state_tmp = ti.field(shape=(edges_num, env_num), dtype=ti.f32)
    edges_line_state_tmp.fill(1.0)

    for i in range(nodes_num):
        for j in range(nodes_num):
            # if i > j:
            for k in range(env_num):
                if a_mtraix_tmp[i, j, k] > 0:
                    edge_idx = edge_idx_list[k]
                    edge_ind_s_tmp[edge_idx, k] = i
                    edge_ind_e_tmp[edge_idx, k] = j
                    for kk in range(2):
                        edge_pos_s_tmp[edge_idx, k, kk] = node_pos_vector_tmp[i, k, kk]
                        edge_con_s_tmp[edge_idx, k, kk] = node_pos_vector_tmp[j, k, kk]
                        edge_pos_e_tmp[edge_idx, k, kk] = node_pos_vector_tmp[j, k, kk]
                        edge_con_e_tmp[edge_idx, k, kk] = node_pos_vector_tmp[i, k, kk]
                    edge_idx_list[k] += 1
    
    # print(f"DEBUG INFO: before function edge_pos_s_tmp: {edge_pos_s_tmp}")
    test_trails.generate_trails(node_pos_vector=node_pos_vector_tmp,
                                nodes_attribute=nodes_attribute,
                                edges_ind_s=edge_ind_s_tmp,
                                edges_ind_e=edge_ind_e_tmp,
                                edges_pos_s=edge_pos_s_tmp,
                                edges_pos_e=edge_pos_e_tmp,
                                edges_con_s=edge_con_s_tmp,
                                edges_con_e=edge_con_e_tmp,
                                edges_line_state=edges_line_state_tmp)
    
    # print(test_trails.nodes.pos.to_numpy())
    # print("edges:")
    # print(test_trails.edges.s_node_idx.to_numpy())
    
    WINDOW_HEIGHT = 540
    AR = 1
    WINDOW_WIDTH = AR * WINDOW_HEIGHT
    gui = ti.GUI("flocking behavior", 
                 res=(WINDOW_WIDTH, WINDOW_HEIGHT),
                 show_gui=True)
    while gui.running:
        gui.clear(0xffffff)
        test_trails.render(gui, 0)
        gui.show()
    
    test_trails.save_to_yaml("./scenario_template/test_tmp")
    test_trails.save_to_json("./scenario_template/test_tmp")


def test_load_from_file(filename):
    ti.init(ti.cpu)
    col_nodes = 9
    row_nodes = 7
    nodes_num = col_nodes * row_nodes
    edges_num = 2 * (col_nodes * (row_nodes-1) + (col_nodes-1) * row_nodes)
    env_num = 2
    test_trails = Trails(nodes_num=nodes_num, edges_num=edges_num, env_num=env_num)

    if filename[-4:] == "json":
        test_trails.load_from_json(filename)
    elif filename[-4:] == "yaml":
        test_trails.load_from_yaml(filename)

    # print(test_trails.nodes.pos.to_numpy())
    WINDOW_HEIGHT = 540
    AR = 1
    WINDOW_WIDTH = AR * WINDOW_HEIGHT
    gui = ti.GUI("flocking behavior", 
                 res=(WINDOW_WIDTH, WINDOW_HEIGHT),
                 show_gui=True)
    while gui.running:
        gui.clear(0xffffff)
        test_trails.render(gui, 0)
        gui.show()


def test_maps(filename_maps, filename_trail):
    ti.init(ti.cpu)
    col_nodes = 9
    row_nodes = 7
    nodes_num = col_nodes * row_nodes
    edges_num = 2 * (col_nodes * (row_nodes-1) + (col_nodes-1) * row_nodes)
    env_num = 2
    grid_n = 540
    WINDOW_HEIGHT = 540
    AR = 1
    WINDOW_WIDTH = AR * WINDOW_HEIGHT

    static_maps_settings_file = filename_maps
    trails_settings_file = filename_trail
    maps = Static_maps(grid_n, nodes_num, edges_num, env_num, 
                       static_maps_settings_file, trails_settings_file,
                       WINDOW_HEIGHT)
   
    gui = ti.GUI("flocking behavior", 
                 res=(WINDOW_WIDTH, WINDOW_HEIGHT),
                 show_gui=True)
    while gui.running:
        gui.clear(0xffffff)
        maps.render(gui, 0)
        gui.show()


def test_maps_2():
    ti.init(ti.cpu)
    col_nodes = 9
    row_nodes = 7
    nodes_num = col_nodes * row_nodes
    edges_num = 2 * (col_nodes * (row_nodes-1) + (col_nodes-1) * row_nodes)
    env_num = 2
    grid_n = 540
    WINDOW_HEIGHT = 540
    max_size = 20
    marign = 20
    node_dis = 50
    AR = 1
    WINDOW_WIDTH = AR * WINDOW_HEIGHT

    node_pos_vector_tmp = ti.field(shape=(nodes_num, env_num, 2), dtype=ti.f32)
    nodes_attribute = ti.field(shape=(nodes_num, env_num), dtype=ti.i8)

    for i in range(col_nodes):
        for j in range(row_nodes):
            node_idx = j * col_nodes + i
            for k in range(env_num):
                node_pos_vector_tmp[node_idx, k, 0] = (marign + node_dis * (i+1)) / grid_n * max_size
                node_pos_vector_tmp[node_idx, k, 1] = (grid_n - (marign + node_dis * (j+1))) / grid_n * max_size
                if i == 0:
                    nodes_attribute[node_idx, k] = 1
                else:
                    nodes_attribute[node_idx, k] = 0
    
    a_mtraix_tmp = ti.field(shape=(nodes_num, nodes_num, env_num), dtype=ti.i8)
    for i in range(nodes_num):
        for k in range(env_num):
            if (i + 1 < nodes_num) and not (i%col_nodes == col_nodes-1):
                a_mtraix_tmp[i, i+1, k] = 1
                a_mtraix_tmp[i+1, i, k] = 1
            if i + col_nodes < nodes_num:
                a_mtraix_tmp[i, i+col_nodes, k] = 1
                a_mtraix_tmp[i+col_nodes, i, k] = 1
            if (i - 1 >= 0) and not (i%col_nodes == 0):
                a_mtraix_tmp[i, i-1, k] = 1
                a_mtraix_tmp[i-1, i, k] = 1
            if i - col_nodes >= 0:
                a_mtraix_tmp[i, i-col_nodes, k] = 1
                a_mtraix_tmp[i-col_nodes, i, k] = 1

    edge_ind_s_tmp = ti.field(shape=(edges_num, env_num), dtype=ti.i8)
    edge_ind_e_tmp = ti.field(shape=(edges_num, env_num), dtype=ti.i8)
    edge_pos_s_tmp = ti.field(shape=(edges_num, env_num, 2), dtype=ti.f32)
    edge_pos_e_tmp = ti.field(shape=(edges_num, env_num, 2), dtype=ti.f32)
    edge_con_s_tmp = ti.field(shape=(edges_num, env_num, 2), dtype=ti.f32)
    edge_con_e_tmp = ti.field(shape=(edges_num, env_num, 2), dtype=ti.f32)
    edges_line_state_tmp = ti.field(shape=(edges_num, env_num), dtype=ti.f32)
    edges_line_state_tmp.fill(1.0)
    edge_idx_list = [0] * env_num

    for i in range(nodes_num):
        for j in range(nodes_num):
            for k in range(env_num):
                if a_mtraix_tmp[i, j, k] > 0:
                    edge_idx = edge_idx_list[k]
                    edge_ind_s_tmp[edge_idx, k] = i
                    edge_ind_e_tmp[edge_idx, k] = j
                    for kk in range(2):
                        edge_pos_s_tmp[edge_idx, k, kk] = node_pos_vector_tmp[i, k, kk]
                        edge_con_s_tmp[edge_idx, k, kk] = node_pos_vector_tmp[j, k, kk]
                        edge_pos_e_tmp[edge_idx, k, kk] = node_pos_vector_tmp[j, k, kk]
                        edge_con_e_tmp[edge_idx, k, kk] = node_pos_vector_tmp[i, k, kk]
                    edge_idx_list[k] += 1

    test_trails = Trails(nodes_num=nodes_num, edges_num=edges_num, env_num=env_num, max_size=max_size)
    test_trails.generate_trails(node_pos_vector=node_pos_vector_tmp,
                                nodes_attribute=nodes_attribute,
                                edges_ind_s=edge_ind_s_tmp,
                                edges_ind_e=edge_ind_e_tmp,
                                edges_pos_s=edge_pos_s_tmp,
                                edges_pos_e=edge_pos_e_tmp,
                                edges_con_s=edge_con_s_tmp,
                                edges_con_e=edge_con_e_tmp,
                                edges_line_state=edges_line_state_tmp)
   
    gui = ti.GUI("flocking behavior", 
                 res=(WINDOW_WIDTH, WINDOW_HEIGHT),
                 show_gui=True)
    
    test_trails.save_to_yaml("./scenario_template/ware_house_test")
    test_trails.save_to_json("./scenario_template/ware_house_test")

    while gui.running:
        gui.clear(0xffffff)
        test_trails.render(gui, 0)
        gui.show()


if __name__ == "__main__":
    filename_maps = './test_env/maps_data.json'
    filename_trail = './test_env/ware_house_test.json'
    test_maps(filename_maps, filename_trail)
    # test_load_from_file(filename_trail)
