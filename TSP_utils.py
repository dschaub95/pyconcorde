import numpy as np
import networkx as nx
import re
import os
import tempfile
import tsplib95
from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
from scipy.spatial import ConvexHull
from itertools import combinations
import random
import matplotlib.pyplot as plt
import matplotlib as mpl 
import time
from itertools import permutations
import sys
from os.path import exists
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class TSP_plotter:
    def __init__(self) -> None:
        pass
    def plot_nx_graph(self, graph, draw_edges=True, tour_length=None, solution=None, title='', save_path='../plots/tour_plot.png'):
        plt.style.use('seaborn-paper')
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=True, sharey=True)
        num_nodes = graph.number_of_nodes()
        labels = dict()
        if not (solution == None):
            labels = {i:i for i in graph.nodes}
            # labels = {solution[i]:i for i in graph.nodes}
            tour_edges = list(zip(solution, solution[1:]))
            tour_edges.append((solution[-1], solution[0]))
        else:
            labels = {i:i for i in graph.nodes}
        pos = {i:graph.nodes[i]['coord'] for i in graph.nodes}
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='y', node_size=200)
        if draw_edges:
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='y', width=1, alpha=0.2)
        if solution:
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=tour_edges, edge_color='r', width=2)
        # Draw labels
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=9)
        # ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.set_xlabel('x-coordinate')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_ylabel('y-coordinate')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400)
        plt.show()

class TSP_solver:
    def __init__(self) -> None:
        self.loader = TSP_loader()
    
    def calc_opt_tour_from_file(self, fname, scale_factor=1):
        solver = TSPSolver.from_tspfile(fname)
        
        start_time = time.time() 
        solution = solver.solve()
        end_time = time.time()
        sol_time = end_time - start_time 
        
        length = solution.optimal_value * scale_factor
        solution = solution.tour
        return length, sol_time, solution
        
    def calc_opt_tour_from_nx(self, graph, scale=6): 
        problem = tsplib95.models.StandardProblem()
        problem.name = 'TSP_Problem'
        problem.type = 'TSP'
        problem.dimension = graph.number_of_nodes()
        problem.edge_weight_type = 'EUC_2D'
        problem.node_coords = {'{}'.format(node[0]): list(np.round((10**scale) * node[1]['coord'], 0)) for node in graph.nodes.items()}
        tmp = tempfile.NamedTemporaryFile(delete=False)
        with open(tmp.name, 'w') as f:
            problem.write(f)
            f.write('\n')
        
        solver = TSPSolver.from_tspfile(tmp.name)
        
        start_time = time.time() 
        solution = solver.solve()
        end_time = time.time()
        sol_time = end_time - start_time 
        
        length = solution.optimal_value / (10**scale)
        solution = solution.tour
        tmp.close()
        os.unlink(tmp.name)
        return length, sol_time, solution
    
    def calc_control_param_from_nx(self, graph, generated=False, **args):
        num_nodes = graph.number_of_nodes()
        l_opt, sol_time = self.calc_opt_tour_from_nx(graph, **args)[0:2]
        if generated:
            area = 1
        else:
            area = self.calc_square_area(graph)
        # node_coords = np.array([graph.nodes[k]['coord'] for k in graph.nodes])
        # area = ConvexHull(node_coords).volume
        control_param = l_opt / np.sqrt(num_nodes * area)
        return control_param, sol_time
    
    def calc_control_param_from_file(self, file_path, generated=False):
        graph = self.loader.load_tsp_as_nx(file_path, scale_factor=1)
        l_opt, sol_time = self.calc_opt_tour_from_file(file_path, scale_factor=1)[0:2]
        num_nodes = graph.number_of_nodes()
        if generated:
            area = 1000000 * 1000000
        else:
            area = self.calc_square_area(graph)
        # node_coords = np.array([graph.nodes[k]['coord'] for k in graph.nodes])
        # area = ConvexHull(node_coords).volume
        control_param = l_opt / np.sqrt(num_nodes * area)
        return control_param, sol_time

    def calc_square_area(self, graph):
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        for node in graph.nodes:
            x = graph.nodes[node]['coord'][0]
            y = graph.nodes[node]['coord'][1]
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
        area = (max_x-min_x) * (max_y-min_y)
        return area

    def add_lengths_and_sols(self, data_dir, scale_factor=1000000, lengths_name='lengths.txt', sols_name='solutions.txt'):
        atoi = lambda text : int(text) if text.isdigit() else text
        natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]
        if data_dir[-1] != '/':
            data_dir = data_dir + '/'
        fnames = os.listdir(data_dir)
        fnames.sort(key=natural_keys)
        with open(data_dir+lengths_name, 'w') as f_len:
            with open(data_dir+sols_name, 'w') as f_sol:
                for fname in fnames:
                    if not '.tsp' in fname or '.sol' in fname:
                        continue
                    length, sol_time, solution = self.calc_opt_tour_from_file(data_dir+fname)
                    length = length/scale_factor
                    f_len.write('{}: {}\n'.format(fname, length))
                    f_sol.write('{}: {}\n'.format(fname, list(solution)))
                    self.del_tmp_files()
                    print(fname)
    
    def add_sol_times(self, data_dir, num_runs=1, scale_factor=1000000, file_name='sol_times.txt', verbose=0):
        
        atoi = lambda text : int(text) if text.isdigit() else text
        natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]
        if data_dir[-1] != '/':
            data_dir = data_dir + '/'
        fnames = os.listdir(data_dir)
        fnames.sort(key=natural_keys)
        num_samples = len([name for name in fnames if '.tsp' in name])
        # create file if not yet created
        if not exists(data_dir+file_name):
            with open(data_dir+file_name, 'w') as f:
                pass
        start_index = 0
        try:
            # check whether solution times were already calculated
            with open(data_dir+file_name, 'r') as f_cur:
                lines = f_cur.readlines()
                file_names = [line.split(':')[0].strip() for k, line in enumerate(lines)]
                indices = [int(name.split('.')[0].split('_')[-1]) for name in file_names]
                start_index = np.max(indices) + 1
        except:
            pass    
        if start_index == num_samples:
            print(f"Already calculated solution times for all {start_index} samples!")
        else:
            print(f"Calculated solution times for {start_index} samples")
            for fname in fnames:
                if not '.tsp' in fname or '.sol' in fname:
                    continue
                index = int(fname.split('.')[0].split('_')[-1])
                if index < start_index:
                    continue
                sol_times = []
                for i in range(num_runs):
                    with suppress_stdout():
                        length, sol_time, solution = self.calc_opt_tour_from_file(data_dir+fname)
                    sol_times.append(str(sol_time))
                    # os.system('clear')
                sol_times = ", ".join(sol_times)
                with open(data_dir+file_name, 'a') as f_time:
                    f_time.write(f"{fname}: {sol_times}\n")
                self.del_tmp_files()
                print(f'Folder name: {data_dir}, Index: {index}, file name: {fname}')

    def brute_solve_tsp(self, graph):
        perms = permutations(list(graph.nodes))
        opt_tour_length = np.inf
        opt_tour = []
        for perm in perms:
            tmp_len = self.calc_tour_length(graph, perm)
            if tmp_len < opt_tour_length:
                opt_tour_length = tmp_len
                opt_tour = list(perm)
            else:
                continue
        return opt_tour_length, opt_tour
    
    def calc_tour_length(self, graph, solution):
        tot_len = 0
        for i in range(np.array(solution).shape[0]):
            if i == np.array(solution).shape[0] - 1:
                tot_len += graph[solution[i]][solution[0]]['weight']
            else:
                tot_len += graph[solution[i]][solution[i + 1]]['weight']
        return tot_len

    def del_tmp_files(self, dir='./', bad_endings=['res', 'sol', 'sav', 'pul']):
        for f in os.scandir(dir):
            ending = f.name.split('.')[-1]
            if f.is_file() and ending in bad_endings:
                os.remove(f.name)

class TSP_loader:
    def __init__(self) -> None:
        pass
    
    def load_multi_tsp_as_nx(self, data_dir, scale_factor=0.0001):
        atoi = lambda text : int(text) if text.isdigit() else text
        natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]
        fnames = os.listdir(data_dir)
        fnames.sort(key=natural_keys)
        graph_list = []
        for fname in fnames:
            if not 'tsp' in fname:
                continue
            g = self.load_tsp_as_nx(data_dir+fname, scale_factor=scale_factor)
            graph_list.append(g)
        return graph_list

    def load_tsp_as_nx(self, file_path, scale_factor=0.0001):
        try:
            problem = tsplib95.load(file_path)
            g = problem.get_graph()
            # remove edges from nodes to itself
            ebunch=[(k,k) for k in g.nodes()]
            g.remove_edges_from(ebunch)
            for node in g.nodes():
                g.nodes[node]['coord'] = np.array(g.nodes[node]['coord']) * scale_factor
            for edge in g.edges:
                g.edges[edge]['weight'] = g.edges[edge]['weight'] * scale_factor
        except:
            g = nx.Graph()
            print("Error!")
        return g

class TSP_generator:
    def __init__(self, g_type, num_min, num_max) -> None:
        self.g_type = g_type
        self.num_min = num_min
        self.num_max = num_max
        self.solver = TSP_solver()
    
    def calc_cparam_problem(self, n=1, desired_cparam=0.75, delta=0.05, max_iter=10000, verbose=1, del_tmp_files=True, **args):
        graph = self.gen_graph(**args)
        
        init_cparam, init_sol_time = self.solver.calc_control_param_from_nx(graph, generated=True, scale=6)
        sol_time = init_sol_time
        
        cparam = init_cparam
        cparam_diff = np.abs(cparam - desired_cparam)
        
        iter = 0
        while cparam_diff >= delta and iter < max_iter:
            cur_node_coords = np.array([graph.nodes[k]['coord'] for k in graph.nodes])
            test_nodes = random.sample(list(graph.nodes), n)
            for test_node in test_nodes:
                cur_node_coords[test_node] = np.random.rand(1, 2)
            new_edges = [(s[0], t[0], np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(cur_node_coords),2)]
            # create new graph
            new_graph = nx.Graph()
            new_graph.add_weighted_edges_from(new_edges)
            feature_dict = {k: {'coord': cur_node_coords[k]} for k in graph.nodes} 
            nx.set_node_attributes(new_graph, feature_dict)
            
            # check fitness of altered graph
            new_cparam, new_sol_time = self.solver.calc_control_param_from_nx(new_graph, generated=True, scale=6)
            
            new_cparam_diff = np.abs(new_cparam - desired_cparam)
            cparam_diff = np.abs(cparam - desired_cparam)
            
            if new_cparam_diff <= cparam_diff:
                graph = new_graph
                cparam = new_cparam
                sol_time = new_sol_time
            else:
                temperature = 1 / (iter + 10)
                boltzmann_coeff = np.exp(-1 * (new_cparam_diff - cparam_diff)/temperature)
                if np.random.rand(1, 1) < boltzmann_coeff:
                    graph = new_graph
                    cparam = new_cparam
                    sol_time = new_sol_time
                else:
                    new_graph.clear()
            cparam_diff = np.abs(cparam - desired_cparam)
            if verbose >= 2:
                print('Iteration: ', iter)
            iter += 1
        if verbose >= 1:
            print('Original cparam: ', init_cparam)
            print("Original solution time: ", init_sol_time)
            print('Final cparam: ', cparam)
            print("Final solution time: ", sol_time)
            print('Number of iterations: ', iter)     
        if del_tmp_files:
            self.solver.del_tmp_files()
        return graph, cparam, iter

    def calc_hard_problem(self, n=1, epsilon=0.1, max_iter=10000, **args):
        graph = self.gen_graph(**args)
        init_cparam, sol_time = self.solver.calc_control_param_from_nx(graph)
        fitness = np.abs(init_cparam - 0.75)
        iter = 0
        while fitness > epsilon and iter < max_iter:
            cur_node_coords = np.array([graph.nodes[k]['coord'] for k in graph.nodes])
            test_nodes = random.sample(list(graph.nodes), n)
            for test_node in test_nodes:
                cur_node_coords[test_node] = np.random.rand(1, 2)
            new_edges = [(s[0], t[0], np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(cur_node_coords),2)]
            # create new graph
            tmp_graph = nx.Graph()
            tmp_graph.add_weighted_edges_from(new_edges)
            feature_dict = {k: {'coord': cur_node_coords[k]} for k in graph.nodes} 
            nx.set_node_attributes(tmp_graph, feature_dict)
            # check fitness of altered graph
            new_control_param, new_sol_time = self.solver.calc_control_param_from_nx(tmp_graph)
            tmp_fitness = np.abs(new_control_param - 0.75)
            if tmp_fitness <= fitness:
                graph = tmp_graph
                fitness = tmp_fitness
            else:
                temperature = 1 / (iter + 10)
                boltzmann_coeff = np.exp(-1 * (tmp_fitness - fitness)/temperature)
                if np.random.rand(1, 1) < boltzmann_coeff:
                    graph = tmp_graph
                    fitness = tmp_fitness
                else:
                    tmp_graph.clear()
            print('Iteration: ', iter)
            iter += 1
        print('Original Fitness: ', np.abs(init_cparam - 0.75))
        print('Final Fitness: ', fitness)
        print('Iterations: ', iter)     
        return graph, fitness, iter
    
    def save_nx_as_tsp(self, graph_list, save_path, scale=6, start_index=0, init_pos=None, goal_pos=None):
        # make sure everything is saved in the save dir
        if save_path[-1] != '/':
            save_path = save_path + '/'
        # create save dir if needed
        if not os.path.isdir(save_path):
            try: 
                os.mkdir(save_path) 
            except OSError as error: 
                print(error) 
        for k, graph in enumerate(graph_list):
            problem = tsplib95.models.StandardProblem()
            problem.name = 'TSP_Problem_{}'.format(start_index + k)
            problem.type = 'TSP'
            problem.dimension = graph.number_of_nodes()
            problem.edge_weight_type = 'EUC_2D'
            # problem.node_coord_type = 'TWOD_COORDS'
            if init_pos == '0,1' and goal_pos == '-0.5,0.5':
                problem.node_coords = {'{}'.format(node[0]): list(np.round((10**scale)*(node[1]['coord']-0.5), 0)) for node in graph.nodes.items()}
            else:
                problem.node_coords = {'{}'.format(node[0]): list(np.round((10**scale)*node[1]['coord'], 0)) for node in graph.nodes.items()}
            # save_path = 'valid_sets/synthetic_nrange_10_20_200/TSP_Problem_{}.tsp'.format(k)
            file_path = save_path + 'TSP_Problem_{}.tsp'.format(start_index + k)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            problem.save(file_path)
            with open(file_path, 'a') as f:
                f.write('\n')
    
    def gen_graphs(self, num_graphs=1000, **args):
        graph_list = []
        for i in range(0, num_graphs):
            graph_list.append(self.gen_graph(**args))
        return graph_list

    def gen_graph(self, pos='0,1'):
        """
        Generates new graphs of different g_type--> used for training or testing
        """
        max_n = self.num_max
        min_n = self.num_min
        g_type = self.g_type
        cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if g_type == 'tsp_2d':
            # slow code, might need optimization
            if pos == '0,1':
                node_postions = np.random.rand(cur_n, 2)
            elif pos == '-0.5,0.5':
                node_postions = np.random.rand(cur_n, 2) - 0.5
            else:
                print("Unknown position input!")
            edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(node_postions),2)]
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
            feature_dict = {k: {'coord': node_postions[k]} for k in range(cur_n)} 
            nx.set_node_attributes(g, feature_dict)
        elif g_type == 'tsp':
            # slow code, might need optimization
            node_postions = np.random.rand(cur_n, 2)
            edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(node_postions),2)]
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
        return g