import numpy as np
import networkx as nx
import tqdm
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

class TSP_plotter:
    def __init__(self) -> None:
        pass
    def plot_nx_graph(self, graph,  tour_length=None, solution=None, title=''):
        plt.style.use('seaborn-paper')
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=True, sharey=True)
        num_nodes = graph.number_of_nodes()
        labels = dict()
        if solution:
            labels = {solution[i]:i for i in graph.nodes}
            tour_edges = list(zip(solution, solution[1:]))
            tour_edges.append((solution[-1], solution[0]))
        else:
            labels = {i:i for i in graph.nodes}
        pos = {i:graph.nodes[i]['coord'] for i in graph.nodes}
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='y', node_size=200)
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
        plt.savefig('plots/tour_plot.png', dpi=400)
        plt.show()

class TSP_solver:
    def __init__(self) -> None:
        self.solution = None
        self.loader = TSP_loader()
    
    def calc_opt_tour_from_file(self, fname, scale_factor=0.0001):
        solver = TSPSolver.from_tspfile(fname)
        solution = solver.solve()
        length = solution.optimal_value * scale_factor
        solution = solution.tour
        self.solution = solution
        return length, solution
        
    def calc_opt_tour_from_nx(self, graph, scale=1): 
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
        
        solver_ = TSPSolver.from_tspfile(tmp.name)
        solution = solver_.solve()
        length = solution.optimal_value / (10**scale)
        solution = solution.tour
        self.solution = solution
        tmp.close()
        os.unlink(tmp.name)
        return length, solution
    
    def calc_control_param_from_nx(self, graph):
        num_nodes = graph.number_of_nodes()
        l_opt = self.calc_opt_tour_from_nx(graph)[0]
        node_coords = np.array([graph.nodes[k]['coord'] for k in graph.nodes])
        area = ConvexHull(node_coords).volume
        control_param = l_opt / np.sqrt(num_nodes * area)
        return control_param
    
    def calc_control_param_from_file(self, file_path):
        graph = self.loader.load_tsp_as_nx(file_path, scale_factor=1)
        l_opt = self.calc_opt_tour_from_file(file_path, scale_factor=1)[0]
        num_nodes = graph.number_of_nodes()
        node_coords = np.array([graph.nodes[k]['coord'] for k in graph.nodes])
        area = ConvexHull(node_coords).volume
        control_param = l_opt / np.sqrt(num_nodes * area)
        return control_param

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
    
    def calc_hard_problem(self, n=1, epsilon=0.1, max_iter=10000):
        graph = self.gen_graph()
        init_control_param = self.solver.calc_control_param_from_nx(graph)
        fitness = np.abs(init_control_param - 0.75)
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
            new_control_param = self.solver.calc_control_param_from_nx(tmp_graph)
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
            iter += 1
        return graph, fitness, iter
    
    def save_nx_as_tsp(self, graph_list, save_path):
        for k, graph in enumerate(graph_list):
            problem = tsplib95.models.StandardProblem()
            problem.name = 'TSP_Problem_{}'.format(k)
            problem.type = 'TSP'
            problem.dimension = graph.number_of_nodes()
            problem.edge_weight_type = 'EUC_2D'
            # problem.node_coord_type = 'TWOD_COORDS'
            problem.node_coords = {'{}'.format(node[0]): list(np.round(10000*node[1]['coord'], 0)) for node in graph.nodes.items()}
            # save_path = 'valid_sets/synthetic_nrange_10_20_200/TSP_Problem_{}.tsp'.format(k)
            file_path = save_path + 'TSP_Problem_{}.tsp'.format(k)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            problem.save(file_path)
            with open(file_path, 'a') as f:
                f.write('\n')
    
    def gen_graphs(self, num_graphs=1000):
        graph_list = []
        for i in range(0, num_graphs):
            graph_list.append(self.gen_graph())
        return graph_list

    def gen_graph(self):
        """
        Generates new graphs of different g_type--> used for training or testing
        """
        max_n = self.num_max
        min_n = self.num_min
        g_type = self.g_type
        cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if g_type == 'tsp_2d':
            # slow code, might need optimization
            nodes = np.random.rand(cur_n, 2)
            edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(nodes),2)]
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
            feature_dict = {k: {'coord': nodes[k]} for k in range(cur_n)} 
            nx.set_node_attributes(g, feature_dict)
        elif g_type == 'tsp':
            # slow code, might need optimization
            nodes = np.random.rand(cur_n, 2)
            edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(nodes),2)]
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
        return g