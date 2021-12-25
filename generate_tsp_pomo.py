from TSP_utils import TSP_solver, TSP_plotter, TSP_generator, TSP_loader
import numpy as np
import os
import networkx as nx
import argparse
import json
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default='./test_sets')
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--scale_factor", type=int, default=8)
    opts = parser.parse_known_args()[0]

    # generate directory and subdirs based on opts
    save_path = f'{opts.save_folder}/uniform_n_{opts.num_nodes}_{opts.num_samples}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # problem_folder = f'{save_path}/problems'
    # solution_folder = f'{save_path}/solutions'
    # heatmap_folder = f'{save_path}/heatmaps'

    generator = TSP_generator(g_type='tsp_2d', num_min=opts.num_nodes, num_max=opts.num_nodes)
    solver = TSP_solver()
    for i in tqdm(range(opts.start_index, opts.num_samples)):
        # generate instance
        graph = generator.gen_graph()
        # solve instance
        length, sol_time, solution = solver.calc_opt_tour_from_nx(graph, scale=8)
        # delete temp files
        solver.del_tmp_files()
        
        # save everything
        idx_str = f'{i}'.zfill(len(str(opts.num_samples)))
        # problem_name = f'TSP_Problem_{idx_str}'
        problem_name = f'tsp_{idx_str}'
        instance_path = f'{save_path}/{problem_name}'
        if not os.path.exists(instance_path):
            os.mkdir(instance_path)



        # save instance data to folder
        # once just save the networkx graph
        # convert numpy arrays to list
        for i in range(graph.number_of_nodes()):
            graph.nodes[i].update({'coord': list(graph.nodes[i]['coord'])})
        nx.write_gml(graph, path=f'{instance_path}/nx_graph.gml', stringizer=str)
        # extract and save node_feats
        node_feats = np.array([graph.nodes[i]['coord'] for i in range(graph.number_of_nodes())])
        np.savetxt(f'{instance_path}/node_feats.txt', node_feats)
        # extract and save edge weights#
        edge_weights = nx.convert_matrix.to_numpy_array(graph)
        np.savetxt(f'{instance_path}/edge_weights.txt', edge_weights)

        # save solution data to folder
        solution_data = {'problem_name': problem_name,
                         'opt_tour_length': length,
                         'opt_tour': solution.tolist()}
        with open(f"{instance_path}/solution.json", 'w') as f:
            # indent=2 is not needed but makes the file human-readable
            json.dump(solution_data, f, indent=2)




################################ OLD VERSION #######################################
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--save_folder", type=str, default='./test_sets')
#     parser.add_argument("--num_nodes", type=int, default=100)
#     parser.add_argument("--num_samples", type=int, default=10000)
#     parser.add_argument("--start_index", type=int, default=0)
#     parser.add_argument("--scale_factor", type=int, default=8)
#     opts = parser.parse_known_args()[0]

#     # generate directory and subdirs based on opts
#     save_path = f'{opts.save_folder}/uniform_n_{opts.num_nodes}_{opts.num_samples}'
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
    
#     # problem_folder = f'{save_path}/problems'
#     # solution_folder = f'{save_path}/solutions'
#     # heatmap_folder = f'{save_path}/heatmaps'

#     generator = TSP_generator(g_type='tsp_2d', num_min=opts.num_nodes, num_max=opts.num_nodes)
#     solver = TSP_solver()
#     for i in tqdm(range(opts.start_index, opts.num_samples)):
#         # generate instance
#         graph = generator.gen_graph()
#         # solve instance
#         length, sol_time, solution = solver.calc_opt_tour_from_nx(graph, scale=8)
#         # delete temp files
#         solver.del_tmp_files()
        
#         # save everything
#         idx_str = f'{i}'.zfill(len(str(opts.num_samples)))
#         # problem_name = f'TSP_Problem_{idx_str}'
#         problem_name = f'problem_{idx_str}'
#         instance_path = f'{save_path}/{problem_name}'
#         if not os.path.exists(instance_path):
#             os.mkdir(instance_path)

#         # save instance data to folder
#         generator.save_nx_as_tsp_single(graph, save_path=instance_path, problem_name=problem_name, scale=8)

#         # save solution data to folder
#         solution_data = {'problem_name': problem_name,
#                          'opt_tour_length': length,
#                          'opt_tour': solution.tolist()}
#         with open(f"{instance_path}/solution.txt", 'w') as f:
#             # indent=2 is not needed but makes the file human-readable
#             json.dump(solution_data, f, indent=2)