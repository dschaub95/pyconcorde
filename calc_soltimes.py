from TSP_utils import TSP_solver, TSP_plotter, TSP_generator, TSP_loader
from s2v_utils import *
import os
import sys


if __name__ == '__main__':
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]
    solver = TSP_solver()
    
    atoi = lambda text : int(text) if text.isdigit() else text
    natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]

    # solver.add_sol_times(data_dir='test_sets/synthetic_n_50_1000', num_runs=10)
    if opt['exec_mode'] == '0':
        
        solver.add_sol_times(data_dir='training_sets/synthetic_n_50_10000', num_runs=10)
        print("0")
    elif opt['exec_mode'] == '1':
        folder_names = os.listdir('test_sets')
        folder_names.sort(key=natural_keys)
        for folder in folder_names:
            if 'delta_0.005' not in folder:
                continue
            solver.add_sol_times(data_dir="test_sets/"+folder, num_runs=10)
            print("1")
    else:
        print("Error!")