import argparse
import pprint as pp
import os
import sys
import tempfile
import shutil
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from concorde.tsp import TSPSolver  # Install from https://github.com/jvkersch/pyconcorde
from tqdm import tqdm  # Install tqdm with `pip install tqdm`

class SuppressOutput:
    """Context manager to suppress all output (stdout and stderr) at OS level."""
    def __enter__(self):
        self._stdout_fd = os.dup(1)
        self._stderr_fd = os.dup(2)
        self._null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(self._null_fd, 1)  # Redirect stdout to /dev/null
        os.dup2(self._null_fd, 2)  # Redirect stderr to /dev/null

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self._stdout_fd, 1)  # Restore stdout
        os.dup2(self._stderr_fd, 2)  # Restore stderr
        os.close(self._null_fd)
        os.close(self._stdout_fd)  # Close the duplicated stdout fd
        os.close(self._stderr_fd)  # Close the duplicated stderr fd

def cleanup_temp_files(temp_dir):
    """Remove files from the program's dedicated temporary directory."""
    if os.path.exists(temp_dir):
        for tmp_file in os.listdir(temp_dir):
            tmp_path = os.path.join(temp_dir, tmp_file)
            try:
                if os.path.isfile(tmp_path):
                    os.unlink(tmp_path)
                elif os.path.isdir(tmp_path):
                    shutil.rmtree(tmp_path)
            except Exception as e:
                print(f"Error cleaning temp file {tmp_file}: {e}")

def generate_and_solve_tsp_instance(args):
    idx, seed, min_nodes, max_nodes, node_dim = args
    np.random.seed(seed)
    num_nodes = np.random.randint(min_nodes, max_nodes +1)
    nodes_coord = np.random.random((num_nodes, node_dim))

    with SuppressOutput():
        solver = TSPSolver.from_data(nodes_coord[:,0], nodes_coord[:,1], norm='GEO')
        solution = solver.solve()

    # Verify that the solution is valid
    if (np.sort(solution.tour) == np.arange(num_nodes)).all():
        # Prepare the output line
        coord_str = " ".join(f"{x} {y}" for x, y in nodes_coord)
        tour_str = " ".join(str(node_idx + 1) for node_idx in solution.tour)
        output_line = f"{coord_str} output {tour_str} {solution.tour[0] + 1}\n"
        return idx, output_line
    else:
        # Return None if the solution is invalid
        return idx, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=100)
    parser.add_argument("--max_nodes", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--node_dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=62279)
    opts = parser.parse_args()

    if opts.filename is None:
        opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}bolsz.txt"

    pp.pprint(vars(opts))

    num_samples = opts.num_samples
    batch_size = 1000  # Adjust batch size as needed

    # Generate a consistent list of seeds
    rng = np.random.RandomState(opts.seed)
    seeds = rng.randint(0, 2**32, size=num_samples)
    indices = np.arange(num_samples)

    # Create a dedicated temporary directory for the program
    program_temp_dir = os.path.join(tempfile.gettempdir(), "tsp_temp")
    os.makedirs(program_temp_dir, exist_ok=True)

    total_written = 0
    batch_number = 0

    with ProcessPoolExecutor() as executor, open(opts.filename, 'w') as f:
        pbar = tqdm(total=num_samples, desc='Generating TSP samples')
        while total_written < num_samples:
            remaining = num_samples - total_written
            current_batch_size = min(batch_size, remaining)

            # Get the seeds and indices for the current batch
            batch_seeds = seeds[total_written:total_written+current_batch_size]
            batch_indices = indices[total_written:total_written+current_batch_size]

            # Create arguments for the worker function
            args_list = [
                (idx, seed, opts.min_nodes, opts.max_nodes, opts.node_dim)
                for idx, seed in zip(batch_indices, batch_seeds)
            ]

            results = executor.map(generate_and_solve_tsp_instance, args_list)
            # Store results in a list to sort them by index
            ordered_results = sorted(results, key=lambda x: x[0])

            for idx, result in ordered_results:
                if result:
                    f.write(result)
                    total_written += 1
                    pbar.update(1)
                else:
                    # Handle invalid result if needed
                    pass

            # Periodically clean up only program's temp files
            cleanup_temp_files(program_temp_dir)
            batch_number += 1

    # Remove the program's temporary directory at the end
    shutil.rmtree(program_temp_dir, ignore_errors=True)

    print(f"Completed generation of {num_samples} samples.")
