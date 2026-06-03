import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import sys


def plot_topology_matrix(topology_matrix, save_name):
    topology_matrix = np.asarray(topology_matrix)
    if topology_matrix.ndim > 2:
        axes = tuple(range(topology_matrix.ndim - 2))
        topology_matrix = topology_matrix.mean(axis=axes)

    num_nodes = topology_matrix.shape[-1]
    plt.imshow(topology_matrix, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.colorbar()

    tick_marks = np.arange(num_nodes)
    plt.xticks(tick_marks, tick_marks, fontsize=8)
    plt.yticks(tick_marks, tick_marks, fontsize=8)

    plt.savefig(save_name)
    plt.clf()


def main():
    # Example: Visualize skeleton topology for a specific sample
    # Usage: python visualize.py [sample_id] [output_file]
    
    sample_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    save_name = sys.argv[2] if len(sys.argv) > 2 else f'vis_graph_sample_{sample_id}.jpg'
    
    try:
        get_graph = np.load('graph.npy', allow_pickle=True)
    except FileNotFoundError:
        print("Error: graph.npy not found!")
        print("Need to generate graph.npy from model inference first.")
        return
    
    if sample_id >= len(get_graph):
        print(f"Error: Sample {sample_id} out of range. Total samples: {len(get_graph)}")
        return
    
    vis_graph = np.asarray(get_graph[sample_id])
    print(f"Visualizing sample {sample_id}...")
    plot_topology_matrix(vis_graph, save_name)
    print(f"Saved to {save_name}")


if __name__ == '__main__':
    main()
