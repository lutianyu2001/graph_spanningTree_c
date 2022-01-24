# graph_spanningTree_c

This program can return these four artifacts from a given weighted, undirected graph:

* A Depth-First Search (DFS) Tree
* All its Articulation Points (AP) and Biconnected Components (BC)
* A Minimum Spanning Tree (MST), using Kruskal's Algorithm
* The Shortest Path Tree (SPT), using Dijkstra's Algorithm

The input file to the program specifies:

1. the number of vertices (int)
2. the number of edges (int)
3. each edge’s two vertices and its weight (int)

Here's a sample for the input, along with the actual graph it represents:

![sample input](https://raw.githubusercontent.com/lutianyu2001/graph_spanningTree_c/main/sample_input_graph.png)

The program will read the file name from keyboard, read the graph information from the file,
run the above-mentioned four algorithms, and display the results on the console. It will
repeatedly ask for an input data file until the user press "CTRL+Z" then "Enter".

Here's a example for the console output using the sample input:

![sample output](https://raw.githubusercontent.com/lutianyu2001/graph_spanningTree_c/main/sample_output.png)
