# Description of homework1 dataset

There are two datasets in hw1_data.zip file, including a Synthetic data and a real world data.

## File format
1. "number".txt contains the edge list of a graph. (i.e. node1 node2)
2. "number"_score.txt contains the exact BC value of each node.

## Synthetic data
1. The data are generated by networkx package with  powerlaw_cluster_graph function.(n=5000, m=4, p=0.05)
2. In hw1_data/Synthetic/5000 folder, there are 30 generated graphs and the exact BC value of each graph.
3. You need to report the averaged metrics with these 30 graphs.

## Real world data
The com-youtue data includes the social network which the author used in the paper.
