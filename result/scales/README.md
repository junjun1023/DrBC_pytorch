# Train
- 10,000 randon generate graph
- [trainset](./../hw1_data/train/200)
- networkx.generators.random_graphs.powerlaw_cluster_graph(n=200, m=4, p=0.05)

# Valid
- 100 randon generate graph
- [validset](./../hw1_data/valid/200)
- networkx.generators.random_graphs.powerlaw_cluster_graph(n=200, m=4, p=0.05)


# Others
- loss: BCELoss
- GT: sigmoid(bc[det] - bc[src])
- PR: sigmoid(bc[det] - bc[src])
- epochs: 500
- Save model based on best validation evaluation of top 10

# Result
![](train_plt.png)
