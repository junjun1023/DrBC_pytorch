# Train
- 10,000 randon generate graph
- [trainset](./../hw1_data/train/200)
- networkx.generators.random_graphs.powerlaw_cluster_graph(n=200, m=4, p=0.05)

# Valid
- 100 randon generate graph
- [validset](./../hw1_data/valid/200)
- networkx.generators.random_graphs.powerlaw_cluster_graph(n=200, m=4, p=0.05)


# Different Settings

| GT/PR         | Sigmoid(y_pr)                 | 1/0                           | y_pr                          | BC_hat                        |  |
|---------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|--|
| Sigmoid(y_gt) | [BCE Loss](2021-03-19\ 18-48) |                               |                               |                               |  |
|               | [MSE Loss](2021-03-19\ 18-51) |                               |                               |                               |  |
| 1/0           | [BCE Loss](2021-03-19\ 18-50) | [BCE Loss](2021-03-21\ 14-49) |                               |                               |  |
| y_gt          |                               |                               | [MSE Loss](2021-03-21\ 04-11) |                               |  |
| BC            |                               |                               |                               | [MSE Loss](2021-03-21\ 04-04) |  |

