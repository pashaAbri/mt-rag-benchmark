# Results - Query Rewrite Strategy (All Domain)

| Method                 | Strategy      | R@1  | R@3  | R@5  | R@10 | nDGC@1 | nDGC@3 | nDGC@5 | nDGC@10 |
| :--------------------- | :------------ | :--- | :--- | :--- | :--- | :----- | :----- | :----- | :------ |
| BM25                   | Last Turn     | 0.08 | 0.15 | 0.20 | 0.27 | 0.17   | 0.16   | 0.18   | 0.21    |
|                        | Query Rewrite | 0.09 | 0.18 | 0.25 | 0.33 | 0.20   | 0.19   | 0.22   | 0.25    |
| BM25 (PyTerrier)       | Last Turn     | 0.082| 0.167| 0.215| 0.299| 0.187  | 0.178  | 0.196  | 0.231   |
|                        | Query Rewrite | 0.090| 0.189| 0.250| 0.349| 0.211  | 0.200  | 0.223  | 0.264   |
|                        | MMR Cluster   |      |      |      |      |        |        |        |         |
| BM25 (Elasticsearch)   | Last Turn     |      |      |      |      |        |        |        |         |
|                        | Query Rewrite |      |      |      |      |        |        |        |         |
|                        | MMR Cluster   |      |      |      |      |        |        |        |         |
| BGE-base 1.5           | Last Turn     | 0.13 | 0.24 | 0.30 | 0.38 | 0.26   | 0.25   | 0.27   | 0.30    |
|                        | Query Rewrite | 0.17 | 0.30 | 0.37 | 0.47 | 0.34   | 0.31   | 0.34   | 0.38    |
| BGE-base 1.5 (Ours)    | Last Turn     |      |      |      |      |        |        |        |         |
|                        | Query Rewrite |      |      |      |      |        |        |        |         |
|                        | MMR Cluster   |      |      |      |      |        |        |        |         |
| Elser                  | Last Turn     | 0.18 | 0.39 | 0.49 | 0.58 | 0.42   | 0.41   | 0.45   | 0.49    |
|                        | Query Rewrite | 0.20 | 0.43 | 0.52 | 0.64 | 0.46   | 0.45   | 0.48   | 0.54    |
| Elser (Ours)           | Last Turn     |      |      |      |      |        |        |        |         |
|                        | Query Rewrite |      |      |      |      |        |        |        |         |
|                        | MMR Cluster   |      |      |      |      |        |        |        |         |
