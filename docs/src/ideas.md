# Ideas

## Parallelization

## New Nearest Neighbor computation

- Use [NearestNeighborDescent](https://github.com/dillondaudert/NearestNeighborDescent.jl)

The `DescentGraph` constructor builds the approximate kNN graph:
```jl
DescentGraph(data, n_neighbors, metric; max_iters, sample_rate, precision)
```
The k-nearest neighbors can be accessed through the `indices` and `distances`
attributes. These are both `KxN` matrices containing ids and distances to each
point's neighbors, respectively, where `K = n_neighbors` and `N = length(data)`.

Example:
```jl
using NearestNeighborDescent
data = [rand(10) for _ in 1:1000] # or data = rand(10, 1000)
n_neighbors = 5

# nn descent search
graph = DescentGraph(data, n_neighbors)

# access point i's jth nearest neighbor:
graph.indices[j, i]
graph.distances[j, i]
```

Once constructed, the `DescentGraph` can be used to find the nearest
neighbors to new points. This is done via the `search` method:
```jl
search(graph, queries, n_neighbors, queue_size) -> indices, distances
```

Example:
```jl
queries = [rand(10) for _ in 1:100]
# OR queries = rand(10, 100)
idxs, dists = search(knngraph, queries, 4)
```

# Questions

- Revoir les notions de pas de temps / nombre d'observations
