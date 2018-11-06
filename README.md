dyngraphplot
===

dyngraphplot is a Python module for the drawing of dynamic force-directed graphs that change over time. It is based on the algorithm by Frishman, Tal in the paper:
[Online Dynamic Graph Drawing](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4433990)

This is a simplified, non-parallel version of that algorithm without the
partitioning steps, but this way it's easier to implement and use, while
performance should still be sufficient for smaller-size graphs.

The implementation heavily relies on the matplotlib and networkx modules.

Installation
----

To install dyngraphplot, do:
```sh
$ pip install dyngraphplot
```

Simple Example
----

Then, to initialize and plot a graph:
```python
import networkx as nx
from dyngraphplot import DynGraphPlot

# create a random graph and plot it
G = nx.fast_gnp_random_graph(50, 0.1)
plot = DynGraphPlot(G)
```

And afterwards, to update the graph:
```python
# update nodes and edges in the graph
new_nodes = [50,51]
new_edges = [(50,20),(51,30), (50,51)]
plot.update(new_nodes, new_edges)
```

Note that `update` returns the updated `networkx.Graph` object, so you can do:
```python
# update a plot, get result and close plot
new_nodes = [50,51]
new_G = plot.update(new_nodes)
new_layout = plot.layout
plot.close()
```

Usage
----

`DynGraphPlot()` is used to initialize the plot and takes as arguments:
  - `G`: NetworkX graph or any object that is valid for `networkx.Graph()`

  - `mode`: Drawing mode for the plot, options are:
    - `'non-blocking'`: Show the plot and update it without blocking running proccess (default)
    - `'blocking'`: Show the plot, block running proccess, must close plot to resume (matplotlib bug doesn't apply in this mode, window is responsive)
    - `'save'`: Save the dynamic graph as a sequence of files in a directory
    - `'hidden'`: Don't plot the graph at all (useful for getting layout x,y of nodes)   
  
  - `plot_box`: Plot position and shape, format: `[x, y, width, height]`

  - `save_dir`: Directory to save the plot sequence in `'save'` mode

  - `save_name`: Filename for the plot files in `'save'` mode (default: `graph.png`)
    individual graphs will be saved as `graph_0.png`, `graph_1.png`, etc.

  - `draw_options`: Graph visual options for `networkx.draw()`  
    arguments are the same as `networkx.draw()`, plus:
    - `edgecolors`: Set color of node borders
    - `edge_labels`: Edge labels in a dictionary keyed by edge two-tuple of text labels
    - `edge_label_attr`: Name of edge attribute to be used as edge label, edges that have that attribute are drawn with it as label

  - `initial_layout`: NetworkX layout function (default: `networkx.spring_layout`)

  - `initial_layout_params`: Parameters dictionary for initial layout (default: `{'k': 0.8}`)  
    _Note: `k` is the elasticity parameter for spring layout_
  
  - `dynamic_layout_params`: Parameters dictionary for dynamic layout
    - `pos_radius`: # radius for placing unconnected new nodes (default: 0.618)
    - `pos_angle`: # angle step for placing unconnected new nodes (default: 3)  
      _Note: this shouldn't be multiple of pi or nodes will often overlap_
    - `pos_score_same`: positioning confidence score for unmoved nodes (default: 1)
    - `pos_score_2`: positioning confidence score for nodes with 2+ placed neighbors (default: 0.25)
    - `pos_score_1`: positioning confidence score for nodes with 1 placed neighbor (default: 0.1)
    - `pos_score_0`: positioning confidence score for nodes without placed neighbors (default: 0)
    - `pin_a`: pinning rigidity, see paper (default: 0.6)
    - `pin_k`: pinning cutoff, see paper (default: 0.5)
    - `pin_weight`: initial pinning weight, see paper (default: 0.35)
    - `force_K`: optimal geometric distance, see paper (default: 0.1)
    - `force_lambda`: temperature decay constant, see paper (default: 0.9)
    - `force_iteration_count`: number of layout iterations (default: 50)
    - `force_dampen`: dampening factor for force application (default: 0.1)
    
<br/>

`DynGraphPlot.update()` is used to update the plot and takes as arguments:
  - `new_nodes`: Iterable containing nodes added in this update
  
  - `new_edges`: Iterable containing edges added in this update
  
  - `rmv_nodes`: Iterable containing nodes removed in this update
  
  - `rmv_edges`: Iterable containing edges removed in this update
  
<br/>

`DynGraphPlot.close()` is used to close the plot

<br/>

A `DynGraphPlot` object also has a number of useful accessible properties:
 - `G`: NetworkX object with the current graph

 - `options`: the `draw_options` used

 - `params`: the `dynamic_layout_params` used

 - `layout`: Dictionary with `[x,y]` position of every node
 
 - `figure`: The matplotlib figure being drawn
 
 - `ax`: The axis of the matplotlib figure


Notes
----

 - **GUI Freeze:** A persistent bug in the matplotlib module causes the plot window to freeze when in interactive (non-blocking) mode. Unforunately, as dyngraphplot needs to draw dynamic graphs, the interactive mode must be used and to allow figure updating. This means that the plot windows generated by dyngraphplot can't be interacted with and therefore can't be resized, zoomed in or saved.
 I'm exploring possible solutions to this using a threading / multiproccessing architecture, but it may be unstable, back-end specific or impossible.

 - **Window position and size:** Because of the previously GUI freeze problem, dyngraphplot includes a _(hopefully)_ OS and back-end agnostic way to set the plot window position and size, since it won't be possible to resize it later.

 - **Node borders:** Normally networkx.draw() fails to pass the `edgecolors` attribute to `matplotlib.scatter`, which means nodes are drawn without borders. A workaround is included in dyngraphplot that manually does this, although this causes a very minor visual bug where labels of overlapping nodes may be drawn with incorrect depths.

License
----

MIT
