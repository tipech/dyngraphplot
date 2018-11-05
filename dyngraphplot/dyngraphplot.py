# -*- coding: utf-8 -*-
"""Dynamic graph visualization with matplotlib

    This module handles drawing of dynamic force-directed graphs that change
    over time. It is based on the algorithm by Frishman, Tal in the paper:

    Online Dynamic Graph Drawing, IEEE Trans. on Visualizations & Graphics
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4433990

    This is a simplified, non-parallel version of that algorithm without the
    partitioning steps, but this way it's easier to implement and use, while
    performance should still be sufficient for smaller-size graphs.
    The implementation heavily relies on the networkx module.

    Example:
        from dyngraphplot import DynGraphPlot
        G = nx.fast_gnp_random_graph(50, 0.1)
        visualizer = DynGraphPlot(G)
        visualizer.update(G, [50,51], [(50,20),(51,30), (50,51)])

"""

import math, tkinter, matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class DynGraphPlot():
    """Handles plotting and visualization of dynamic graphs.

    At initialization, launch the matplotlib figure window and draws the
    initial graph

    Args:
        G: NetworkX graph to be drawn
        window_box: Window position and shape, format: [x, y, width, height]
        draw_options: Graph visual options for networkx.draw()
            also supports edge_attr_label: select attribute as edge label
        initial_layout: NetworkX layout function (default: nx.spring_layout)
        initial_layout_params: Parameters dictionary for initial layout
            (default: {'k': 0.8})
        dynamic_layout_params: Parameters dictionary for dynamic layout

    Attributes:
        G: NetworkX graph or any object that is valid for networkx.Graph()
        options: Graph visual options for networkx.draw()
        params: Parameters dictionary for dynamic layout
        layout: Dictionary with [x,y] position of every node
        figure: The matplotlib figure being drawn
        ax: The axis of the matplotlib figure being drawn with the graph


    """
    
    def __init__(self, graph, window_box=None, draw_options={},
            initial_layout=nx.spring_layout,        # default layout is spring
            initial_layout_params={'k': 0.8},    # spring elasticity
            dynamic_layout_params={}):

        # if it's a networkX graph, save it as is
        if isinstance(graph, nx.Graph) or issubclass(type(graph), nx.Graph):
            self.G = graph

        # otherwise read it into a graph (won't be directed)
        else:
            self.G = nx.Graph(graph)

        # set display options and update with any user specified
        self.options = {'node_size': 800,
                        'node_color': '#80B3F2B3', # Light blue
                        'linewidth': 0.2,
                        'edgecolors': 'black', 
                        'width': 0.5,
                        'font_size': 8}
        self.options.update(draw_options)

        # set layout properties and update with any user specified (see paper)
        self.params = { 'pos_radius': 0.62, # reasonable radius
                            'pos_angle': 3,      # mustn't be multiple of pi
                            'pos_score_same': 1, # for unmoved obj
                            'pos_score_2': 0.25, # obj with 2+ placed neighbrs
                            'pos_score_1': 0.1,  # obj with 1 placed neighbors
                            'pos_score_0': 0,    # obj without placed neighbrs
                            'pin_a': 0.6,        # pinning rigidity
                            'pin_k': 0.5,        # pinning cutoff
                            'pin_weight': 0.35,  # initial pinning weight
                            'force_K': 0.8,      # optimal geometric distance
                            'force_lambda': 0.8, # temperature decay constant
                            'force_iteration_count': 50 # layout iterations
                            }
        self.params.update(dynamic_layout_params)

        # apply initial layout
        self.layout = initial_layout(self.G, **initial_layout_params)
        
        self._count = 0 # count unplaceble that we put in circle around center

        # plotting graph
        plt.autoscale(tight=True)             # configure autoscaling
        plt.ion()                             # matplotlib interactive mode
        self.figure, self.ax = plt.subplots() # setup figure
        self.set_figure_window(window_box)    # position window
        self.ax.axis("off")                   # hide axis
        self.ax.set_position([0, 0, 1, 1])    # use entire box
        self.options.update({'ax': self.ax})  # include axis in options
        self.draw()                           # draw initial graph


    def update(self, new_nodes=[], new_edges=[], rmv_nodes=[], rmv_edges=[]):
        """Update the graph plot when new data is coming.

        Args:
            new_nodes: Nodes added in this update
            new_edges: Edges added in this update
            rmv_nodes: Nodes removed in this update
            rmv_edges: Edges removed in this update

        Returns:
            The modified NetworkX graph, in case update() was used for changes

        """

        # add nodes implicitly created from edges to new_nodes
        new_nodes = list(new_nodes) + [node for edge in new_edges
            for node in [edge[0],edge[1]] if node not in self.G.nodes]

        # all the nodes that will be affected but remain in graph
        affected_nodes = (set(list(new_nodes)
            + [node for edge in new_edges for node in [edge[0],edge[1]]]
            + [node for edge in rmv_edges for node in [edge[0],edge[1]]]
            + [neighbor for node in rmv_nodes
                for neighbor in self.G.neighbors(node)])
            - set(rmv_nodes))

        # step 1 of paper
        self.position_new_nodes(new_nodes, new_edges)

        # now that nodes were added, apply other changes
        self.G.add_edges_from(new_edges)
        self.G.remove_edges_from(rmv_edges)
        self.G.remove_nodes_from(rmv_nodes)

        # steps 2, 4 and 6 of paper
        self.pin_nodes(affected_nodes)
        new_layout = self.move_nodes()
        self.interpolate_nodes(new_layout)

        # draw results
        self.draw()

        # return the updated network
        return self.G


    def position_new_nodes(self, new_nodes, new_edges):
        """Handle initial positioning of new nodes.

        Args:
            new_nodes: Nodes added in this update
            new_edges: Edges added in this update

        """

        # old nodes have perfect positioning confidence 
        nx.set_node_attributes(self.G, self.params['pos_score_same'],
            'pos_score')

        # place new nodes (assuming they were added in same order)
        for new_node in new_nodes:
            
            # get new edges of the new_node in new_node-first tuple list
            node_edges = ([(edge[0], edge[1]) for edge in new_edges
                            if edge[0] == new_node]
                        + [(edge[1], edge[0]) for edge in new_edges
                            if edge[1] == new_node])

            # edges connecting to old nodes and can be used for positioning
            pos_edges = [edge for edge in node_edges if edge[1]
                in list(self.G.nodes)]

            # get the positions of the old nodes
            pos_nodes = [tuple(self.layout[edge[1]]) for edge in pos_edges]

            # multiple positioning connections
            if len(pos_nodes) > 1:

                #find barycenter and add to layout
                self.layout[new_node] = np.mean(pos_nodes, axis=0)

                # more old links, better positioning confidence
                pos_score = self.params['pos_score_2'] # medium confidence

            # single positioning connection
            elif len(pos_nodes) == 1:

                # place new node 50% further away from center
                self.layout[new_node] = np.array([pos_nodes[0][0] * 1.2,
                    pos_nodes[0][1] * 1.2])

                pos_score = self.params['pos_score_1'] # low confidence

            # no positioning connections
            else:

                # place in circle around center
                radius = self.params['pos_radius']
                angle = self._count * self.params['pos_angle'] # rotate
                self._count += 1

                # rotate based on count, nodes wont overlap for 300 rotations
                pos_x = radius * math.cos(angle)
                pos_y = radius * math.sin(angle)
                self.layout[new_node] = np.array([pos_x, pos_y])

                pos_score = self.params['pos_score_1'] # no confidence

            self.G.add_node(new_node, pos_score=pos_score)


    def pin_nodes(self, affected_nodes):
        """Calculates the pinning weights of nodes.

        Args:
            affected_nodes: Nodes whose neighborhood changes but remain in G

        """
        a = self.params['pin_a'] # higher a -> lower neighbor influence
        k = self.params['pin_k'] # D cutoff parameter
        w_initial_pin = self.params['pin_weight'] # initial pin weight

        # local pin weights calculation sweep
        for node in self.G.nodes:

            # calculate average pos_score of neighbors
            scores = [self.G.node[neighbor]['pos_score']
                for neighbor in self.G.neighbors(node)]
            if len(scores) > 0:
                neighbor_score = np.mean(scores)
            else:
                neighbor_score = 0
            
            # calculate pin weight
            self.G.node[node]['pin_weight'] = (
                a * self.G.node[node]['pos_score']
                 + (1 - a) * neighbor_score)

        D = {}    # Distance class dictionary (see paper)

        # global pin weights calculation
        D[0] = affected_nodes
        remaining_nodes = set(self.G.nodes) - D[0]

        i = 1 # distance-to-modification

        # repeated sweeps until nodes classified by distance to new ones
        while len(remaining_nodes) > 0:

            # calculate neighbors of previous D
            neighbors = set([ neighbor for node in D[i-1]
                for neighbor in list(self.G.neighbors(node))])

            # calculate next distance class, restrict to only unvisited nodes
            new_D = neighbors.intersection(remaining_nodes)

            # if we can't get to any more nodes
            if len(new_D) == 0:
                remaining_nodes = set()    # empty remaining to terminate loop
                i -= 1                     # dmax

            else:
                # store new D, discard visited nodes
                D[i] = new_D
                remaining_nodes = remaining_nodes - D[i]

                # going deeper (further)
                i += 1

        d_cutoff = k * i # distance cutoff

        # global sweep, assign pinning weights to classes of nodes
        for i in D:

            # special case where new nodes affect entire graph
            if d_cutoff == 0:

                # set pin weight to zero for all nodes
                for node in D[i]:
                    self.G.node[node]['pin_weight'] = 0

            # nodes beyond cutoff
            elif i > d_cutoff:

                # set pin weight to one for all nodes
                for node in D[i]:
                    self.G.node[node]['pin_weight'] = 1

            # nodes before cutoff
            else:

                # set pin weight according to formula
                for node in D[i]:
                    self.G.node[node]['pin_weight'] = (w_initial_pin
                        ** (1 - i / d_cutoff))


    def move_nodes(self):
        """Calculate the force applied to each node and move it.

        Returns:
            The new layout dictionary with node positions changed

        """
        K = self.params['force_K']      # optimal geometric node distance
        K2 = K**2                           # pre-computing square of K
        t = K * math.sqrt(len(self.G))      # initial annealing temp,see paper
        l = self.params['force_lambda'] # temperature decay constant

        iter_count = self.params['force_iteration_count'] # nr of iters
        frac_done = 0                       # fraction counter
        frac_increment = 1 / iter_count     # fraction counter increment

        new_layout = self.layout.copy()     # will hold the next layout

        # iterations loop
        for i in range(1, iter_count):
            
            # force calculation loop, O(n^2) because no partitioning
            for v in self.G.nodes:
                if frac_done > self.G.node[v]['pin_weight']:

                    pos_v = self.layout[v] # get position of u


                    # calculate repulsion to all other nodes
                    F_repulsion = sum([
                        self.calculate_repulsion(pos_v,self.layout[u], K2)
                        for u in self.G.nodes if u != v])

                    # calculate attraction to connected nodes
                    F_attraction = sum([
                        self.calculate_attraction(pos_v, self.layout[u], K)
                        for u in self.G.neighbors(v)])

                    # calculate total force and move object
                    F_total = F_repulsion + F_attraction

                    # make sure F_total isn't 0 or 0,0 because no other nodes
                    if (isinstance(F_total, np.ndarray) and
                        (F_total[0] != 0 or F_total[0] != 0)):

                        # calculate magnitude amd adjust position accordingly
                        F_total_mag = math.sqrt(F_total[0]**2 + F_total[1]**2)
                        new_layout[v] += (min(t, F_total_mag)
                            * F_total / F_total_mag)


            # increment controls for next iteration
            t *= l
            frac_done += frac_increment

        return new_layout


    def calculate_repulsion(self, pos_v, pos_u, K2):
        """Calculate the repulsion force between two nodes.

        Args:
            pos_v: Position of first node v
            pos_u: Position of second node u
            K2: Optimal distance parameter^2, precomputed to save on time

        """
        diff = pos_v - pos_u

        # in case points overlap
        if diff[0] == 0 and diff[1] == 0:
            return [K2, K2]

        # otherwise use formula
        else:
            return K2 * diff / abs(diff[0]**2 + diff[1]**2)


    def calculate_attraction(self, pos_v, pos_u, K):
        """Calculate the attraction force between two nodes.

        Args:
            pos_v: Position of first node v
            pos_u: Position of second node u
            K2: Optimal distance parameter

        """
        diff = pos_v - pos_u

        # in case points overlap
        if diff[0] == 0 and diff[1] == 0:
            return [0, 0]

        # otherwise use formula
        else:
            return math.sqrt(diff[0]**2 + diff[1]**2) * diff / K


    def interpolate_nodes(self, new_layout):
        """Interpolate positions of nodes between layout changes.

        Args:
            new_layout: The new layout dictionary with node position changes

        """

        # go through moved nodes
        for node in self.G.nodes:
            if (new_layout[node][0] != self.layout[node][0] or 
                new_layout[node][1] != self.layout[node][1]):

                # linearly interpolate last new with old
                self.layout[node] = ((1 - self.G.node[node]['pin_weight'])
                    * (new_layout[node] - self.layout[node]))


    def draw(self):
        """Draw or update the drawing of the current graph plot."""

        # insert new layout in options
        options = self.options
        options.update({'pos': self.layout})

        plt.cla() #clear previous drawing

        # workaround for draw_networkx() not passing edgecolors to scatter.
        nodes = nx.draw_networkx_nodes(self.G, **options)
        if nodes is not None:
            nodes.set_edgecolor(options['edgecolors'])
        nx.draw_networkx_edges(self.G, **options)
        nx.draw_networkx_labels(self.G, **options)

        edge_labels = {}

        # edge label dictonary was passed
        if 'edge_labels' in self.options:
            edge_labels = self.options['edge_labels']

        # edge attribute was set as label
        if 'edge_label_attr' in self.options:

            # update edge_labels dictionary with label edge attribute
            # but for each edge only if that edge has that attribute
            edge_labels.update({
                (edge[0], edge[1]): edge[2][self.options['edge_label_attr']]
                for edge in self.G.edges(data=True)
                if self.options['edge_label_attr'] in edge[2]})

        # if any edge labels were actually specified, draw them
        if len(edge_labels) > 0:
            options['edge_labels'] = edge_labels
            nx.draw_networkx_edge_labels(self.G, **options)

        self.figure.canvas.draw()                 # draw graph
        self.figure.canvas.start_event_loop(0.01) # freeze fix


    def set_figure_window(self, window_box):
        """Set figure window upper left corner and size in pixels.

        Sometimes may not work depending on OS and/or GUI backend.

        Args:
            window_box: Window position and shape, format: [x,y,width,height]
        
        """

        # if actual arguments were provided
        if window_box != None:

            # get arguments
            x, y, width, height = window_box

            # get screen PPI
            tk_root = tkinter.Tk()
            ppi = tk_root.winfo_screenwidth() / (tk_root.winfo_screenmmwidth()
                / 25.4)
            tk_root.destroy()

            # set the size
            self.figure.set_size_inches(width / ppi, height / ppi,
                forward=True)

            # get the matplotlib backend for the position
            backend = matplotlib.get_backend()

            # depending on backend, use appropriate method
            if backend == 'TkAgg':
                self.figure.canvas.manager.window.wm_geometry("+%d+%d"
                    % (x, y))
            elif backend == 'WXAgg':
                self.figure.canvas.manager.window.SetPosition((x, y))
            else:
                # This works for QT and GTK
                # You can also use window.setGeometry
                self.figure.canvas.manager.window.move(x, y)


    def close(self):
        """Close the figure window."""

        plt.close(self.figure)