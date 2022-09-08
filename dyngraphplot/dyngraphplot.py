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
        inport networkx as nx
        from dyngraphplot import DynGraphPlot
        plot = DynGraphPlot(nx.fast_gnp_random_graph(50, 0.1))
        plot.update([50,51], [(50,20),(51,30), (50,51)])

"""

import os, math, copy, matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# suppress matplotlib warning caused by networkx draw()
import warnings
warnings.simplefilter("ignore",
    category=matplotlib.cbook.MatplotlibDeprecationWarning)


class DynGraphPlot():
    """Handles plotting and visualization of dynamic graphs.

    At initialization, launch the matplotlib figure window and draws the
    initial graph

    Args:
        G: NetworkX graph to be drawn
        mode: drawing mode, options: 'blocking','non-blocking','save','hidden'
            (default: 'non-blocking')
        save_dir: the directory to save figure sequence in (if in 'save' mode)
        save_name: the figure save_name for display or saving purposes
        plot_box: Window position and shape, format: [x, y, width, height]
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
    
    def __init__(self, G, mode="non-blocking",
            save_dir="./",
            save_name="graph",
            plot_box=None,
            draw_options={},
            initial_layout=nx.spring_layout,
            initial_layout_params={'k': 0.8},
            dynamic_layout_params={}):

        # if it's a networkX graph, save it as is
        if isinstance(G, nx.Graph) or issubclass(type(G), nx.Graph):
            self.G = G

        # otherwise read it into a graph (won't be directed)
        else:
            self.G = nx.Graph(G)

        # get drawing mode
        if mode in ['blocking','non-blocking','save','hidden']:
            self.mode = mode
        else:
            raise(ValueError("Invalid plot mode!"))

        # store arguments
        self.save_dir = save_dir
        self.save_name = save_name
        self.plot_box = plot_box

        # setup store directory
        if mode == 'save':
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

        # set display options and update with any user specified
        self.options = {'node_size': 800,
                        'node_color': '#80B3F2B3', # Light blue
                        'linewidth': 0.2,
                        'edgecolors': 'black', 
                        'width': 0.5,
                        'font_size': 8}
        self.options.update(draw_options)

        # set layout properties and update with any user specified (see paper)
        self.params = { 'pos_radius': 0.5,       # reasonable radius
                            'pos_angle': 3,      # mustn't be multiple of pi
                            'pos_score_same': 1, # for unmoved obj
                            'pos_score_2': 0.25, # obj with 2+ placed neighbrs
                            'pos_score_1': 0.1,  # obj with 1 placed neighbors
                            'pos_score_0': 0,    # obj without placed neighbrs
                            'pin_a': 0.6,        # pinning rigidity
                            'pin_k': 0.5,        # pinning cutoff
                            'pin_weight': 0.35,  # initial pinning weight
                            'force_K': 0.1,      # optimal geometric distance
                            'force_lambda': 0.9, # temperature decay constant
                            'force_iteration_count': 50, # layout iterations
                            'force_dampen': 0.1  # motion dampening factor
                            }
        self.params.update(dynamic_layout_params)

        # apply initial layout
        self.layout = initial_layout(self.G, **initial_layout_params)
        
        self._frame = 0 # count how many updates there have been
        self._count = 0 # count unplaceble that we put in circle around center

        # plotting graph

        # if non-blocking, enable matplotlib interactive mode
        if self.mode == 'non-blocking':
            plt.ion()
            self.setup_figure()  # setup the plot

        if self.mode != 'hidden':
            self.draw()          # draw initial graph



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
                for neighbor in nx.all_neighbors(self.G, node)])
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

        self._frame += 1    # count one more update

        # draw results
        if self.mode != 'hidden':
            self.draw()


        # return the updated network
        return self.G


    def position_new_nodes(self, new_nodes, new_edges):
        """Handle initial positioning of new nodes.

        Args:
            new_nodes: Nodes added in this update
            new_edges: Edges added in this update

        """

        # if graph has nodes, get size of the plot box, i.e. max(width,height)
        if len(self.G) > 0:
            size = max(np.ptp([node for node in self.layout.values()], 0))
        else:
            size = 2 # usual matplotlib size

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
                radius = 0.5 * size * self.params['pos_radius']
                angle = self._count * self.params['pos_angle'] # rotate
                self._count += 1

                # rotate based on count, nodes wont overlap for 300 rotations
                pos_x = radius * math.cos(angle)
                pos_y = radius * math.sin(angle)
                self.layout[new_node] = np.array([pos_x, pos_y])

                pos_score = self.params['pos_score_0'] # no confidence

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

            # if there are neighbors
            if len(list(nx.all_neighbors(self.G, node))) > 0:
            
                # calculate average pos_score of neighbors
                scores = [self.G.nodes[neighbor]['pos_score']
                    for neighbor in nx.all_neighbors(self.G, node)]
                neighbor_score = np.mean(scores)

                # calculate pin weight
                self.G.nodes[node]['pin_weight'] = (
                    a * self.G.nodes[node]['pos_score']
                     + (1 - a) * neighbor_score)

            # otherwise pin weight is just the node's pos_score
            else:
                node_attributes = self.G.nodes[node]
                node_attributes['pin_weight'] = node_attributes['pos_score']

        D = {}    # Distance class dictionary (see paper)

        # global pin weights calculation
        D[0] = affected_nodes
        remaining_nodes = set(self.G.nodes) - D[0]

        i = 1 # distance-to-modification

        # repeated sweeps until nodes classified by distance to new ones
        while len(remaining_nodes) > 0:

            # calculate neighbors of previous D
            neighbors = set([ neighbor for node in D[i-1]
                for neighbor in list(nx.all_neighbors(self.G, node))])

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


            # special case, new nodes are disconnected or affect entire graph
            if d_cutoff == 0:


                # set pin weight to zero for all nodes
                for node in D[i]:
                    self.G.nodes[node]['pin_weight'] = w_initial_pin

            # nodes beyond cutoff
            elif i > d_cutoff:

                # set pin weight to one for all nodes
                for node in D[i]:
                    self.G.nodes[node]['pin_weight'] = 1

            # nodes before cutoff
            else:

                # set pin weight according to formula
                for node in D[i]:
                    self.G.nodes[node]['pin_weight'] = (w_initial_pin
                        ** (1 - i / d_cutoff))


    def move_nodes(self):
        """Calculate the force applied to each node and move it.

        Returns:
            The new layout dictionary with node positions changed

        """
        K = self.params['force_K']      # optimal geometric node distance
        t = K * math.sqrt(len(self.G))  # initial annealing temp,see paper
        l = self.params['force_lambda'] # temperature decay constant
        d = self.params['force_dampen'] # movement dampening factor

        iter_count = self.params['force_iteration_count'] # nr of iters
        frac_done = 0                       # fraction counter
        frac_increment = 1 / iter_count     # fraction counter increment

        new_layout = copy.deepcopy(self.layout) # will hold the next layout

        # iterations loop
        for i in range(0, iter_count):
            
            # force calculation loop, O(n^2) because no partitioning
            for v in self.G.nodes:
                if frac_done > self.G.nodes[v]['pin_weight']:

                    pos_v = new_layout[v] # get position of u

                    # if there are other nodes in the graph
                    if len(self.G.nodes) > 1:

                        # calculate repulsion to all other nodes
                        F_repulsion = K**2 * sum([
                            self.calculate_repulsion(pos_v,new_layout[u])
                            for u in self.G.nodes if u != v])

                        # if node has neighbors
                        if len(list(nx.all_neighbors(self.G, v))) > 0:

                            # calculate attraction to connected nodes
                            F_attraction = sum([self.calculate_attraction(
                                pos_v, new_layout[u])
                                for u in nx.all_neighbors(self.G, v)
                                if u != v]) / K

                        # node has no neighbors
                        else:

                            # calculate attraction to center of plot
                            F_attraction = (self.calculate_attraction(
                                pos_v, np.array([0, 0])) / K)

                        # calculate total force with dampening
                        F_total = d * (F_repulsion + F_attraction)

                        # calculate magnitude amd adjust position accordingly
                        F_total_mag = math.sqrt(F_total.dot(F_total))
                        F_total_mag = max(F_total_mag, 0.0001) # prevent 0

                        new_layout[v] += (min(t, F_total_mag)
                            * F_total / F_total_mag)


            # increment controls for next iteration
            t *= l
            frac_done += frac_increment

        return new_layout


    def calculate_repulsion(self, pos_v, pos_u):
        """Calculate the repulsion force between two nodes.

        Args:
            pos_v: Position of first node v
            pos_u: Position of second node u

        """
        diff = pos_v - pos_u

        # in case points overlap exactly in either dimension
        if diff[0] == 0:
            diff[0] = 0.0001
        if diff[1] == 0:
            diff[1] = 0.0001

        return diff / abs(diff[0]**2 + diff[1]**2)


    def calculate_attraction(self, pos_v, pos_u):
        """Calculate the attraction force between two nodes.

        Args:
            pos_v: Position of first node v
            pos_u: Position of second node u

        """
        diff = pos_u - pos_v

        # in case points overlap exactly in either dimension
        if diff[0] == 0:
            diff[0] = 0.0001
        if diff[1] == 0:
            diff[1] = 0.0001

        return math.sqrt(diff[0]**2 + diff[1]**2) * diff


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
                self.layout[node] += ((1 - self.G.nodes[node]['pin_weight'])
                    * (new_layout[node] - self.layout[node]))


    def setup_figure(self):
        """Create a plot and configure the figure in it."""

        self.figure, self.ax = plt.subplots() # setup figure
        self.set_figure_window()              # position window
        self.ax.axis("off")                   # hide axis
        self.ax.autoscale(tight=True)         # configure autoscaling
        self.ax.set_position([0, 0, 1, 1])    # use entire box
        self.options.update({'ax': self.ax})  # include axis in options


    def draw(self):
        """Draw or update the drawing of the current graph plot."""

        # re-create figure if not in interactive mode
        if self.mode == 'blocking' or self.mode == 'save':
            self.setup_figure()

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

        self.figure.canvas.draw()   # draw graph

        # blocking mode, show()
        if self.mode == 'blocking':
            plt.show()

        # interactive mode, apply a small delay to allow GUI drawing
        elif self.mode == 'non-blocking':
            self.figure.canvas.start_event_loop(0.01) # freeze fix

        # save mode, save each iteration to a different file
        elif self.mode == 'save':
            directory = self.save_dir.rstrip('/') + '/'

            # make sure extension exists, then split filename,extension
            name = os.path.splitext( self.save_name if '.' in self.save_name
                                else self.save_name + '.png')

            # add frame number in between
            frame_name = name[0] + '_' + str(self._frame) + name[1]
            plt.savefig(directory + frame_name)


    def set_figure_window(self):
        """Set figure window upper left corner and size in pixels.

        Sometimes may not work depending on OS and/or GUI backend.

        Args:
            plot_box: Window position and shape, format: [x,y,width,height]
        
        """

        # if actual arguments were provided
        if self.plot_box != None:

            # use tkinter to get window size (otherwise not needed)
            import tkinter

            # get arguments
            x, y, width, height = self.plot_box

            # get screen PPI
            tk_root = tkinter.Tk()
            ppi = tk_root.winfo_screenwidth() / (tk_root.winfo_screenmmwidth()
                / 25.4)
            tk_root.destroy()

            # set the size
            self.figure.set_size_inches(width / ppi, height / ppi,
                forward=True)

            # if in mode where plot window exists
            if self.mode == 'blocking' or self.mode == 'non-blocking':

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

        # only valid in non-blocking mode
        if self.mode == 'non-blocking':
            plt.close(self.figure)