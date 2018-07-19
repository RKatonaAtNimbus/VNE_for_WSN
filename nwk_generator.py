from cooja_control.dgrm_gen import DGRMConfig
from cooja_control.clustered_net import ClusteredLayout
import cooja_control.csc_test_generator as cooja_gen
from random import shuffle      # so we can shuffle the net coords 


class NwkGenerator(DGRMConfig):
    """A graph with randomly positioned nodes and the edge weights representing
    the PRR, computed with a distance-prr model.
    """

    def __init__(self, num_nodes, avg_cluster_size, randomize=False):
        """Will create the network with num_nodes, filling a square area.
        Nodes are arranged into clusters of width = 2*max_tx_range.

        Params:
        num_nodes           -- number of nodes to be deployed
        avg_cluster_size    -- average number of nodes per cluster
        randomize           -- if set, the node coordinate list is
                               randomly permuted before building the
                               adjacency matrix. Default False
        """
        self.num_nodes = num_nodes
        self.substrate_edges = []
        super(NwkGenerator, self).__init__()
        # generate the random node coordinates
        self.layout = ClusteredLayout(num_nodes, avg_cluster_size, 2)
        self.net_coords = self.layout.generate_net()
        # The number of nodes can change during nwk generation
        self.num_nodes = len(self.net_coords)
        if randomize:
            # Randomizing the list of net coords has the effect of
            # randomizing where the IDs of the nodes are overlaid.
            # So, node X can take different locations with different
            # random permutations of the net coords list.
            shuffle(self.net_coords)
        super(NwkGenerator, self).build_from_net(self.net_coords,
                                                 self.layout.area_width)

    def add_edge(self, src_id, dst_id, rx_ratio, rssi, lqi):
        """Override the method from DGRMConfig.
        Ignore the rssi and lqi.
        """
        #print "New edge ", src_id, dst_id, rx_ratio, rssi, lqi
        self.substrate_edges.append((src_id, dst_id, rx_ratio))
        super(NwkGenerator, self).add_edge(src_id, dst_id, rx_ratio, rssi, lqi)

    def output_csc(self, csc_file):
        """Generates a CSC file so we can visualise the network in Cooja.
        Parameters:
        csc_file    -- path to the csc file to be generated.
        """
        motes = []
        motes.append(cooja_gen.NodeDescriptor(self.net_coords[0][0],
                                              self.net_coords[0][1],
                                              0,
                                              'sky1',
                  '/home/victor/kits/contiki/examples/rime/example-unicast.sky',
                                              'sky'))
        motes.extend([cooja_gen.NodeDescriptor(self.net_coords[i][0],
                                              self.net_coords[i][1],
                                              i,
                                              'sky2',
                  '/home/victor/kits/contiki/examples/rime/example-unicast.sky',
                                              'sky')
                     for i in xrange(1, len(self.net_coords))])
        cooja_gen.csc_test_generator(super(NwkGenerator, self),
                                     motes,
                                     csc_file,
                                     600,
                                     {})


    def output_for_heuristic(self):
        """Generates the adjacency matrix to be used with the heuristic.
        The adj matrix is a dictionary node:[(neighb, (1-prr)*100)]

        Returns:
        Tuple: (network coordinates of nodes, adjacency matrix)
        """
        adjacency_matrix = {}
        for (src,dst,prr) in self.substrate_edges:
            if src in adjacency_matrix:
                adjacency_matrix[src].append((dst, (1-prr)*100))
            else:
                adjacency_matrix[src] = [(dst, (1-prr)*100)]
        return self.net_coords, adjacency_matrix

def output_for_lp(_file, nodes, substrate_edges):
    """Output data to a file object, in the format required by
    PyOMO.
    Returns: -1 if cannot write to file, 0 otherwise.

    Parameters:
    _file       -- File object to write data to.
    nodes       -- List of node coordinates
    substrate_edges -- List of edges, defined as (src, dest, prr, capacity, latency)
    """
    if _file is None: return -1
    try:
        _file.write("\n###############################################\n")
        _file.write("\n# Substrate model\n")
        # Substrate nodes
        _file.write("set V_s := %s;\n"
                     % ' '.join(map(str, range(0, len(nodes)))))
        # Substrate edges
        _file.write("set E_s := %s;\n"
                          % ' '.join([str((s,d)) for (s,d,p,c,l) in substrate_edges]))
        # Link capacity, set at 100
        _file.write("param cap_l := %s;\n"
              % '\n'.join(["%d %d %d" % (s,d,c) for (s,d,p,c,l) in substrate_edges]))
        # Link latency, set at 1ms
        _file.write("param lat := %s;\n"
                  % '\n'.join(["%d %d %d" % (s,d,l) for (s,d,p,c,l) in substrate_edges]))
        # Link reliability
        _file.write("param rel := %s;\n"
                % '\n'.join(["%d %d %0.4f"% (s,d,p) for (s,d,p,c,l) in substrate_edges]))

        # Close this
        _file.write("##################################################")
    except Exception as e:
        print e
        return -1
    return 0

