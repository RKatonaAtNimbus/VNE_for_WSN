"""
This keeps track of nodes on an area.

Nodes are represented by position and there is a 
common radio range constant.

Nodes are stored and sorted based on their distance
to the 0,0 corner of the deployment area, in bands.
"""
from math import sqrt

class Node:
    def __init__(self, x, y, _id=0):
        self.x = x
        self.y = y
        self._id = _id
        self.dist = sqrt(self.x*self.x + self.y*self.y)
        self.neighbs = list()

    def reaches(self, node, radio_range):
        dist = sqrt((self.x - node.x)*(self.x - node.x) + (self.y - node.y)*(self.y-node.y))
        return dist <= radio_range

    def move(self, x, y):
        """Moves the node to the new position.

        Updates the distance from 0,0 and resets the neighbour list.
        """
        self.x = x
        self.y = y
        self.dist = sqrt(self.x*self.x + self.y*self.y)
        self.neighbs = []

    def __repr__(self):
        return str(self._id)

class NodeStore:
    """
    Nodes are stored and sorted based on their distance
    to the 0,0 corner of the deployment area, in bands equal
    to the radio range.

    There is a first level that represents the bands.
    This points to lists of ordered node representations.
    """

    def __init__(self, area_diag, radio_range):
        #print area_diag, type(area_diag), radio_range, type(radio_range)
        self.bands = list([[] for i in range(area_diag/radio_range+1)])
        self.node_list = [] 
        self.area_diag = area_diag
        self.radio_range = radio_range

    def add(self, node):
        """
        Inserts the node into the corresponding
        distance band
        """
        try:
            band = self.bands[int(node.dist/self.radio_range)]
        except IndexError:
            print "Index error, dist =", node.dist
            raise IndexError
        band.append(node)
        self.node_list.append(node)
        
    def find_neighbours(self, node):
        """
        Locates the nodes that are within radio reach of @node
        """
        start_band = int(max(node.dist-self.radio_range, 0)/self.radio_range)
        end_band = int(min(node.dist+self.radio_range, self.area_diag)/self.radio_range)

        neighbs = []
        try:
            for b in self.bands[start_band:end_band+1]:
                for n in b:
                    if n._id != node._id:
                        if node.reaches(n, self.radio_range):
                            neighbs.append(n)
        except TypeError:
            print "start:", start_band, "end:", end_band
            raise TypeError
        return neighbs

    def get_nodes(self):
        """
        Returns a list with all the nodes in the network
        """
        return self.node_list

    def get_max_neighbs(self):
        neighb_count = [len(n.neighbs) for n in self.node_list]
        return max(neighb_count)

    def remove_node(self, node):
        """Removes node from the store"""
        
        # find the node's band
        try:
            band = self.bands[int(node.dist/self.radio_range)]
        except IndexError:
            print "Index error, dist =", node.dist
            raise IndexError
        _idx = -1
        for i, n in enumerate(band):
            if n._id == node._id:
                _idx = i
                break
        if _idx == -1:
            raise KeyError("Couldn't find node %d in store" % node._id)
        del band[_idx]

        # delete node from node list
        _idx = -1
        for i, n in enumerate(self.node_list):
            if n._id == node._id:
                _idx = i
                break
        if _idx == -1:
            raise KeyError("Node %d not found in store node list" % node._id)
        del self.node_list[_idx]
