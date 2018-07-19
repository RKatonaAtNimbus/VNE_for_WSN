from random import random, randint, seed, sample

class VirtualNetGenerator(object):
    def __init__(self, num_interest_points, max_error, num_links=None):
        """Generates a random virtual network request
        to be deployed on a substrate network.

        A virtual network request contains a list of virtual links.

        Virtual links are defined using
        * interest point where sensing is required, and maximum allowed sensing
            error
        * destination node (0 for now)
        * capacity 0<c<=100
        * latency in ms
        * reliability 0<c<=1 (float)

        Parameters:
        num_interest_points     -- number of available interest points in the 
                                   environment
        max_error               -- range for error values allowed by requests
        num_links               -- number of virtual links in this virtual net.
                                   Not defined by default, in which case it is
                                   random.
        """
        if num_links is None:
            num_links = randint(2, 11)  # Like prev work
        self.links = []
        for l in xrange(num_links):
            capacity = randint(1, 20) # TODO replace with max capacity
            latency = randint(100, 1000)
            reliability = 0.5   # Reliability is fixed 
            #src, dst = sample(xrange(1,num_substrate_nodes), 2)
            # The sink is fixed as node 0
            src = randint(0, num_interest_points-1)
            error = random()*max_error  # Defined for temperature process
            self.links.append(((src, error), 0, capacity, latency, reliability))

    def get_source(self, link):
        """
        Returns the source of the specified link

        Parameters:
        link    -- the index of the link in the VN
        """
        return self.links[link][0]

    def convert_to_heuristic(self):
        return (1000,
                self.links[0][0],   # the interest point index and max error
                0,                  # the base station (destination)
                {                   # link parameters
                    'load': self.links[0][2],       # required load
                    'latency': self.links[0][3],    # required latency
                    'plr': (1-self.links[0][4])*100 # required reliability
                })

    def __repr__(self):
        return str(self.links)

def output_vn_requests_for_lp(_file, vn_reqs):
    if _file is None: return -1

    try:
        _file.write("\n#########################################\n")
        _file.write("# Virtual networks\n")
        e_r_string = ""
        req_l_string = ""
        req_lat_string = ""
        req_rel_string = ""
        for vn in vn_reqs:
            e_r_string += ' '.join(str((s,d)) for ((s,e),d,c,l,r) in vn.links)+' '
            req_l_string += '\n'.join("%d %d %d"%(s,d,c)
                                                for ((s,e),d,c,l,r) in vn.links) +'\n'
            req_lat_string += '\n'.join("%d %d %d"%(s,d,l)
                                                for ((s,e),d,c,l,r) in vn.links) +'\n'
            req_rel_string += '\n'.join("%d %d %0.4f"%(s,d,r)
                                                for ((s,e),d,c,l,r) in vn.links) +'\n'
        _file.write("set E_r := %s;\n" % e_r_string)
        _file.write("param req_l := %s;\n" % req_l_string)
        _file.write("param req_lat := %s;\n" % req_lat_string)
        _file.write("param req_rel := %s;\n" % req_rel_string)
    except Exception as e:
        print e
        return -1

    return 0

def output_vn_links_for_lp(_file, links):
    """
    Basically same as output_vn_requests_for_lp, but works directly on links,
    instead of VNR objects.
    """
    if _file is None: return -1

    try:
        _file.write("\n#########################################\n")
        _file.write("# Virtual networks\n")
        e_r_string = ""
        req_l_string = ""
        req_lat_string = ""
        req_rel_string = ""
        for vn in links:
            e_r_string += ' '.join(str((s,d)) for ((s,e),d,c,l,r) in links)+' '
            req_l_string += '\n'.join("%d %d %d"%(s,d,c)
                                                for ((s,e),d,c,l,r) in links) +'\n'
            req_lat_string += '\n'.join("%d %d %d"%(s,d,l)
                                                for ((s,e),d,c,l,r) in links) +'\n'
            req_rel_string += '\n'.join("%d %d %0.4f"%(s,d,r)
                                                for ((s,e),d,c,l,r) in links) +'\n'
        _file.write("set E_r := %s;\n" % e_r_string)
        _file.write("param req_l := %s;\n" % req_l_string)
        _file.write("param req_lat := %s;\n" % req_lat_string)
        _file.write("param req_rel := %s;\n" % req_rel_string)
    except Exception as e:
        print e
        return -1

    return 0

def output_for_heuristic(_file):
    pass
