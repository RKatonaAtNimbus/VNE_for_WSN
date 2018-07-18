'''Virtual Network Embedding Algorithm'''

import networkx as nx


WSN = nx.Graph()  #represents the substrate network resources
VNR = set()     #virtual network requests
M = None       #comitted mappings

#a conflict graph representing interference is required
CG = nx.Graph()

def get_vnrs(vnr_list):


    vnr1 = (1000, (34, {'load': 17}, 1, {'load': 17}), {'load': 17, 'plr': 40})
    vnr2 = (1000, (14, {'load': 12}, 1, {'load': 12}), {'load': 12, 'plr': 40})
    vnr3 = (1000, (3, {'load': 10}, 1, {'load': 10}), {'load': 10, 'plr': 40})
    vnr4 = (1000, (22, {'load': 8}, 1, {'load': 8}), {'load': 8, 'plr': 40})
    vnr5 = (1000, (37, {'load': 7}, 1, {'load': 7}), {'load': 7, 'plr': 40})
    vnr6 = (1000, (6, {'load': 6}, 1, {'load': 6}), {'load': 6, 'plr': 40})
    vnr7 = (1000, (13, {'load': 11}, 1, {'load': 11}), {'load': 11, 'plr': 40})
    vnr8 = (1000, (32, {'load': 5}, 0, {'load': 5}), {'load': 5, 'plr': 40})
    vnr9 = (1000, (38, {'load': 3}, 0, {'load': 3}), {'load': 3, 'plr': 40})

#    vnr1 = (1000, (4, {'load': 17}, 0, {'load': 17}), {'load': 17, 'plr': 40})
#    vnr2 = (1000, (3, {'load': 12}, 0, {'load': 12}), {'load': 12, 'plr': 40})
#    vnr3 = (1000, (76, {'load': 10}, 109, {'load': 10}), {'load': 10, 'plr': 40})
#    vnr4 = (1000, (62, {'load': 8}, 109, {'load': 8}), {'load': 8, 'plr': 40})
#    vnr5 = (1000, (97, {'load': 7}, 109, {'load': 7}), {'load': 7, 'plr': 40})
#    vnr6 = (1000, (144, {'load': 6}, 109, {'load': 6}), {'load': 6, 'plr': 40})
#    vnr7 = (1000, (19, {'load': 11}, 109, {'load': 11}), {'load': 11, 'plr': 40})
#    vnr8 = (1000, (32, {'load': 5}, 109, {'load': 5}), {'load': 5, 'plr': 40})
#    vnr9 = (1000, (38, {'load': 3}, 109, {'load': 3}), {'load': 3, 'plr': 40})





    return vnr_list
 #   return [vnr1,vnr2,vnr3,vnr4,vnr5,vnr6,vnr7]
#    return [vnr5,vnr4,vnr3,vnr2,vnr1]
#    return [vnr9, vnr8, vnr6, vnr5, vnr4]
#    return [vnr4, vnr5, vnr6, vnr8, vnr9]








