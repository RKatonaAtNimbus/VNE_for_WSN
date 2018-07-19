"""
The main module that implements the VNE algorithm using parallel processing.
It takes in a pickle file as input. The arguments of each input vector that represents a test case are described in the README.

For each iteration/test case, first it parses the input vector and creates a WSN object which represents the substrate and 
holds its attributes in a networkx DiGraph. Then it perfoms the node mapping for a batch of requests in the main process serially.
Next the link mapping is performed in parallel using 'p' number of processes where 'p' is the number of available processors.
The solutions of the embedding for all sequences/permutations of a test case are recorded. The results are stored in a pickle file
and the captured metrics are explained in the README.
"""

import sys
from scipy import spatial
import matplotlib.pyplot as plt
import config
import itertools
from wsn_substrate import WSN
import networkx as nx
from link_weight import LinkCost
import time
import visualize as vis
from itertools import islice
import pickle
import copy
import multiprocessing
import math
import networkx as nx


def show_dataStructs(position):
    print "\n @@",position
    print "config.VWSNs",config.VWSNs
    print "config.perms_list",config.perms_list
    print "config.vns_per_perm",config.vns_per_perm
    print "config.perm_prefix",config.perm_prefix
    print "config.current_perm_emb_costs",config.current_perm_emb_costs
    print "config.overall_cost",config.overall_cost
    print "config.best_embeddings",config.best_embeddings
    print "config.max_accepted_vnrs",config.max_accepted_vnrs
    print "config.active_vns",config.active_vns
    print "config.already_mapped_vnrs",config.already_mapped_vnrs
    print "config.current_key_prefix",config.current_key_prefix
    print "config.current_perm",config.current_perm
    print "\n"

def update_node_attribs(nodes, node, load):
    for n, d in nodes.nodes_iter(data=True):
        config.total_operations += 1
        if n == int(node):
            d['load'] = d['load']+int(load)
            #d['rank'] = len(adj[n])

def update_link_attribs(wsn,u, v, plr, load):
    config.total_operations += 1
    if load is not -1:
        # update link load
        wsn[u][v]['load'] += load
    link_weight = LinkCost(wsn[u][v]['plr'], wsn[u][v]['load'])
    # update link weight
    wsn[u][v]['weight'] = link_weight.get_weight(link_weight)

def on_line_vn_request():
    source = input(" source node: ")
    if source != "":
        sink = input(" sink node: ")
    else:
        return
    if sink != "":
        quota = input(" quota: ")
    else:
        return
    if quota != "":
        VWSN_nodes = (int(source), {'load': int(quota)},
                      int(sink), {'load': int(quota)})
        link_reqiurement = {'load': int(quota), 'plr': 40}
        vnr = (1000, VWSN_nodes, link_reqiurement)
        config.online_flag = True
        embed(vnr,0,True)
    else:
        return

def map_links_cost(e_list, e_set, link_requirement,wsn):
    #print "e_list",e_list
    #print "e_set",e_set
    link_embedding_cost = 0
    current_weight = 0
    for u,v in e_set:
        config.total_operations += 1
        required = (e_list.count((u,v)) * link_requirement['load'])
        current_weight = wsn[u][v]['weight']
        update_link_attribs(wsn, int(u), int(v), link_requirement['plr'], required)
        link_embedding_cost += current_weight #weighted cost
        #link_embedding_cost +=(required * current_weight) #weighted cost
        #link_embedding_cost += (required  * wsn[u][v]['plr'])
    return link_embedding_cost

def map_nodes_cost(all_path_nodes, required_load,wsn):
    node_embedding_cost = 0
    for idx, pn in enumerate(all_path_nodes):
        config.total_operations += 1
        update_node_attribs(wsn, pn, required_load)
        node_embedding_cost += (required_load)
    return node_embedding_cost

def check_link_reliability_constraints(shortest_path, required_reliability,wsn):
    print 'required_reliability',required_reliability/100.0
    link_list = []
    #  number of re-transmissions
    K = 2
    #if shortest_path is not None:
    if len(shortest_path) > 0:
        for u,v in shortest_path:
            plr_ = 1 - wsn[u][v]['plr']/100.0
            plr = 1 - (1 - plr_)**K
            link_list.append(plr)
        path_reliability = reduce(lambda x, y: x * y, link_list)
        print 'path_reliability',path_reliability
        if path_reliability <= required_reliability/100.0:

	    print "RELIABILITY FAILED!"
            return False
        else:
            print path_reliability, "RELIABILITY OK", shortest_path
            return True
    else:
        #pass
        print "shortest path length is 0!!",shortest_path

def check_link_constraints(shortest_path,e_list, e_list2, load, required_plr, wsn):
    VN_links = nx.DiGraph()
    if not check_link_reliability_constraints(shortest_path, required_plr, wsn):  # need to fix 1 hop paths
        worst_link = (0,0)
        worst_plr = 0.0
        #print(shortest_path)
        for u, v in shortest_path:
            if wsn.edges[u,v]['plr'] >= worst_plr:
                #print(u,v,wsn.edge[u][v]['plr'],worst_plr)
                worst_plr = wsn.edges[u,v]['plr']
                worst_link = (u, v)
        print ("worst_link",worst_link)
        return worst_link,VN_links
    for u,v in e_list2:
        config.verify_operations += 1
        required_load = load * e_list.count((u,v))
        if wsn.edges[u,v]['load'] + required_load > 100:
            print("Link",u, v,"requires",wsn.edges[u,v]['load']," + ",required_load, "but have not got enough")
            return (u,v),VN_links
        else:
            VN_links.add_edge(u,v, **{'load':required_load})
            return_value = (0,0)
    return return_value,VN_links

def check_node_constraints(nodes_in_path, required_load, wsn):
    VN_nodes = nx.DiGraph()
    for idx,n in enumerate(nodes_in_path):
        config.verify_operations += 1
#        VN_nodes.add_node(n, load=required_load )
        # The old format used in version 1.9
        VN_nodes.add_node(n, **{'load': required_load})
        if wsn.node[n]['load'] + required_load > 100:
            if idx == 0:
                    #print("Source node",n," has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
            elif idx == (len(nodes_in_path) - 1):
                    #print("Sink node",n,"has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
            else:
                    #print("Relay node",n,"has - ",wsn.node[n]['load'],"but require",+ required_load )
                    return n, VN_nodes
    return 0, VN_nodes

def get_shortest_path(graph, frm, to, load, required_plr):
    print "SP algorithm is ", config.sp_alg_str
    #IMPORTANT!!
    #The below file of the networkx library must be modified to return 2 parameters (path, length) instead of 1 (path)
    # nx.__file__
    # eg. <HOME_OF_PYTHON>/site-packages/networkx/algorithms/shortest_paths/weighted.py
     
    if config.sp_alg_str == "Dijkstra":
        length, path = nx.bidirectional_dijkstra(graph, source=frm, target=to,  weight='weight')
    else:
        length, path = nx.astar_path_length(graph, source=frm, target=to, heuristic=None, weight='weight')
    config.verify_operations += 1
    if (path is None) or (length >= 10000000):
        return None, None
    s_path = []
    #print('Shortest path is ', end="")
    for idx, p in enumerate(path):
        config.verify_operations += 1
        if idx != len(path) - 1:
            s_path.append((path[idx], path[idx + 1]))
    #print("Shortest path links  ", s_path,"\n")
    return s_path, path

def get_max_edge_load(wsn,node, is_source):
    max_load = 0

    for n in adjacencies[node]: #config.reduced_adj[node]:
        config.verify_operations += 1
        #print "wsn[node][n]['load']", node,n
        link_load = wsn[node][n]['load']
        link_load2 = wsn[n][node]['load']
        max_load = max([link_load,link_load2])
        #if link_load > max_load:
            #max_load = link_load
        #if not is_source:
            #link_load = wsn[n][node]['load']
            #if link_load > max_load:
                #max_load = link_load
    return max_load

def check_frm_to_links(wsn,node,link_requirement):
    for n in config.reduced_adj[node]:
        config.verify_operations += 1
        if (100 - wsn[node][n]['load'] < link_requirement['load']) or (100 - wsn[n][node]['load'] < link_requirement['load']):
            return False
    return True

def remove_vn(vn):
    for u,v in vn[1].edges():
        update_link_attribs(config.committed_wsn, int(u), int(v), -1, -(vn[1][u][v]['load']))
    for n in vn[0].nodes():
        update_node_attribs(config.committed_wsn,n,-(vn[0].node[n]['load']))
    return True

# currently not used
def get_k_shortest_paths(wsn,source,sink,k,weight=None):
    k_paths = nx.shortest_simple_paths(wsn, source=source, target=sink, weight=weight)
    k_paths = islice(k_paths, k)
    for p in k_paths:
        print(p)
    return k_paths

def get_min_hops(sink):
    config.reduced_adj = list(adjacencies)
    min_hops = {}
    #print(len(nx.shortest_path(config.wsn,source=(56),target=sink))-1)
    for n in config.wsn.nodes():
        if n !=0:
            min_h = nx.shortest_path(config.wsn,source=(n),target=sink)
            if min_h is None:
                print "NO PATH EXISTS!!!!!!!!!!!!!!!!!!!!!"
                min_hops.update({(n, sink): (1000)})
            else:
                #min_h = get_shortest_path(config.wsn,n,sink)
                min_hops.update({(n,sink):(len(min_h)-1)})
                #print(n,"to",sink,"is", (len(min_h)-1),"hops via",min_h)
                #get_k_shortest_paths(config.wsn, n, sink, 5, weight=None)
    return min_hops

def commit(VN_nodes, VN_links, node_requirement, link_requirement,e_list, e_list2, path_nodes, shortest_path,wsn):
    #print "\nCOMMIT",path_nodes[0],"\n"
    config.total_operations += 1
    n_cost = map_nodes_cost(VN_nodes.nodes(), node_requirement,wsn)
    l_cost = map_links_cost(e_list, e_list2, link_requirement,wsn)
    #print "LINK REQUIREMENT",link_requirement
#    cost = (n_cost+l_cost) #node costs are not used by MILP either
    cost = l_cost
    current_vn = {'vnr': config.current_vnr, 'src': int(path_nodes[0]), 'load': link_requirement['load'], 'hop': link_requirement['hop'], 'nodes': nx.DiGraph(VN_nodes), 'links': nx.DiGraph(VN_links), 'shortest_path': list(shortest_path), 'path_nodes': list(path_nodes), 'cost': float(cost)}
    config.VWSNs.append(current_vn)
    if config.online_flag:
        vis.display_edge_attr(config.committed_wsn)
        vis.display_node_attr(config.committed_wsn)
        vis.display_vn_node_allocation(VN_nodes)
        vis.display_vn_edge_allocation(VN_links)
        vis.plotit(VN_links, shortest_path, path_nodes, 0)
        config.active_vns.append(current_vn)
    config.current_perm_emb_costs.update({path_nodes[0]: cost})

# Link Mapping begins here by seeking a feasible shortest path 
def verify_feasibility(link_requirement, frm, to, node_requirement,required_plr):
    #print("verify")
    config.verify_operations += 1
    hops = min_hops_dict.get((frm,to))
    max_load = 0
    wsn = nx.DiGraph()
    if config.online_flag:
        wsn = config.committed_wsn
    else:
        wsn = config.wsn_for_this_perm
    node_check, VN_nodes = check_node_constraints([frm, to], node_requirement, wsn)
    max_load = max([get_max_edge_load(wsn, frm, True),get_max_edge_load(wsn, to, False)])
    if not check_frm_to_links(wsn,to, link_requirement):
        #print("Sink node ", to, "does not have enough link resource\nEMBEDDING FAILED!")
        return False
    if not check_frm_to_links(wsn,frm, link_requirement):
        #print("Source node ", frm, "does not have enough link resource\nEMBEDDING FAILED!")
        return False
    # This may need to be disabled as Victor is not using it
    if node_check != 0:
        #print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return False
    if hops > 2:
        if (link_requirement['load'] * 2) > (100 - max_load):
            #print("Failed!",frm,to," cannot support",link_requirement['load'], "load request\nEMBEDDING FAILED!")
            return False
    # Find least weight path
    shortest_path, path_nodes = get_shortest_path(config.current_wsn, frm, to, link_requirement, required_plr)
    if shortest_path is None:
        #print(shortest_path,"No feasible path!-EMBEDDING HAS FAILED!")
        return False
    if len(shortest_path) == 0:
        print "shortest path is 0", shortest_path
    e_list, e_set = get_conflicting_links(path_nodes)
    ##get list of unique nodes from conflicting link list
    effected_nodes = []
    for u, v in e_set:
        config.verify_operations += 1
        if u not in effected_nodes:
            effected_nodes.append(u)
        if v not in effected_nodes:
            effected_nodes.append(v)
    node_check, VN_nodes = check_node_constraints(effected_nodes, node_requirement, wsn)
    #This may need to be disabled as MILP is not using it
    if node_check != 0:
       #print("node ", node_check, "does not have enough resource\nEMBEDDING FAILED!")
        return False
    else:
        link_check, VN_links = check_link_constraints(shortest_path,e_list, e_set, link_requirement['load'], link_requirement['plr'],wsn)
    if link_check == (0,0):
        #print("++SUCCESSFUL EMBEDDING++")
        config.has_embedding = True
        commit(VN_nodes, VN_links, node_requirement, link_requirement, e_list, e_set, path_nodes, shortest_path, wsn)
        config.feasible = True
        return False
    else:
        if link_check not in config.avoid:
            config.avoid.append(link_check)
        if recalculate_path_weights(frm, to, path_nodes, shortest_path):
            #print("recalculate_path_weights returned TRUE!!!\nEMBEDDING FAILED! ")
            return False
        #else:
            #print("recalculate_path_weights returned OK")
    check_again(link_requirement, frm, to, node_requirement,required_plr)
        #verify_feasibility(link_reqiurement, frm, to, node_requirement)

# Used in order to avoid depth issues due to the recursive calls  
def check_again(link_reqiurement, frm, to, node_requirement,required_plr):
    config.verify_operations += 1
    if config.feasible == False:
        is_failed = verify_feasibility(link_reqiurement, frm, to, node_requirement,required_plr)
        #if is_failed:
            #print("Verify failed!!! ")
            #pass

def get_conflicting_links(path_nodes):
    #print "path_nodes",path_nodes
    config.verify_operations += 1
    tx_nodes = list(path_nodes)
    tx_nodes.pop()
    effected_edges = []
    for i, tx in enumerate(tx_nodes):
        #print i,"rx",rx
        config.verify_operations += 1
        #if i != 0:
        effected_edges.extend(conflicting_links_dict[tx][path_nodes[i+1]])
        #effected_edges.extend(conflicting_links_dict[path_nodes[tx]][i+1])
    effected_edges_set = list(set(effected_edges))
    return effected_edges, effected_edges_set

# Has no effect
def remove_insufficient_links(required_link_quality,required_link_load):
    for item in wsn_substrate.get_link_quality():
        if item[1] >= 100.0 - required_link_quality:
            #print (item[0][0],item[0][1],"-",100.0-item[1])
            #print (item[0][0], "-", item[0][1])
            #config.reduced_adj[item[0][0]].remove(item[0][1])
            #config.current_wsn_removed_edges.remove_edge(item[0][0],item[0][1])
            config.current_wsn[item[0][0]][item[0][1]]['weight'] = 10000000
    for link in config.current_wsn.edges(data=True):
        #counter1 += 1
        if 100-link[2]['load'] < required_link_load:
            link[2]['weight']  = 10000000

def evaluate_perms(current_perm):
   #print ("current_perm",current_perm)
    keys = []
    for k in current_perm:
        ##print(k)
        config.total_operations += 1
        keys.append(k)
    source_nodes = []
    overall_cost = current_perm[keys[0]]['overall_cost']
    for k, v in current_perm[keys[0]]['embeddings'].items():
        config.total_operations += 1
        source_nodes.append(k)
    if len(config.best_embeddings) != 0:
        config.total_operations += 1
        if config.max_accepted_vnrs < len(source_nodes):
            config.max_accepted_vnrs = len(source_nodes)
            current_key = list(config.best_embeddings.keys())
            config.best_embeddings.pop(current_key[0], 0)
            config.best_embeddings.update(
                {str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
            del config.committed_wsn
            config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
            del config.active_vns
            config.active_vns = list(config.VWSNs)
        elif config.max_accepted_vnrs == len(source_nodes):
            current_key = list(config.best_embeddings.keys())
            best_cost = config.best_embeddings[current_key[0]]['overall_cost']
            if best_cost > overall_cost:
                config.best_embeddings.pop(current_key[0], 0)
                config.best_embeddings.update(
                    {str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
                del config.committed_wsn
                config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
                del config.active_vns
                config.active_vns = list(config.VWSNs)
    else:
        config.total_operations += 1
        config.best_embeddings.update({str(source_nodes): {'overall_cost': overall_cost, 'permutation': keys[0]}})
        config.max_accepted_vnrs = len(source_nodes)
        del config.committed_wsn
        config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
        del config.active_vns
        config.active_vns = list(config.VWSNs)
        config.acceptance = config.max_accepted_vnrs

# evaluate the quality of the embeddings based on the objective function [min(Cost(max(|VNRs|)))]
def evaluate_perms_(current_perm):
   #print ("current_perm", current_perm)
    keys = []
    for k in current_perm:
        #print("keys-",k)
        config.total_operations += 1
        keys.append(k)
    source_nodes = []
    overall_cost = float(current_perm[keys[0]]['overall_cost'])
    for k, v in current_perm[keys[0]]['embeddings'].items():
        config.total_operations += 1
        #print("source_nodes-",k,"-v-",v)
        source_nodes.append(k)
    if len(source_nodes) > config.max_accepted_vnrs:
        config.best_embeddings.update({str(source_nodes): {'overall_cost': float(overall_cost), 'permutation': keys[0]}})
        current_key = list(config.best_embeddings.keys())
        config.max_accepted_vnrs = len(source_nodes)
        del config.committed_wsn
        config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
        del config.active_vns
        config.active_vns = list(config.VWSNs)
    elif len(source_nodes) == config.max_accepted_vnrs and len(source_nodes) != 0 :
        current_key = list(config.best_embeddings.keys())
        best_cost = float(config.best_embeddings[current_key[0]]['overall_cost'])
        if best_cost > overall_cost:
            config.best_embeddings.pop(current_key[0], 0)
            config.best_embeddings.update({str(source_nodes): {'overall_cost': float(overall_cost), 'permutation': keys[0]}})
            del config.committed_wsn
            config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
            del config.active_vns
            config.active_vns = list(config.VWSNs)

# memoize and use already calculated sequences to eliminate duplicate work
def memoize_perms():
    #show_dataStructs("\nMEMOIZE_PERMS")
    if len(config.prefix_length) < len(config.current_key_prefix):
        config.prefix_length.append(list(config.current_key_prefix))
        config.already_mapped_vnrs.update({str(config.current_key_prefix): {
            'graph': nx.DiGraph(config.wsn_for_this_perm), 'embeddings': dict(config.current_perm_emb_costs),
            'overall_cost': float(config.overall_cost), 'best_embeddings': dict(config.best_embeddings), 'vwsns': list(config.VWSNs), 'success':dict(config.vns_per_perm)}})

    elif config.prefix_length[len(config.current_key_prefix) - 1] != config.current_key_prefix:
        config.already_mapped_vnrs.pop(str(config.prefix_length[len(config.current_key_prefix) - 1]))
        config.prefix_length[len(config.current_key_prefix) - 1] = config.current_key_prefix
        config.already_mapped_vnrs.update({str(config.current_key_prefix): {
            'graph': nx.DiGraph(config.wsn_for_this_perm), 'embeddings': dict(config.current_perm_emb_costs),
            'overall_cost': float(config.overall_cost), 'best_embeddings': dict(config.best_embeddings), 'vwsns': list(config.VWSNs), 'success':dict(config.vns_per_perm)}})

#uses suffix _ for output file
def recalculate_path_weights(frm,to,path_n,shortest_path):
    for (u, v) in config.avoid:
        config.link_penalize_operations += 1
        #print("recalculate",u,v)
        path_nodes = list(path_n)
        path_n.reverse()
        config.avoid.remove((u, v))
        config.current_wsn[u][v]['weight'] = 10000000 #penalize link

        if len(path_nodes) == 2:
            #print("Source node", frm, "does not have enough resource in a single-hop path.\nEMBEDDING HAS FAILED!")
            #for n in config.reduced_adj[frm]:
                #config.link_penalize_operations += 1
                #config.current_wsn[frm][n]['weight'] = 10000000  # make path cost unfeasible
            return True
        elif (v, u) in shortest_path:
            #print(v, u, "v-u link in path does not have enough resource!")
            #print(config.current_wsn[v][u]['weight'])
            config.current_wsn[v][u]['weight'] = 10000000
            return False
        elif (u, v) in shortest_path:
            #print(u, v, "u-v link in path does not have enough resource!")
            config.current_wsn[u][v]['weight'] = 10000000
            #config.current_wsn[shortest_path[shortest_path.index(u)-1]][u]['weight'] = 10000000  #about this I'm not sure!!
            return False
        elif u == frm:
            #if (len(path_nodes) <= 3) or (v != path_nodes[1]):
            if (len(path_nodes) <= 3):
                #print("Source node u+", u, "does not have enough resource in 2-hops path.\nEMBEDDING HAS FAILED!")
                #for n in config.reduced_adj[frm]:
                    #config.link_penalize_operations += 1
                    #print(config.reduced_adj[frm])
                    #print(frm,n)
                    #config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
                return True
            #this is probably redundant
            elif v == path_nodes[1]:
                for n in path_n:
                    config.link_penalize_operations += 1
                    if n in config.reduced_adj[frm]:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                        #print("Source node u-", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
            else:
                #print("Source node u", u, "does not have enough resource. This case has not been properlyy handled yet!\nEMBEDDING HAS FAILED!")
                config.current_wsn[path_n[1]][path_n[0]]['weight'] = 10000000
                #user_input = input('?: ')
                #if user_input is 0:
                    #return False
            return False
        elif v == frm:
            #if (len(path_nodes) <= 3) or (u != path_nodes[1]):
            if len(path_nodes) <= 3:
                #print("Source node v+", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                #for n in config.reduced_adj[frm]:
                    #config.link_penalize_operations += 1
                    #config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
                return True
            # both cases below are probably redundant here
            elif u == path_nodes[1]:
                for n in path_n:
                    config.link_penalize_operations += 1
                    if n in config.reduced_adj[frm]:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                        #print("Source node v-", v, "does not have enough resource.\nEMBEDDING HAS FAILED!")
                        return False
            else:
                #print("Source node v", v, "does not have enough resource. This case has not been handled yet!\nEMBEDDING HAS FAILED!")
                config.current_wsn[path_n[1]][path_n[0]]['weight'] = 10000000
                #user_input = input('??: ')
                #if user_input is '':
                    #return False
            return False
        elif u == to:
            #print("Sink node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
            #for n in config.reduced_adj[frm]:
                #config.link_penalize_operations += 1
                #config.current_wsn[frm][n]['weight'] = 10000000  # make path cost unfeasible
            return True
        elif v == to:
            #print(u,v)
            #print("path_n", path_n)
            #print("path_n[1]",path_n[1])
            #print("path_n[2]", path_n[2])
            #print("config.reduced_adj[u-1]",config.reduced_adj[u-1])
            #print("config.reduced_adj[u]", config.reduced_adj[u])
            #print("config.reduced_adj[v]", config.reduced_adj[v])
            if (u != path_n[1]) and (path_n[2] not in config.reduced_adj[u]):
                config.current_wsn[path_n[1]][path_n[0]]['weight'] = 10000000
                #print("Sink node u", u, "does not have enough resource.\nEMBEDDING HAS FAILED!")
 #               for n in config.reduced_adj[to-1]:
                #for n in config.reduced_adj[frm]:
                    #config.link_penalize_operations += 1
                    #config.current_wsn[frm][n]['weight'] = 10000000 #make path cost unfeasible
                #return True
            elif (u != path_n[1]) and (path_n[2] in config.reduced_adj[u]):
                #print(u, v, "u-v link is in right angle to path!! Penalize link [path_n[2]][path_n[1]]!")

                config.current_wsn[path_n[2]][path_n[1]]['weight'] = 10000000  # make path cost unfeasible
            elif u == path_n[1]:
              #print(u, v, "u-v link is in path! Penalize link (predecessor of u)->u!")
                config.current_wsn[path_n[2]][u]['weight'] = 10000000 #make path cost unfeasible
            elif path_n[2] in config.reduced_adj[u]:
                #This may be redundant here!!
                #print(u, v, "u-v link is in right angle to path!! Penalize link [path_n[2]][path_n[1]]!")
                config.current_wsn[path_n[2]][path_n[1]]['weight'] = 10000000 #make path cost unfeasible
                if len(path_n) > 3:
                    config.current_wsn[path_n[3]][path_n[2]]['weight'] = 10000000 #make path cost unfeasible##############################
            else:
                #print(v," is source. This case has not been handled yet!")
                user_input = input('???: ')
                if user_input is '':
                    return False
            return False

        ##double check this section!
        ##changed from if to elif
        elif u in path_nodes:
            #print(u, "u in path does not have enough resource!")
            config.current_wsn[path_nodes[path_nodes.index(u) - 1]][u]['weight'] = 10000000
            return False
        elif v in path_nodes:
            #print(v, "v in path does not have enough resource!")
            config.current_wsn[path_nodes[path_nodes.index(v) - 1]][v]['weight'] = 10000000
            return False
        else:
            for n in path_n:
                config.link_penalize_operations += 1
                nbr = config.reduced_adj[n]
                if u in nbr:
                    #print(u, "u is a neighbor of",n,"and does not have enough resource!")
                    config.current_wsn[n][u]['weight'] = 10000000
                    if n != frm:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                    else:
                        config.current_wsn[frm][path_nodes[1]]['weight'] = 10000000
                    return False
                elif v in nbr:
                    #print(v, "v is a neighbor of",n," and does not have enough resource!!!")
                    config.current_wsn[n][v]['weight'] = 10000000
                    if n != frm:
                        config.current_wsn[path_n[path_n.index(n) + 1]][n]['weight'] = 10000000
                    else:
                        config.current_wsn[frm][path_nodes[1]]['weight'] = 10000000
                    return False
                    #show_penalized_links()
    return True

def reinitialize():
    # VNR specific fields (used in embed())
    config.current_wsn = {}  # copy of wsn_for_this_perm represents the substrate
    config.reduced_adj = dict()  # copy of original adjacency list of substrate
    config.avoid = []  # holds the links that need to be penalised
    config.has_embedding = False  # indicates if there are existing appings (used for ONLINE)
    config.feasible = False  # controls the termination of re-tries (used for verify_feasibility)
    config.online_flag = False  # controls embedding mode

    # permutation specific fields
    config.wsn_for_this_perm = {}  # copy of wsn represents the substrate
    config.VWSNs = []  # feasible embeddings for each/current permutation
    config.current_perm_emb_costs = {}  # embeddings and their costs for current permutation
    config.overall_cost = 0  # total cost for all embeddings in each/current permutation
    config.committed_wsn = {}  # copy of wsn_for_this_perm after identifying final/best embedding for current perm
    config.max_accepted_vnrs = 0  # highest number of vnrs in current perm
    config.min_feasible_vnrs = 0  # lower bound

    config.already_mapped_vnrs = {}  # memoizes results of previous embeddings (used in independent perm blocks)
    config.best_embeddings = {}  # best embeddings for each combination (ultimately the optimal solution) [('[8, 4, 45]', {'overall_cost': 56371.374759009785, 'permutation': 0})]

    # test case specific fields
    config.wsn = {}  # the original instance of the initialized substrate graph
    config.active_vns = []  # copy of VWSNs, stores the list of embedded vns (used by generate_output to update mapping_dictionary)

    #config.sp_alg_str = "Dijkstra"  # identifies path finder algorithm used for test
    config.main_sink = 0  # identifies sink node id used for test
    config.X = 0  # used to set graph dimensions manually
    config.Y = 0  # used to set graph dimensions manually

    # Output, first three copied from input vector (used in pickle file)
    config.nwksize = 0
    config.numvn = 0
    config.numvn_to_permute = 0
    config.iteration = 0
    # Following result from algorithm execution
    config.proc_time = 0.0
    # acceptance = config.max_accepted_vnr / config.numvn
    config.mapping = dict()  #: dictionary vlink:[slinks],
    config.overall_cost = 0.0
    config.acceptance = 0
#    config.result_vectors = []  # stores test results until written into pickle file

    # performance metrics /fields
    config.start = 0.0
    config.perm_counter = 0
    config.total_operations = 0
    config.dijkstra_operations = 0
    config.link_penalize_operations = 0
    config.verify_operations = 0
    config.plot_counter = 0

    config.current_perm = []
    config.previous_perm = []
    config.perm_prefix = []
    config.vns_per_perm = {}  # success/fail of each vnrs per perms
    config.perms_list = {}  # store success/fail of vnrs for all perms

    config.prefix_length = []  # used to identify already calculated sequences to eliminate duplicate work
    config.current_key_prefix = []

    config.current_perm_block_acceptance = 0  # max accepted vnrs for current perm block
    config.current_perm_block_cost = 0.0  # max embedding cost for current perm block
    config.current_perm_block_results = {}
    config.current_perm_results = {}
    config.current_perm_block_best = {'first_find': 0,'permutation': 0, 'acceptance':0.0,'overall_cost':0.0,'committed':None,'mappings':None}

    config.current_test_best = {'first_find': 0,'permutation': 0, 'acceptance':0.0,'overall_cost':0.0,'committed':None,'mappings':None}
    config.first_best = {}

    config.current_vnr = None
    config.current_vnlist = None
    config.vnlist = None

    config.isbestcombo = True
    config.first_find_flag = False
    config.second_find_flag = False
    config.first_find = 0
    config.solution_progress = []

    # node mapping specific parameters
    config.source_list = []
    config.src_loads = []
    config.subset_sums = []

def calculate_cost():
    config.overall_cost = 0.0
    for u, v in config.wsn_for_this_perm.edges():
        config.overall_cost += config.wsn_for_this_perm[u][v]['weight']

def map_nodes(vnrs_list):
    #  convert vnrs to proper format (OBSOLETE)
    #myList = vne.get_vnrs(vnrs_list)
    myList = vnrs_list
    #  sort vnrs in non decreasing order by load
    myList.sort(key=lambda x: x[3]['load'], reverse=True)
    allocated_nodes = []
    for indx, vnr in enumerate(list(myList)):
        print vnr[3]['load']," - ",vnr
        #  get requested interest point
        ip = vnr[1][0]
        #  get id of closest node
        closest_node, closest_node_hops = get_closest_node(ip)
        if closest_node_hops == 1000:
            print "Non feasible request!",vnr
            myList.remove(vnr)
            continue
        print "closest_node",closest_node,closest_node_hops
        #  get list of neighbors for closest node
        neighbors = []
        try:
            neighbors = list(adjacencies[closest_node])
            neighbors.append(closest_node)
	    # 2nd hop neighbors
            nneighbors = []
            for neigh in neighbors:
                nneighbors.extend(adjacencies[neigh])
            nneighbors = list(set(nneighbors).difference(set(neighbors)))
            #print "neighbors", neighbors
            #print "nneighbors",nneighbors
            #  get list of feasible candidate neighbors that satisfy err_rate requirement
            candidates = [(min_hops_dict[(n,config.main_sink)],n) for n in neighbors if n != 0 and config.err_matrix[vnr[1][0]][n] < vnr[1][1] ]
            #sorted_candidates = [i[1] for i in sorted(candidates, key=lambda x: x[0])]
            #print "candidates", candidates
            #  if no neighbor satisfy err_rate requirement then check nodes in err_matrix
            if len(candidates) == 0:
                #non_neighbor_candidates = [(min_hops_dict[(idx,config.main_sink)],idx) for idx, err in enumerate(config.err_matrix[vnr[1][0]]) if (err < vnr[1][1]) and (idx != 0) and (closest_node_hops+1 >= min_hops_dict[(idx,config.main_sink)] >= closest_node_hops-1)]
                non_neighbor_candidates = [(min_hops_dict[(n,config.main_sink)],n) for n in nneighbors if n != 0 and config.err_matrix[vnr[1][0]][n] < vnr[1][1] ]
                #print "non_neighbor_candidates",non_neighbor_candidates
                #  if no feasible node in err_matrix either then remove current vnr from list
                if len(non_neighbor_candidates) == 0:
                    print "Non feasible request!", vnr
                    myList.remove(vnr)
                    continue
                else:
                    candidates = non_neighbor_candidates
                    #print "max_hop",max_hops
                    #print "non_neighbor_candidates", candidates
            candidates_load = []
            #  calculate total required load based on hop distance to
            #if len(candidates) != 0:
            for h,n in candidates:
                #print h,n
                sink_load = 0
                l = vnr[3]['load']
                if h == 1:
                    #print "1-",vnr[3]['load']
                    sink_load = l
                elif h >= 2:
                    #print "2-",vnr[3]['load']
                    sink_load = 2 * l
                candidates_load.append((sink_load,h,n,l))
            candidates_load.sort(key=lambda x: x[1])
            print "---candidates_load",candidates_load
            closest_cand_hop = candidates_load[0][1]

            if len(allocated_nodes) == 0:
                node_to_allocate = candidates_load[0][2]
                allocated_nodes.append(node_to_allocate)
            else:
                node_t_a = candidates_load[0][2]
                if node_t_a not in allocated_nodes and is_interfering(node_t_a, allocated_nodes) == False:
                    node_to_allocate = node_t_a
                    allocated_nodes.append(node_to_allocate)
                else:
                    has_alternative = True
                    for alt_node in candidates_load:
                        if alt_node[2] not in allocated_nodes and is_interfering(alt_node[2], allocated_nodes) == False:
                            node_to_allocate = alt_node[2]
                            allocated_nodes.append(node_to_allocate)
                            has_alternative = True
                            break
                        else:
                            has_alternative = False
                    if not has_alternative:
                        node_to_allocate = node_t_a
                        allocated_nodes.append(node_to_allocate)

            vnr[3].update({'candidates': candidates_load, 'src': node_to_allocate, 'sink_load': candidates_load[0][0],'hop': closest_cand_hop})
            print "src",vnr[3]['src']
        #print "vnr",vnr[3]['candidates']
        except:
            print "Non feasible request!", vnr
            # remove vnr if cannot meet with sensing error constraint
            myList.remove(vnr)
            continue
    #myList.sort(key=lambda x: x[3]['candidates'][0][3], reverse=True)

###########Check if there are duplicates####################
    is_unique = False
    duplicates = {}
    src_nodes = []
    for v in myList:
        #print v[3]['src']
        #print "vnr", v[3]['candidates']
        snode = v[3]['src']
        #if len(v[3]['candidates']) > 1:
        try:
            duplicates[snode] += 1
        except:
            duplicates[snode] = 1
        if snode not in src_nodes:
            src_nodes.append(snode)

    duplicates = {k:v for k,v in duplicates.iteritems() if v > 1}
#    duplicates,myList = remove_duplicates(duplicates,myList)
    print "duplicates--",duplicates
    myList.sort(key=lambda x: x[3]['load'])#, reverse=True)
    if len(duplicates) > 0:
        remove_duplicates_(duplicates, myList, src_nodes)
      
    # identify load required at the sink
    #print myList
    sink_loads = []
    sink_loads = [(v[3]['src'],v[3]['sink_load']) for v in myList]
    config.subset_sums = []
    subset_sum(sink_loads, 100)
    config.src_loads = [(v[3]['src'],v[3]['sink_load'],v[3]['load'],v[3]['hop']) for v in myList]#list(sink_loads)
    #print "sink_loads",sink_loads
    config.numvn_to_permute = len(myList)
    return myList

def is_interfering(node_t_a,allocated_nodes):
    for allocated_n in allocated_nodes:
        if allocated_n == node_t_a:
            return True
        for n in adjacencies[allocated_n]:
            if n == node_t_a:
                #print "INTERFERING"
                return True
    #print "NOT INTERFERING"
    return False

def remove_duplicates_(duplicates, r_list, src_nodes):
    for j in range(0, len(duplicates)):
        for r in r_list:
            if len(duplicates) == 0:
                print "EMPTY Duplicates",duplicates
                break
            s = r[3]['src']
            if s in duplicates.keys():# == k:
                if duplicates[s] <= 1:
                    del duplicates[s]
                    continue
                #  if there are other candidates
                candidates = r[3]['candidates']
                if len(candidates) > 1:
                    for c in candidates:
                        if c[2] != s and c[2] not in src_nodes and c[2] not in duplicates.keys() and is_interfering(c[2], src_nodes) == False:
                            # new source
                            r[3]['src'] = c[2]
                            duplicates[c[2]] = 1
                            duplicates[s] -= 1
                            src_nodes.append(c[2])
                            break
                        #else:
                            #print "CANDIDATE'S ALREADY USED"
                else:
                    print "NO ALTERNATIVE CANDIDATE"

def remove_duplicates(duplicates, myList):
    if len(duplicates) == 0:
        print "NO MORE DUPLICATES"
        return duplicates,myList
    for dup in duplicates.keys():
        dup_vnrs = [dv for dv in list(myList) if dv[3]['src'] is dup and len(dv[3]['candidates'][dv[3]['hop'] - 1]) > 1]
        dup_vnrs.sort(key=lambda x: len(x[3]['candidates'][x[3]['hop'] - 1]), reverse=True)
        occurance = {}
        for r in dup_vnrs:
            for v in r[3]['candidates'][r[3]['hop']-1]:
                print "v[2]",v[2]
                try:
                    occurance[v[2]] += 1
                except:
                    occurance[v[2]] = 1
        import operator
        sorted_occurance = sorted(occurance.items(), key=operator.itemgetter(1))
        for k,v in occurance.items():
            if v == 1:
                for r in myList:
                    if r[3]['src'] == k:
                        for s in r[3]['candidates'][r[3]['hop'] - 1]:
                            if k == s[2]:
                                r[3]['src'] = s[2]
                                print "new src is",r[3]['src']
                                break


def subset_sum(numbers, target, partial=[]):
    s = sum([l for (n,l) in partial])
    if s <= target:
    #    print len(partial),"sum(%s)=%s of %s" % (partial,s, target)
        try:
            config.subset_sums.append((len(partial),s,partial))
        except:
            print "ERROR APPENDING PARTIAL SUM"
    if s >= target:
        return 
    for i in range(len(numbers)):
        n = numbers[i]
        remaining = numbers[i+1:]
        subset_sum(remaining, target, partial + [n])

#  Takes in an iterestpoint as arg, retrieves the coordinate of the interest point
#  Finds the coordinate of the closest node and returns its id
def get_closest_node(ip):
    ip_x = interest_coordinates[ip][0]
    ip_y = interest_coordinates[ip][1]
    #print "coordinate of IP",ip," is ", ip_x, ip_y
    closest = spatial.KDTree(node_coordinates)
    closest_n = closest.query([(ip_x, ip_y)])[1][0]
    try:
        closest_n_hop = min_hops_dict[closest_n, config.main_sink]
    except:
        closest_n_hop = 1
        #print "closest node ID is:", closest_node
    return closest_n, closest_n_hop

def find_best_source(vnr,indx):
    #  requested interest point
    ip = vnr[1][0]
    #  id of closest node
    closest_node = get_closest_node(ip)
    #  list of neighbors for closest node
    neighbors = list(adjacencies[closest_node])
    neighbors.append(closest_node)
    print "neighbors of ",closest_node, neighbors
    #  get list of nodes and err_rate that satisfy err_rate constraint for the required interest point
    sensors = [(err,idx) for idx,err in enumerate(config.err_matrix[vnr[1][0]]) if (err < vnr[1][1]) and (idx != 0)]
    if len(sensors) == 0:
        return False
    #  sort sensors based on err_rate
    sorted_sensors = [i[1] for i in sorted(enumerate(sensors), key=lambda x: x[1])]
    #  get node id from above sorted tuple
    sorted_sensors_idx = [s[1] for i,s in enumerate(sorted_sensors)]
    #  get neighbors that satisfy err_rate constraint
    least_err_sensors = [n for n in sorted_sensors_idx if n in neighbors]

    #  add first item from feasible source nodes list as src to vnr
    if len(least_err_sensors) != 0:
        #if config.source_list
        source_node = least_err_sensors[0]
        #print "min_hops_dict",min_hops_dict
        vnr[3].update({'src':source_node, 'min_hops':min_hops_dict[(source_node,config.main_sink)]})
        #print "neigh",list(adjacencies[source_node])
        #print vnr
        return 0
    else:
        min_hop = 1000
        for n in sorted_sensors_idx:
            #  distance to closest_node
            min_h = len(nx.shortest_path(config.wsn, source=(n), target=closest_node))
            if min_h < min_hop:
                min_hop = min_h
                vnr[3].update({'src': n, 'min_hops':min_hops_dict[(n,config.main_sink)]})
                #print "neigh", list(adjacencies[n])
                #print vnr
            #print "hop between", closest_node, "and", n, "is", min_h
        return 0
   
def generate_independent_perm_blocks(vnrs_list, result_q, min_acceptance, min_feasible_q, min_accept, solution_progress_q):
    no_of_processes = multiprocessing.cpu_count()
    #print("vne.get_vnrs(vnrs_list)", vne.get_vnrs(vnrs_list))
    perms = tuple(itertools.permutations(vnrs_list, r=None))
    vnrs_size = len(vnrs_list)
    start_indx = 0
    if vnrs_size > 3:
        idx_increm = math.factorial(vnrs_size) / no_of_processes
    else:
        idx_increm = math.factorial(vnrs_size)
    for v in range(0,no_of_processes):# vnrs_size):
        #config.total_operations += 1
        end_indx = start_indx + idx_increm
        independent_perm_block = itertools.islice(perms, start_indx, end_indx)
        #independent_perm_block = itertools.islice(perms2, start_indx, end_indx)
        #print len(tuple(independent_perm_block))
        start_indx = end_indx
        jobs.append(multiprocessing.Process(target=process_independent_perm_block, args=(independent_perm_block,result_q, min_acceptance,min_feasible_q, min_accept, solution_progress_q)) )
    #jobs.append(multiprocessing.Process(target=process_independent_perm_block, args=(perms,result_q,min_acceptance)) )

def get_lower_bound(perms, sum_src):
    print "get_lower_bound"
    config.online_flag = False
#    config.current_perm_block_best = {'first_find': 0,'permutation': 0, 'acceptance': 0.0, 'overall_cost': 0.0, 'committed': None, 'mappings': None}
    config.current_perm_block_results = {}  # stores final results for current perm block
    config.current_perm_block_acceptance = 0  # max accepted vnrs for current perm block
    config.current_perm_block_cost = 0.0  # max embedding cost for current perm block
    config.best_embeddings = {}
    config.max_accepted_vnrs = 0
    config.active_vns = []  # ?
    config.current_perm_results = {}  # stores final results for current perm (goes into current_perm_block_results)
    del config.wsn_for_this_perm
    config.wsn_for_this_perm = nx.DiGraph(config.wsn)
    del config.VWSNs
    config.VWSNs = []
    del config.current_perm_emb_costs  # ?
    config.current_perm_emb_costs = dict()  # ?
    del config.overall_cost
    config.overall_cost = 0.0
    del config.vns_per_perm
    config.vns_per_perm = dict()
    config.feasible = False
    config.current_key_prefix = []

    for idx, vnr in enumerate(perms):
        current_success = True
        # concatenate current source to previous sequence
        #config.current_key_prefix = config.current_key_prefix + [(idx, vnr[1][0])]
        
        # optmize work effort by avoiding to process known unfeasible sequence of requests
        # check the state of the same request in the previous perm and if it has failed at a
        # higher index/position then skip it
        # (cases when same source node requested multiple times in a perm needs to be handled yet)
        # condition makes sense since there is no previous perm of perm 0 nor vnr 0
        if sum_src is None:
            embed(vnr, idx, current_success)
        elif (vnr[3]['src'],vnr[3]['sink_load'],) in sum_src[2]:
            embed(vnr, idx, current_success)
    lwr_bound = int(len(config.VWSNs))
    #print "lower_bound", lwr_bound #, config.current_key_prefix, float(config.overall_cost)
    return lwr_bound

def embed(vnr,vnr_idx,prev_success):
    print "embed", vnr
    config.current_vnr = vnr
    vwsn_nodes = vnr[2]
    link_reqiurement = vnr[3]
    key = 'src'
    if key in vnr[3]:
        frm = vnr[3][key]
    else:
        #  key is missing
        return False
    to = list(vnr)[2]
    node_requirement = vnr[1][1]#vwsn_nodes[1]['load']
    del config.avoid
    config.avoid = []
    if config.online_flag:
        print("ONLINE EMBEDDING")
        config.VWSNs = []
        config.current_perm_emb_costs = {}
        if config.has_embedding == False:
            config.committed_wsn = nx.DiGraph(config.wsn)
        config.current_wsn = nx.DiGraph(config.committed_wsn)
        config.reduced_adj = list(config.committed_wsn.adjacency_list())
    else:
        #check if current sequence has been memoized
        if str(config.current_key_prefix) in config.already_mapped_vnrs:
            config.wsn_for_this_perm = nx.DiGraph(config.already_mapped_vnrs[str(config.current_key_prefix)]['graph'])
            config.current_perm_emb_costs = dict(config.already_mapped_vnrs[str(config.current_key_prefix)]['embeddings'])    #.update({path_nodes[0]: cost})
            config.overall_cost = float(config.already_mapped_vnrs[str(config.current_key_prefix)]['overall_cost'])
            config.best_embeddings = dict(config.already_mapped_vnrs[str(config.current_key_prefix)]['best_embeddings'])
            config.VWSNs = list(config.already_mapped_vnrs[str(config.current_key_prefix)]['vwsns'])
            config.vns_per_perm = dict(config.already_mapped_vnrs[str(config.current_key_prefix)]['success'])
            #  it has been memoized so use cached results
            return False
        else:
            #show_dataStructs("ELSE, NOT IN MEMOIZED")
            del config.current_wsn
            config.current_wsn = nx.DiGraph(config.wsn_for_this_perm)
            del config.reduced_adj
            config.reduced_adj = copy.deepcopy(adjacencies)
            #remove_insufficient_links(vnr[3]['plr'],vnr[3]['load'])
            config.perm_counter += 1
            config.feasible = False
            required_plr = vnr[1][1]
            check_again(link_reqiurement, frm, to, node_requirement,required_plr)
            return True

def process_independent_perm_block(independent_perm_block,result_q, min_acceptance, min_feasible_q, min_accept, solution_progress_q):
    config.skipit = 0
    config.online_flag = False
    perms = independent_perm_block
    config.current_perm_block_best = {'first_find': 0,'permutation': 0, 'acceptance':0.0,'overall_cost':0.0,'committed':None,'mappings':None}
    config.current_perm_block_results = {}  # stores final results for current perm block
    config.current_perm_block_acceptance = 0  # max accepted vnrs for current perm block
    config.current_perm_block_cost = 0.0  # max embedding cost for current perm block
    config.best_embeddings = {}
    config.max_accepted_vnrs = 0
    config.active_vns = []  # ?
    config.p_id = multiprocessing.current_process()._identity[0]
    config.proc = int(config.p_id) - (config.iteration * 4) + (start_iter * 4)
    config.min_feasible_vnrs = l_bound.value #min_accept.value #min_acceptance  # sets lower bound
    #print "config.min_feasible_vnrs",config.min_feasible_vnrs,"min_accept",min_accept.value
    #print min_acceptance,"min_accept",min_accept.value
    #config.min_feasible_vnrs = min_feasible_q.get()  # sets lower bound
    #min_feasible_q.put(config.min_feasible_vnrs)
    for i, per in enumerate(perms):
        #print i, "th permutation", multiprocessing.current_process(), "min feasible is:", config.min_feasible_vnrs
        config.current_perm_results = {}  # stores final results for current perm (goes into current_perm_block_results)
        del config.wsn_for_this_perm
        config.wsn_for_this_perm = nx.DiGraph(config.wsn)
        del config.VWSNs
        config.VWSNs = []
        del config.current_perm_emb_costs  # ?
        config.current_perm_emb_costs = dict()  # ?
        del config.overall_cost
        config.overall_cost = 0.0
        del config.vns_per_perm
        config.vns_per_perm = dict()
        config.feasible = False
        config.current_key_prefix = []
        for idx, vnr in enumerate(per):
            current_success = True
            # concatenate current source to previous sequence
            current_vnr_id = (vnr[3]['src'],vnr[1][0],vnr[3]['load'])
            config.current_key_prefix = config.current_key_prefix + [current_vnr_id] # config.current_key_prefix + [(idx, vnr[1][0])]
            #print "config.current_key_prefix",config.current_key_prefix
            current_accepted = int(len(config.VWSNs))
            #min_feasible_vnrs = min_feasible_q.get()
            #min_feasible_q.put(min_feasible_vnrs)

            # check if it is still feasible to beat current lower bound/best
            #if False:
            #print config.numvn,idx,current_accepted,config.min_feasible_vnrs
            if (config.numvn - idx +  current_accepted) < l_bound.value: #config.min_feasible_vnrs:
                current_success = False # make the algorithm skip this request
            else:
                # reduce work effort by avoiding to process known unfeasible sequence of requests
                # check the state of the same request in the previous perm and if it has failed at a
                # higher index/position then skip it
                # (cases when same source node requested multiple times in a perm needs to be handled yet)
                # condition makes sense since there is no previous perm of perm 0 nor previous position of vnr 0
                if i > 0 and idx > 0:
                    # get the position of current vnr in previous perm
                    previous_position = list(config.perms_list[i - 1][current_vnr_id].keys())[0] #list(config.perms_list[i - 1][vnr[1][0]].keys())[0]
                    # get the result of current vnr in previous perm
                    success = config.perms_list[i - 1][current_vnr_id].get(previous_position) #config.perms_list[i - 1][vnr[1][0]].get(previous_position)
                    # check if current vnr was in a higher position in the previous perm and if it Failed
#  should also check this with condition if previous_position <= idx and success is False:
                    if previous_position < idx and success is False:
                        # set flag to False so it indicates that it must be skipped
                        current_success = False
            if embed(vnr, idx, current_success):
                #print multiprocessing.current_process()
                current_success = config.feasible
            elif i > 0:
                # get the position of current vnr in previous perm
                #previous_position = list(config.perms_list[i - 1][current_vnr_id].keys())[0] #list(config.perms_list[i - 1][vnr[1][0]].keys())[0]
                # get the result of current vnr in previous perm
                #current_success = config.perms_list[i - 1][current_vnr_id].get(previous_position) #config.perms_list[i - 1][vnr[1][0]].get(previous_position)
                current_success = config.vns_per_perm[current_vnr_id][idx]
            else:
                print "UNHANDLED??"
            # memoize success/fail of current vnr
            config.vns_per_perm.update({current_vnr_id: {idx: current_success}}) #({vnr[1][0]: {idx: current_success}})
            #print "config.vns_per_perm",config.vns_per_perm
            # memoize the ordered subsets for each sequence up to the n-2 left most positions
            # last 2 items must be calculated anyways
            if idx < len(per) - 2:
                memoize_perms()
        # if current perm has at least 2 previous
        #if multiprocessing.current_process().name == "Process-1":
            #print multiprocessing.current_process().name,"config.perms_list",config.perms_list
        if i > 1:
            config.perms_list.pop(i - 2)
        current_accepted = int(len(config.VWSNs))
        calculate_cost()
        config.current_perm_results.update({str(config.current_key_prefix): {'permutation':i,'acceptance': float(current_accepted)/float(config.numvn),
                                                                             'overall_cost': float(config.overall_cost),
                                                                             'vwsns': list(config.VWSNs)}})
        config.perms_list.update({i: config.vns_per_perm})

        # check if current acceptance rate is at least lower bound, otherwise don't evaluate
        config.solution_progress.append({'time': time.time() - config.start, 'proc_id': config.proc, 'perm': i,
                                         'acceptance': float(current_accepted)/float(config.numvn), 'cost': float(config.overall_cost)}) #,'vwsns': list(config.VWSNs)})

        if current_accepted > l_bound.value: #config.min_feasible_vnrs:
            #print "|",current_accepted, l_bound.value, config.min_feasible_vnrs
            l_bound.value = current_accepted
            evaluate(False)
        elif current_accepted == l_bound.value:
            #print "||", current_accepted, l_bound.value, config.min_feasible_vnrs
            evaluate(False)
    result_q.put(config.current_perm_block_best)
    solution_progress_q.put(config.solution_progress)
 
def evaluate(is_debug):
    #if config.current_perm_block_best['acceptance'] != config.current_perm_block_acceptance:
        #print config.current_perm_block_best['acceptance'], config.current_perm_block_acceptance,"config.current_perm_block_acceptance"
    config.eval_counter += 1
    perm_result = copy.deepcopy(config.current_perm_results[config.current_perm_results.keys()[0]])
    #if is_debug:
        #print "config.iteration",config.iteration
        #print config.eval_counter, "-eval-", config.proc, "-proc-", perm_result['permutation'], '-permutation'
        #print "perm_results",perm_result
        #print "config.current_perm_block_acceptance",config.current_perm_block_acceptance
        #print "config.current_perm_block_cost",config.current_perm_block_cost
    if config.first_find_flag == False:
        #print "EVALUATE"
        config.first_find_flag = True
        config.first_find = perm_result['permutation']
        #print "config.first_find", config.first_find
        #print multiprocessing.current_process(),"perm_result['overall_cost']", perm_result['overall_cost'], "perm", perm_result['permutation']
    #if config.current_perm_block_acceptance == perm_result['acceptance']:
    #print config.current_perm_block_best['acceptance'] ,perm_result['acceptance'],config.current_perm_block_best['overall_cost'], perm_result['overall_cost']
    if config.current_perm_block_best['acceptance'] == perm_result['acceptance']:
        #if config.current_perm_block_cost > perm_result['overall_cost']:
        if config.current_perm_block_best['overall_cost'] > perm_result['overall_cost']:
            #config.current_perm_block_cost = perm_result['overall_cost']
            config.current_perm_block_best = {'process': int(config.proc), 'first_find': int(config.first_find),
                                              'permutation': int(perm_result['permutation']),
                                              'acceptance': float(perm_result['acceptance']),
                                              'overall_cost': float(perm_result['overall_cost']),
                                              'committed': nx.DiGraph(config.wsn_for_this_perm),
                                              'mappings': list(perm_result['vwsns']),'srcs':list(config.src_loads)}
            config.current_perm_block_best.update({
                'proc_time': time.time() - config.start})
            config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
    #elif config.current_perm_block_acceptance < perm_result['acceptance']:
    elif config.current_perm_block_best['acceptance'] < perm_result['acceptance']:
        config.first_find = perm_result['permutation']
#        config.current_perm_block_acceptance = perm_result['acceptance']
#        config.current_perm_block_cost = perm_result['overall_cost']
        config.current_perm_block_best = {'process': int(config.proc), 'first_find': int(config.first_find),
                                          'permutation': int(perm_result['permutation']),
                                          'acceptance': float(perm_result['acceptance']),
                                          'overall_cost': float(perm_result['overall_cost']),
                                          'committed': nx.DiGraph(config.wsn_for_this_perm),
                                          'mappings': list(perm_result['vwsns']), 'srcs': list(config.src_loads)}
        config.current_perm_block_best.update({
            'proc_time': time.time() - config.start})
        config.committed_wsn = nx.DiGraph(config.wsn_for_this_perm)
    if is_debug:
        print "Eval - config.current_perm_block_best",config.current_perm_block_best


def select_best(perm_block_results):
    while perm_block_results.empty() is False:
        #perm_block_result = overall_results.get()
        perm_block_result = perm_block_results.get()
        #print "perm_block_result", perm_block_result
        if config.current_test_best['acceptance'] < perm_block_result['acceptance']:
            config.current_test_best = perm_block_result
        elif config.current_test_best['acceptance'] == perm_block_result['acceptance']:
            if config.current_test_best['overall_cost'] > perm_block_result['overall_cost']:
                config.current_test_best = perm_block_result
    return config.current_test_best

def generate_output(best):
    acceptance_rate = float(best['acceptance'])# / float(config.numvn)
    cpu_count = multiprocessing.cpu_count()
    print "cpus-",cpu_count
    if config.numvn_to_permute > 3:
        slice_size = math.factorial(config.numvn_to_permute) / cpu_count #4
    else:
        slice_size = math.factorial(config.numvn_to_permute)
    #slice_size = math.factorial(config.numvn_to_permute) / cpu_count
    print slice_size,"slice size of", math.factorial(config.numvn_to_permute), multiprocessing.cpu_count()
    first_f = int(best['first_find']+ (slice_size * (best['process'] - 1)))
    print first_f, "first find of first_find, process", best['first_find'], best['process']
    output_dict = {
        # First three copied from input vector
        'nwksize': config.nwksize,
        'numvn': config.numvn,
        'numvn_to_permute': config.numvn_to_permute,
        'iteration': config.iteration,
        # Following result from algorithm execution
        'first_find': first_f,  # unclear/incorrect
        'process': best['process'],
        'proc_time': config.proc_time,
        'acceptance': acceptance_rate,
        'mapping': best['mappings'],
        'objective': best['overall_cost'],
        'committed': best['committed'],
        'first_best':config.first_best,
        'best_proc_time':best['proc_time'],
        'density': config.density,
        'progress': progress,
        'vnlist' : config.vnlist,
        'current_vnlist' : config.current_vnlist,
        'isbestcombo': config.isbestcombo
    }
    config.result_vectors.append(output_dict)

def write_to_File():
    output_file_name = ''
    if is_fixed_vnr:
        output_file_name = dir_path + 'results/'  + str(num_vnrs) + '/' + str(start_iter)+'_'+str(finish_iter)+'_' + 'fixed_vnrs_' + str(fixed_iter) + '_' + input_file_name
    else:
        #output_file_name = dir_path2 + 'results/test/' + str(num_vnrs) + '/' + str(start_iter) + '_' + str(finish_iter) + '_' + input_file_name + '-*'
        output_file_name = dir_path2 + str(start_iter) + '_' + str(finish_iter) + '_' + input_file_name + '-*'
    try:
        with open(output_file_name, 'w') as handle:
            pickle.dump(config.result_vectors, handle)
    except Exception as e:
        print (e)
        return -1
    return 0


if __name__ == '__main__':
    has_not_completed = True
    #  if True the same request is used for all iterations/topology, different request for each otherwise
    is_fixed_vnr = False #True
    nwk_size = 50
    num_vnrs = 1
    fixed_iter = 4 # IGNORE
    iter_limit = 999
    # to reduce the size of output files
    if num_vnrs == 8:
        iter = 200
    else:
        iter = 1000
    start_iter = 0
    finish_iter = start_iter + iter
    dir_path = '/media/roland/Docker/ftp/results/InputFiles/'
    dir_path2 = './InputVectors/'
    dir_path3 = '../VNE_LP_/input_vectors/with_node_constraints/variable_topology/varied_topology/'#'tests/50/parallel/'
    input_file_name = 'input_vector_' + str(nwk_size) + '_' + str(num_vnrs) + '.pickle'
    test_vectors = pickle.load(open(dir_path2+input_file_name, 'rb'))

    while has_not_completed:
        for test_case in test_vectors:
            jobs = list()
            interest_coordinates = test_case['interests']
            node_coordinates = test_case['substrate_coords'] #.output_for_heuristic()[0]
            result_q = multiprocessing.Queue()
            min_feasible_q = multiprocessing.Queue()
            reinitialize()
            config.nwksize = test_case['nwksize']
            config.numvn = len(test_case['vnlist'])
            config.density = test_case['density']
            solution_progress_q = multiprocessing.Queue()
            if config.numvn >= 1:
                temp_proc = test_case['temp_process']
                config.err_matrix = temp_proc.temperature_process_mae(node_coordinates, interest_coordinates)
                config.iteration = test_case['iteration']
                if config.iteration  >= start_iter and config.iteration  < finish_iter :# and config.iteration < 10:
                    vnrs_list = []
                    if is_fixed_vnr:
                        for vnrs in fixed_vnrs:#test_case['vnlist']:
                            converted_vnr = vnrs.convert_to_heuristic()
                            vnrs_list.append(converted_vnr)
                    else:
			# Explanation of parsing the input files is in the Readme 
                        for vnrs in test_case['vnlist']:
                            converted_vnr = vnrs.convert_to_heuristic()
                            vnrs_list.append(converted_vnr)
		    config.numvn = len(vnrs_list)
                    config.vnlist = list(vnrs_list)
                    generated_wsn = []
                    generated_wsn = test_case['adjacency_matrix']# .output_for_heuristic()
                    wsn_substrate = []
                    wsn_substrate = WSN(config.X,config.Y,generated_wsn)
                    wsn_substrate.set_nones_position(node_coordinates) #generated_wsn[0])
                    adjacencies = []
                    adjacencies = wsn_substrate.get_adjacency_list() #.get_wsn_substrate().adjacency_list()
                    config.wsn = []
                    config.wsn = wsn_substrate.get_wsn_substrate()
                    two_hops_list = wsn_substrate.get_two_hops_list()
                    conflicting_links_dict = []
                    conflicting_links_dict = wsn_substrate.get_conflicting_links()
                    all_path_lngths = dict(nx.all_pairs_shortest_path_length(config.wsn))
                    avg_pth_lngths = {}
                    for k,v in all_path_lngths.items():
                        lngth = 0.0
                        for m,l in v.items():
                            lngth += l
                        avg_pth_lngths.update({k:(float(lngth)/float(len(all_path_lngths)),)})
                    print "avg_pth_lngths",avg_pth_lngths
                    conflict_weights = []
                    for key,value in conflicting_links_dict.items():
                        for k,v in value.items():
                            conflict_weights.append((len(v),key,k))
                    print "conflict_weights",conflict_weights
                    print "adjacencies",adjacencies
                    sorted_conflict_weights = sorted(conflict_weights, reverse=True)
                    confl_wghts = []
                    for w in sorted_conflict_weights:
                        neigbrs = len(adjacencies[w[1]]) + len(adjacencies[w[2]])
                        print w[1],"->",w[2],"INEX",w[0],"#neighbors",neigbrs,"avg_pth_lnght", avg_pth_lngths[w[1]], avg_pth_lngths[w[2]], config.wsn[w[1]][w[2]]["weight"], "ratios", neigbrs*w[0], float(neigbrs)/float(w[0]), float(w[0])/float(neigbrs)
                        confl_wghts.append((float(w[0])/float(neigbrs),)+w)
                    print ""
                    print "confl_wghts",confl_wghts
                    min_hops_dict = []
                    min_hops_dict = get_min_hops(config.main_sink)
                    print "min_hops_dict",min_hops_dict
                    init_cost = wsn_substrate.get_initial_link_weight()
                    max_hops = 0
                    for k,v in min_hops_dict.iteritems():
                        if v > max_hops:
                            max_hops = v
                    shortest_path, path_nodes = [],[]
                    config.committed_wsn = nx.DiGraph(config.wsn)
                    #config.sp_alg_str = "A*"
                    config.sp_alg_str = "Dijkstra"
                    config.skipit = 0
### Embedding starts here ####
                    config.start = time.time()
                    sorted_feasible_vnrs_list = []
                    ## Node Mapping starts here
                    sorted_feasible_vnrs_list = map_nodes(list(vnrs_list))
                    print "node mapping finished in", (time.time() - config.start),"s"
                    config.subset_sums.sort(key=lambda x: x[0])  # , reverse=True)
                    best_combo = ()
                    least_sink_load = 100
                    low_bound = 0
                    inital_bound = 0
		    ## Establish initial lower bound
                    inital_bound = get_lower_bound(sorted_feasible_vnrs_list,None)
                    calculate_cost()
                    config.current_perm_results.update(
                        {str(config.current_key_prefix): {'permutation': -2, 'acceptance': float(int(len(config.VWSNs)))/float(config.numvn),
                                                          'overall_cost': float(config.overall_cost),
                                                          'vwsns': list(config.VWSNs)}})
                    print "config.current_perm_results",config.current_perm_results
                    print "init_cost_",init_cost
                    # establish initial result baseline
                    evaluate(True)
                    config.current_perm_block_best.update({
                        'proc_time': time.time() - config.start})
                    config.first_best = dict(config.current_perm_block_best)
                    src_combo = []
                    sum_load = 0
                    if inital_bound > 0:
                        for vn in config.first_best['mappings']:
                            l = 0
                            #if vn[2] > 1:
                            if vn['hop'] > 1:
                                l = 2*vn['load']
                            else:
                                l = vn['load']
                            sum_load += l
                            src_combo.append((vn['src'],l))
                    best_combo = ((len(src_combo),sum_load,src_combo))
                    vnr_set = []
                    for sum_src in config.subset_sums:
                        if sum_src[0] >= inital_bound:
                            vnr_set = set(vnr_set) | set(sum_src[2])
                            low_bound = get_lower_bound(sorted_feasible_vnrs_list,sum_src)
                            calculate_cost()
                            config.current_perm_results.update(
                                {str(config.current_key_prefix): {'permutation': -1, 'acceptance': float(int(len(config.VWSNs)))/float(config.numvn),
                                                                  'overall_cost': float(config.overall_cost),
                                                                  'vwsns': list(config.VWSNs)}})

                            if low_bound > inital_bound:
                                inital_bound = low_bound
                                best_combo = sum_src
                                least_sink_load = sum_src[1]
                                #print " > config.current_perm_results",config.current_perm_results[config.current_perm_results.keys()[0]]
                                evaluate(True)
                                #print "evaluate -2"
                                config.current_perm_block_best.update({
                                                          'proc_time':time.time() - config.start})
                                config.first_best = dict(config.current_perm_block_best)
                                #print config.wsn_for_this_perm.edge[0][adjacencies[0][0]]['load'],"-",sum_src[1]
                            elif low_bound == inital_bound:
                                #if sum_src[1] < least_sink_load: #config.wsn_for_this_perm.edge[0][adjacencies[0][0]]['load'] < :
                                if config.overall_cost < config.current_perm_block_best['overall_cost']:
                                #if config.overall_cost < config.first_best['overall_cost']:
                                #if config.current_perm_block_cost > config.first_best['overall_cost']:
                                    inital_bound = low_bound
                                    best_combo = sum_src
                                    least_sink_load = sum_src[1]
				    evaluate(True)
                                    config.current_perm_block_best.update({
                                        'proc_time': time.time() - config.start})
                                    config.first_best = dict(config.current_perm_block_best)
                    if len(vnr_set) != config.numvn: # == set(best_combo[2])
                        print "size-",len(vnr_set)
                    print (time.time() - config.start),"first_best",config.first_best
                    first_cost = 0
                    first_load = 0
                    if inital_bound > 0:
                        for u, v in config.first_best['committed'].edges():
                            first_cost += config.first_best['committed'][u][v]['weight']
                            first_load += config.first_best['committed'][u][v]['load']

                    #print "setting lower bound finished in",
                    #lower_bound = get_lower_bound(sorted_feasible_vnrs_list)
                    min_feasible_q.put(inital_bound)
                    l_bound = multiprocessing.Value('i',inital_bound)
                    reduced_sorted_feasible_vnrs_list = []
                    #for request in sorted_feasible_vnrs_list:
                    #    if (request[3]['src'], request[3]['sink_load']) in best_combo[2]:
                    #        print (request[3]['src'], request[3]['sink_load'])
                    #        reduced_sorted_feasible_vnrs_list.append(request)
                    #sorted_feasible_vnrs_list = reduced_sorted_feasible_vnrs_list
		    
		    ## exhaustive parallel search starts here
                    generate_independent_perm_blocks(sorted_feasible_vnrs_list, result_q, inital_bound, min_feasible_q, l_bound, solution_progress_q )
                    del config.current_vnlist
                    config.current_vnlist = list(sorted_feasible_vnrs_list)
                    #print ("--nwksize", config.nwksize, "numvn", config.numvn, "iter", config.iteration)
                    #vis.display_edge_attr(config.wsn)
                    for j in jobs:
                        j.start()
                    overall_results = multiprocessing.Queue()
                    progress = []
                    for j in jobs:
                        j.join()
                        res = result_q.get()
                        #print "res",res.items()
                        prog = solution_progress_q.get()
                        progress.extend(prog)
                        overall_results.put(res)
                        #print overall_results
                    for j in jobs:
#                        j.join()
                        j.terminate()
                    best = select_best(overall_results)
                    if best['mappings'] is not None:
                        print "--best",best
                        for b in best['mappings']:
                            #print "b",b
                            #print "best", b[0],b[1],b[2]
                            sink_l = 0
                            if b['hop'] > 1:
                                sink_l = 2*b['load']
                            else:
                                sink_l = b['load']
                            if ((b['src'], sink_l)) not in best_combo[2]:
                                print b['src'], sink_l, "NOT IN BEST"
                                config.isbestcombo = False
                            #vis.display_vn_edge_allocation(b[4])
                        #for srcs in config.first_best['srcs']:
                        #    print "srcs",srcs
                        jobs = []
                        end = time.time()
                        config.proc_time = (end - config.start)
                        print("Optimal solution is:",'in process',best['process'],'first_find',best['first_find'],"permutation-", best['permutation'],"acceptance-",best['acceptance'],"overall_cost-",best['overall_cost'],"processing_time-",config.proc_time)
            #           print("Optimal solution is:",'first_find',best['first_find'],"permutation-", best['permutation'],"acceptance-",best['acceptance'],"overall_cost-",best['overall_cost'],"processing_time-",config.proc_time)
                        # print "best['mappings']['vwsns']",best['mappings']['vwsns']
                        #vis.display_edge_attr(best['committed'])
                        print "Sink link load is ",best['committed'].edge[0][adjacencies[0][0]]['load']
                        print "post embedding"
                        final_cost = 0
                        final_load = 0
                        for u,v in best['committed'].edges():
                            final_cost += best['committed'][u][v]['weight']
                            final_load += best['committed'][u][v]['load']
                            #print best['committed'][u][v]
                        #vis.display_edge_attr(best['committed'])
                        print "init cost", init_cost
                        print "first cost", first_cost
                        print "first load", first_load
                        print "final cost",final_cost
                        print "final load", final_load

                        config.acceptance = config.max_accepted_vnrs
                        generate_output(best)
                        #time.sleep(1)
                    else:
                        print "NO SOLUTION!"
                    print "\n------------------------------------\n\n"

     #               vis.draw_substrate(wsn_substrate)
     #               vis.draw_graph(wsn_substrate)

        write_to_File()
        del config.result_vectors
        config.result_vectors = list()
        print "Test cases" + str(start_iter) +"_"+ str(finish_iter) +"completed"
        start_iter = finish_iter #start_iter + 100
        finish_iter = start_iter + iter
        if start_iter > iter_limit:
            has_not_completed = False
            print "TESTS COMPLETED"
            break
    

