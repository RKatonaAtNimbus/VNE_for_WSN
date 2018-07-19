"""
Script for testing the VNE solvers, LPP and heuristic.

Generate a network.
Generate virtual network requests.
"""
from vnr_generator import output_vn_links_for_lp

from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
from pyutilib.common import ApplicationError

#from vne_lp import model
from mc_flow_lp import model
from mc_flow_cost import model as model_min_cost
from itertools import combinations
from time import time

from temperature_process_gen import TemperatureProcess
import networkx as nx
from network_builder import NodeStore, Node
from math import sqrt
from copy import deepcopy
from os import remove as remfile    # To remove files

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

max_comms_range = 500

###############################################################################
#
#    Code for processing results.
#
# The goal of the algorithm is to embed the maximum number of networks,
# so k out of n. To do this slack is introduced in the constraints, and
# the objective is to minimize the slack.
# After running once, a set of embeddings needs to be selected that satisfy
# all the constraints (no more slack) and minimizes the amount of resources
# allocated.
###############################################################################

def output_for_lp(_file, nodes, substrate_edges):
    """Output a network to a file object, in the format required by
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

def generate_lp_input(substrate, adj_matrix, interests, temp_proc, vnrs, lp_path):
    """
    Generate the input file for the LP solver.

    Parameters:
    substrate       -- Coordinates of the substrate nodes
    adj_matrix      -- Adjacency matrix of the substrate.
                       Indices follow the substrate list.
    interests       -- List of interest points.
    temp_proc       -- Temperature process to generate the error matrix.
    vnrs            -- List of virtual network request.
                       VirtualNetGenerator objects.
    lp_path         -- Where the input file should be saved.

    Returns:
    error code and mapping of meta_node_id: vnr
    Error codes:
    0   for success
    -1  for formatting errors - mapping is None
    -2  if none of the VNRs can be embedded - mapping is None
    """
    #
    # Must generate:
    #   - augmented graph, including meta nodes for interest points and
    #       meta edges
    #   - don't need the error matrix, just need to find the nodes that respect
    #       the measurement error.
    #
    # Meta edges are first - which are the possible substrate nodes?
    #   - start with the 2-hop neighbs of the meta node
    #   - eliminate the nodes that exceed the maximum meas error
    #   
    # Use only the interest points from the vnrs.

    # Mapping meta_node_id -> VNR
    metanode_vnr = dict()

    # Convert the temperature process to something we can work with
    tp = TemperatureProcess(1)
    tp.area_width = temp_proc.area_width
    tp.athmosphere = temp_proc.athmosphere
    tp.num_sources = temp_proc.num_sources
    tp.temp_sources = temp_proc.temp_sources
    temp_proc = tp

    def valid_meas(node, vnr):
        """Check if the node is a valid source for the vnr, from meas error pov"""
        ip = interests[vnr.links[0][0][0]]
        meas_err = temp_proc.get_meas_error((node.x, node.y), ip)
        return meas_err < vnr.links[0][0][1]

    # Generate augmented graph
    # Format for meta_nodes: (meta_node, [neighbs])
    meta_nodes = []

    # First generate the node store
    # Need to get area width, so find min and max x,y
    x_coords, y_coords = zip(*substrate)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    #area_diag = sqrt((x_max-x_min)**2 + (y_max-y_min)**2)
    area_diag = tp.area_width * sqrt(2)
    node_store = NodeStore(int(area_diag), max_comms_range)
    for idx,(x,y) in enumerate(substrate):
        node_store.add(Node(x, y, idx))
    meta_node_id = len(substrate)
    # Keep track of the Virtual Links that can be embedded
    # * use virtual links rather than the orig vnrs because we have to keep
    #   those intact
    feasible_vn_links = []
    # Now generate meta nodes and links
    for vnr in vnrs:
        ip = vnr.links[0][0][0]     # Awful, I know...
        ip_coords = interests[ip]
        meta_node = Node(ip_coords[0], ip_coords[1], meta_node_id)
        #print "Meta node", meta_node_id, "assigned to", ip
        # Find the closest node
        nodes = list(node_store.get_nodes())
        nodes.sort(key=lambda n: sqrt((n.x-ip_coords[0])**2+(n.y-ip_coords[1])**2))
        meta_neighbs = [nodes[0]]
        # Find 2-hop neighbours
        two_hop = []
        for n in meta_neighbs:
            two_hop.extend(node_store.find_neighbours(n))
        meta_neighbs.extend(two_hop)
        # If there are no candidates in the first hop neighbourhood, look 2 hops
        if len(filter(lambda node: valid_meas(node, vnr), meta_neighbs)) == 0:
            two_hop = []
            for n in meta_neighbs:
                two_hop.extend(node_store.find_neighbours(n))
            meta_neighbs.extend(two_hop)
        # Eliminate duplicates
        meta_neighbs = list(set(meta_neighbs))
        # Eliminate nodes with high meas error
        meta_neighbs = filter(lambda node: valid_meas(node, vnr), meta_neighbs)
        # Remove the sink - no sensing there
        try:
            meta_neighbs.remove(node_store.get_nodes()[0])
        except ValueError:
            pass
        # If there are no candidates, meta node cannot be used
        if len(meta_neighbs) == 0:
            logger.info("IP %d has no candidate", ip)
            continue
        # Now we consider the meta node and we include the VNR for embedding
        meta_nodes.append((meta_node, meta_neighbs))
        # This VNR can be embedded, just modify the source id to the meta node
        vl = vnr.links[0]
        feasible_vn_links.append(((meta_node_id, vl[0][1]), 0, vl[2], vl[3], vl[4]))
        # Map this meta node to the original VNR for later
        metanode_vnr[meta_node_id] = vnr
        meta_node_id += 1

    if len(feasible_vn_links) == 0:
        logger.warning("No VNRs can be embedded")
        return -2, None

    # Build list of substrate edges
    substrate_edges = []
    for src, dests in adj_matrix.items():
        substrate_edges.extend(
                # src, dest, prr, capacity, latency
                # update prr to account for 2 retransmissions
                [(src,dst,
                    int((1-(plr/100.)**2)*10**4)/float(10**4),
                    100, 1)
                    for (dst, plr) in dests])
    # Add the meta edges
    for meta_node, neighbs in meta_nodes:
        substrate_edges.extend(
                # meta_node, substrate_candidate, perfect link, very high cap, 0 latency
                [(meta_node._id, n._id, 1, 10**6, 0) for n in neighbs])

    # Meta node coordinates
    meta_coords = [(meta_node.x, meta_node.y)
                            for (meta_node, neighbs) in meta_nodes]

    ##### Get generating!
    lp_file = open(lp_path, 'w')
    if output_for_lp(lp_file, substrate + meta_coords, substrate_edges)!= 0:
        logger.error("Error writing network model")
        return -1, None
    if output_vn_links_for_lp(lp_file, feasible_vn_links) != 0:
        logger.error("Error writing VN requests")
        return -1, None
    lp_file.close()
    return 0, metanode_vnr

def is_path(src, dest, _link_list):
    """Determines if there's a path between src and dest in link_list"""
    link_list = list(_link_list)
    available_links = len(link_list)
    def find_link_starting_at(node, links):
        for l in links:
            if l[0] == node:
                return l
        return None
    l = (None, src)
    while l[1] != dest:
        src, dst = l
        l = find_link_starting_at(l[1], link_list)
        if l is None:
            logger.warning("Didn't find %s in %s", str((src, dst)), str(link_list))
            return False, 0
        link_list.remove(l)
        available_links -= 1
    if available_links > 0:
        return True, available_links
    return l[1] == dest, 0

def _process_(solver, inst):
    """
    Solves an instance multiple times to eliminate loops in the 
    embeddings.

    Parameter:
    inst        -- model instance.

    Returns:
    the instance solution without loops.
    """
    def get_mapped_links(inst):
        links_mapped = dict()
        for k,y in inst.y_vl_vr.items():
            if y.value == 1:
                links_mapped.setdefault((k[0], k[1]), []).append((k[2], k[3]))
        return links_mapped
    extra_links = 1
    while extra_links > 0:
        extra_links = 0 
        logger.info("Solving")
        try:
            solver.solve(inst)
        except ApplicationError:
            return
        links_mapped = get_mapped_links(inst)
        for vn, links in links_mapped.items():
            stat, extra = is_path(vn[0], vn[1], links)
            extra_links += extra
            if stat:
                inst.flow_length[vn[0], vn[1]].value = len(links) - extra
                inst.flow_length[vn[0], vn[1]].fixed = True
                #logger.debug("For %s fixed flow length at %d", vn, len(links) - extra)
            else:
                logger.error("Error")
                return
    logger.info("Done")
    #for k,y in inst.y_vl_vr.items():
    #    if y.value == 1:
    #        print k[0], k[1], "->", k[2], k[3]
    #return inst

def conflict_set(inst, a, b):
    """Returns the conflict set of a link i,j"""
    cs = [(a,b)]
    cs.extend((i,k) for (i,k) in inst.E_real if i == b)
    cs.extend((k,l) for (k,l) in inst.E_real if (k,b) in inst.E_real and k != a and l!= a)
    cs.extend((j,i) for (j,i) in inst.E_real if j == a and i != b)
    cs.extend((i,j) for (i,j) in inst.E_real if j == a and i != b)
    cs.extend((k,l) for (k,l) in inst.E_real if (a,l) in inst.E_real and\
                        k != a and k != b and l != b and (k,b) not in inst.E_real)
    return cs

def algorithm(input_vector, version=''):
    """
    The objective of the program has three components:
        - amount of resources allocated
        - capacity slack
        - QoS slack,
    with their own weights.

    Algorithm steps:
    1. Give priority to the capacity slack weight. Solve. Minimising 
    the objective implies that we will determine if all the requests can be
    embedded or not. If there's capacity slack, go to 2. Otherwise, go to 3.
    2. Give priority to the allocation. From the embeddings that
    have slack in the constraints select the one with highest resource
    usage. Eliminate then go to 1. 
    3. Return the allocation.

    Parameters:
    input_vector  -- data input file.

    Returns:
    Final allocation as (objective value, mapping, acceptance ratio), or None.
    """
    solver = SolverFactory('cbc')

    def get_slack(inst):
        """Solving once  -- return the capacity violating VNs"""
        ## Organise the allocations into a dictionary, keyed by substrate links
        allocations = {}
        for s,t,i,j in inst.y_vl_vr.keys():
            if inst.y_vl_vr[s,t,i,j].value == 1:
                allocations.setdefault((i,j), []).append(((s,t), inst.req_l[s,t]))
        ## Analyze the slack
        link_cap_slack = [] # VNs that cause capacity slack
        for k,y in inst.capacity_slack.items():
            if y.value > 0:
                # k[0] -> k[1] link has capacity slack.
                # Determine the link's conflict area
                # Which VNs have an impact on this link?
                for l in conflict_set(inst, k[0], k[1]):
                    if l in allocations:
                        link_cap_slack.extend(allocations[l])
        latency_slack = []  # vns that have latency slack
        for k,y in inst.latency_slack.items():
            if y.value > 0:
                latency_slack.append((k,y.value))
        reliability_slack = []  # vns that have reliability slack
        for k,y in inst.reliability_slack.items():
            if y.value > 0:
                reliability_slack.append((k,y.value))
        return link_cap_slack, latency_slack, reliability_slack

    def get_link_cap_slack(inst, metanode_vnr):
        """
        Determines which VNs influence which link and how many times
        
        Returns dictionary:
        link:[(vn, count)]
        """
        ## Organise the allocations into a dictionary, keyed by substrate links
        allocations = {}
        for s,t,i,j in inst.y_vl_vr.keys():
            if inst.y_vl_vr[s,t,i,j].value == 1:
                allocations.setdefault((i,j), []).append(((s,t), inst.req_l[s,t]))
        ## Analyze the slack
        link_cap_slack = {}
        for k,y in inst.capacity_slack.items():
            if y.value > 0:
                # k[0] -> k[1] link has capacity slack.
                # Determine the link's conflict area
                # Which VNs have an impact on this link?
                link_effect = {}
                for l in conflict_set(inst, k[0], k[1]):
                    if l not in allocations: continue
                    for vn in allocations[l]:
                        _vn = metanode_vnr[vn[0][0]].links[0][0][0]
                        _vn_load = metanode_vnr[vn[0][0]].links[0][2]
                        if _vn in link_effect:
                            link_effect[_vn] += _vn_load
                        else:
                            link_effect[_vn] = _vn_load
                link_cap_slack[k] = (y.value, link_effect)
        return link_cap_slack

    def vn_allocation(inst, vn):
        """Return the resources occuppied by this VN"""
        alloc = []
        for k,y in inst.y_vl_vr.items():
            if y.value == 1 and k[0] == vn[0] and k[1] == vn[1]:
                alloc.extend(conflict_set(inst, k[2], k[3]))
        return list(set(alloc))

    def print_slack(slack):
        s = []
        for k,y in slack.items():
            if abs(y.value) > 0:
                s.append((k, y.value))
        s.sort(key=lambda x:x[1])
        logger.debug("%s", str(s))

    def is_there_slack(slack):
        for k,y in slack.items():
            if abs(y.value) > 0:
                return True
        return False

    def equals(l1, l2):
        if len(l1) != len(l2): return False
        _l1 = deepcopy(l1)
        _l2 = deepcopy(l2)
        _l1.sort()
        _l2.sort()
        for a,b in zip(_l1, _l2):
            if a != b: return False
        return True

    def update_summary(summary_set, _set, link_usage):
        for idx, l in enumerate(summary_set):
            if equals(_set, l[0]):
                if link_usage[1][0] >= l[1][1][0]:
                    summary_set[idx] = (_set, link_usage)
                return
        summary_set.append((_set, link_usage))

    def test_remove_subset(link_slack, subset):
        """
        Verify what slack will be left after removing a subset
        """
        to_remove = link_slack[1][0]
        for vn, slack in link_slack[1][1].items():
            if vn in subset:
                to_remove -= slack
        return to_remove

    def find_smallest_subsets(link_slack, init_size):
        """
        Find the smallest subset of link_slack that if removed, 
        will eliminate the slack
        """
        to_remove = link_slack[1][0]
        conflicts = link_slack[1][1].items()
        # sort conflicts in decreasing order of contribution to slack
        conflicts.sort(key=lambda x:x[1], reverse=True)
        solution = []
        for size in range(init_size,len(conflicts)+1):
            for i in range(len(conflicts)):
                found = False
                for _iter in combinations(conflicts[i:], size):
                    if sum(zip(*_iter)[1]) >= to_remove:
                        solution.append(list(zip(*_iter)[0]))
                        found = True
                    else:
                        pass
                        #break
                if not found: break
            if len(solution) > 0: break
        return solution

    def remove_candidate_from(vnr_list, candidate):
        """
        If vnr_list is a list of VirtualNetGenerator objects and
        candidate represents a list of VN source nodes,
        return the list on VNs without the candidates.
        """
        cand_set = set(candidate)
        rem_list = list()
        for vn in vnr_list:
            if vn.links[0][0][0] in cand_set:
                pass
            else:
                rem_list.append(vn)
        return rem_list

    def build_solution(objective, mapping, acceptance, start_time):
        return {
                'nwksize': input_vector['nwksize'],
                'numvn': input_vector['numvn'],
                'iteration': input_vector['iteration'],
                'acceptance': acceptance,
                'mapping': mapping,
                'objective': objective,
                'proc_time': time() - start_time
                }

    # Measure the start time
    start_time = time()

    lp_input = 'tmp_lp_input_%s_%d.dat'%(input_vector['nwksize'], input_vector['numvn'])
    nwk_gen = input_vector['substrate_coords']
    adj_matrix = input_vector['adjacency_matrix']
    ips = input_vector['interests']
    temp_proc = input_vector['temp_process']
    vnrs = input_vector['vnlist']
    vnr_list = list(vnrs)

    logger.info("Solving for vnrs %s", str(vnrs))

    # Generate the input based on the current vn list
    errcode, metanode_vnr = generate_lp_input(nwk_gen, adj_matrix, ips,
                                              temp_proc, vnr_list, lp_input)

    if errcode < 0:
        # if errcode is -2 need to return a valid solution, 0 acc rate
        if errcode == -2:
            return build_solution(0, {}, 0, start_time)
        else:
            return None
    inst = model.create_instance(lp_input)

    mappable_vns = set([v.links[0][0][0] for v in metanode_vnr.values()])

    vnr_list_copy = []
    for vn in vnr_list:
        if vn.links[0][0][0] in mappable_vns:
            vnr_list_copy.append(vn)
    vnr_list = vnr_list_copy

    # Step 0 -- eliminate VNs that have unadmittable reliability
    inst.objective_reliability.activate()
    inst.objective_capacity_priority.deactivate()
    inst.objective_resource_priority.deactivate()
    _process_(solver, inst)
    
    # Find the vns that cause reliability slack
    reliability_slack = []  # vns that have reliability slack
    for k,y in inst.reliability_slack.items():
        if y.value > 0:
            reliability_slack.append((k,y.value))
    if len(reliability_slack) > 0:
        logger.info("We have non-feasible VNs (for reliability)")
        logger.info("%s",reliability_slack)
        for vn_meta in reliability_slack:
            vn = metanode_vnr[vn_meta[0][0]]
            logger.debug("Removing %s", str(vn))
            vnr_list.remove(vn)

    slack = [0]
    # Store VN list subsets that represent the possible solutions
    # Also include the size of the candidate subsets
    solutions = [(list(vnr_list), 1)]
    best_sol = None
    while len(solutions) > 0:
        #logger.info("Possible solutions: %s", str(solutions))
        # Pop the first possible solution
        (vnr_list, init_size) = solutions[0]

        # If the current candidate solution is empty, go to the next
        if len(vnr_list) == 0:
            solutions = solutions[1:]
            continue

        # If there's a solution with higher acceptance ratio than what we 
        # could get here from init_size (which is size of subset to _remove_)
        # there's no reason to expand more.
        if best_sol is not None\
                and len(vnr_list) - init_size < len(best_sol['mapping'].items()):
            solutions = solutions[1:]
            continue

        logger.info("Testing %s", str(vnr_list))
        # Step 1 -- Solve, minimising the total slack
        logger.info("Step 1: minimising slack")

        # Generate the input based on the current vn list
        errcode, metanode_vnr = generate_lp_input(nwk_gen, adj_matrix, ips,
                                                  temp_proc, vnr_list, lp_input)

        if errcode < 0:
            solutions = solutions[1:]
            continue
            # if errcode is -2 need to return a valid solution, 0 acc rate
            #if errcode == -2:
                # TODO - don't return but explore the next in line
                # return build_solution(0, {}, 0, start_time)
            #else:
            #    return None
        inst = model.create_instance(lp_input)

        # Set up the objectives
        inst.objective_reliability.deactivate()
        inst.objective_capacity_priority.activate()
        inst.objective_resource_priority.deactivate()

        # Solve
        _process_(solver, inst)

        # Get the slack
        cap_slack, lat_slack, rel_slack = get_slack(inst)
        slack = cap_slack + lat_slack + rel_slack

        if len(slack) == 0:      # No slack, no constraint violation
            # This is a valid solution
            # How much load on each substrate link
            sub_load = {}
            # Retrieve the resource allocation
            mapping = {}
            mapping_meta = {}   # Mapping that uses metanodes for VNRs
            for k,y in inst.y_vl_vr.items():
                if y.value == 1:
                    #src_coords = ips[metanode_vnr[k[0]].links[0][0][0]]
                    src_coords = metanode_vnr[k[0]].links[0][0][0]
                    vlink = (src_coords, k[1])    # Virtual link
                    slink = (k[2], k[3])    # Substrate link
                    mapping.setdefault(vlink, []).append(slink)
                    mapping_meta.setdefault((k[0], 0), []).append(slink)
                    # Do not count meta edges
                    if k[0] == k[2]:
                        continue
                    # How much quota allocated on the substrate?
                    quota = inst.req_l[k[0], k[1]]
                    for lnk in conflict_set(inst, k[2], k[3]):
                        sub_load[lnk] = sub_load.setdefault(lnk, 0) + quota
            #print sub_load
            # Compute the embedding cost now
            cost = 0
            link_quals = dict()
            for vnr, _map in mapping_meta.items():
                for lnk in _map:
                    # Do not count meta edges
                    if vnr[0] == lnk[0]: 
                        continue
                    quota = inst.req_l[vnr]
                    for l in conflict_set(inst, lnk[0], lnk[1]):
                        # The link weight is
                        weight = 40*sub_load[l] + 3*50 + 2*(100 - inst.rel[l]*100) + 781.25
                        link_quals[l] = 100-inst.rel[l]*100
                        cost += quota*weight
            #print link_quals
            # Get the resulting objective value of the integer program
            objective = value(inst.objective_capacity_priority)
            acceptance = len(vnr_list)/float(len(vnrs))
            logger.debug("Potential solution with acc %s and cost %s",
                                    str(acceptance), str(cost))
            # Is this the new best?
            if best_sol is None or (
                    best_sol['acceptance'] < acceptance
                    or cost < best_sol['objective']):
                best_sol = build_solution(cost, mapping, acceptance, start_time)
                logger.info("New solution: %s", best_sol)

            # This is a solution, no need to reduce the acceptance ratio more,
            # so eliminate the candidate
            solutions = solutions[1:]
            #return build_solution(cost, mapping, acceptance, start_time)
        else:                       # There is slack
            # Step 2 -- highest weight to the resources
            logger.info("Step 2")

            logger.debug("Capacity slack...",)
            logger.debug("%s", str(is_there_slack(inst.capacity_slack)))
            #print_slack(inst.capacity_slack)
            #print "Latency slack"
            #print is_there_slack(inst.latency_slack)
            #print_slack(inst.latency_slack)
            logger.debug("Reliability slack",)
            logger.debug("%s",str(is_there_slack(inst.reliability_slack)))
            print_slack(inst.reliability_slack)

            inst = model.create_instance(lp_input)
            # Set up the objectives
            inst.objective_reliability.deactivate()
            inst.objective_capacity_priority.deactivate()
            inst.objective_resource_priority.activate()
            _process_(solver, inst)
            cap_slack, lat_slack, rel_slack = get_slack(inst)
            slack = cap_slack + lat_slack + rel_slack
            # What if there's no slack - there must be, if not, error!
            if len(slack) == 0:
                logger.error("ERROR - no slack detected in step 2")
                return None

            # Which VN should be removed?
            cap_slack = get_link_cap_slack(inst, metanode_vnr)
            cap_sets_summary = []
            for l, vns in cap_slack.items():
                update_summary(cap_sets_summary, vns[1].keys(), (l, vns))

            if len(cap_sets_summary) == 0:
                # This means that there is no capacity slack, just reliability
                logger.debug("Cap sets summary is empty!")
                solutions = solutions[1:]
                continue

            #cap_sets_summary.sort(key=lambda x: x[1][1][0])
            cap_sets_summary.sort(key=lambda x: len(x[0]))
            logger.debug("Cap sets summary: %s", str(cap_sets_summary))

            # We start with the largest capacity slack and determine
            # the smallest size subsets that solve that.
            # From those, we find the subsets that can also solve the
            # capacity slack of the other nodes
            candidates = []
            while len(candidates) == 0 and init_size < len(vnr_list):
                for s in find_smallest_subsets(cap_sets_summary[-1][1], init_size):
                    # This bit is to avoid duplicates
                    if len(candidates) > 0:
                        exists = False
                        for c in zip(*candidates)[1]:
                            if equals(c,s):
                                exists = True
                                break
                        if exists: continue
                    rem_slack = 0
                    for l in cap_sets_summary[:-1]:
                        slack = test_remove_subset(l[1], s)
                        if slack < 0:
                            slack = 0
                        rem_slack += slack
                    candidates.append((rem_slack, s))
                init_size += 1

            if init_size == len(vnr_list):
                # We have gone through all possible subsets for this vnr list,
                # eliminate from possible solutions
                solutions = solutions[1:]
            else:
                # Update the start size for subset searching
                solutions[0] = (solutions[0][0], init_size)

            if len(candidates) == 0:
                continue

            # If we have any zero-slack candidates, use them
            tmp_c = filter(lambda x:x[0] == 0, candidates)
            if len(tmp_c) > 0:
                candidates = zip(*tmp_c)[1]
            else:
                candidates = [candidates[0][1]]
            logger.info("Candidates: %s", str(candidates))
            # Add all the candidates to the list of potential solutions
            for c in candidates:
                new_sol = remove_candidate_from(vnr_list, c)
                # Only add candidates if they have higher acceptance than crt best
                if (len(new_sol) > 0
                        and (best_sol is None
                            or len(best_sol['mapping'].items()) < len(new_sol))):
                    logger.info("Adding solution %s", str(new_sol))
                    solutions = [(new_sol, 1)] + solutions

#            # First retrieve only the source nodes
#            slack_vns = list(set(zip(*slack)[0]))
#            # Determine how much conflict each VN creates
#            conflict_amount = []
#            for meta_vn in slack_vns:
#                vn_load = metanode_vnr[meta_vn[0]].links[0][2]
#                conflict_amount.append((meta_vn[0], zip(*cap_slack)[0].count(meta_vn)*vn_load))
#                logger.debug("VN:%s Conflicts:%s Total:%s", str(metanode_vnr[meta_vn[0]]), str(zip(*cap_slack)[0].count(meta_vn)), str(zip(*cap_slack)[0].count(meta_vn)*vn_load))
#
#            # Sort the VNs that cause violation on the amount of resources taken
#            conflict_amount.sort(key=lambda x:x[1], reverse=True)
#            #slack_vns.sort(key=lambda x: len(vn_allocation(inst, x))*metanode_vnr[x[0]].links[0][2], reverse=True)
#            logger.debug("Conflict amount: %s", str(conflict_amount))
#            #logger.debug("%s", str(zip([metanode_vnr[x[0]] for x in slack_vns], map(lambda x:len(vn_allocation(inst, x)), slack_vns))))
#            # Eliminate the one with highest resource usage
#            highest = conflict_amount[0]
#            # Get the vnr with this source from the mapping
#            vn = metanode_vnr[highest[0]]
#            logger.debug("Highest resource usage with slack: %s", str(vn))
#            logger.debug("%s", str(conflict_amount[0]))
#            # Delete the vn from the list
#            #print "VN list before", vnr_list
#            vnr_list.remove(vn)
#            #print "VN list after", vnr_list
    if best_sol is not None:
        return best_sol
    else:
        return build_solution(None, None, 0, start_time)
#    if len(vnr_list) == 0:
#        return build_solution(None, None, 0, start_time)        # 0 acceptance ratio
#    return None


def solve_min_cost_fixed_acceptance(input_vector):
    solver = SolverFactory('cbc')

    lp_input = 'tmp_lp_input_%s_%d_%d.dat'%\
            (input_vector['nwksize'], input_vector['numvn'], input_vector['iteration'])
    nwk_gen = input_vector['substrate_coords']
    adj_matrix = input_vector['adjacency_matrix']
    ips = input_vector['interests']
    temp_proc = input_vector['temp_process']
    vnrs = input_vector['vnlist']
    vnr_list = list(vnrs)

    logger.info("Solving for vnrs %s", str(vnrs))

    # Generate the input based on the current vn list
    errcode, metanode_vnr = generate_lp_input(nwk_gen, adj_matrix, ips,
                                              temp_proc, vnr_list, lp_input)
    if errcode < 0:
        return None

    inst = model_min_cost.create_instance(lp_input)
    # Remove the lp input so we don't have pollution
    remfile(lp_input)

    try:
        stat = solver.solve(inst,options_string="seconds=600")
    except ApplicationError:
        # TODO handle error
        print "Application Error, no solution"
        return None

    # Extract the cost (objective)
    cost = value(inst.objective_cost)
    # Extract the mapping
    mapping = {}
    for k,y in inst.y_vl_vr.items():
        if y.value == 1:
            #src_coords = ips[metanode_vnr[k[0]].links[0][0][0]]
            src_coords = metanode_vnr[k[0]].links[0][0][0]
            vlink = (src_coords, k[1])    # Virtual link
            slink = (k[2], k[3])    # Substrate link
            mapping.setdefault(vlink, []).append(slink)
            # Do not count meta edges
            if k[0] == k[2]:
                continue

    return {
            'nwksize': input_vector['nwksize'],
            'numvn': input_vector['numvn'],
            'iteration': input_vector['iteration'],
            'mapping': mapping,
            'objective': cost
            }

from copy import deepcopy
from itertools import combinations
def find_best_cost_fixed_acceptance(input_vector, accepted):
    """
    Knowing that in input_vector we have the given number of
    accepted VNRs, find the best cost by going through all
    the combinations of k (accepted) out of n.
    """
    # Create a tmp IV
    tmp_iv = deepcopy(input_vector)
    accepted = int(accepted)
    tmp_iv['numvn'] = accepted
    input_vns = tmp_iv['vnlist']
    # Set a min score variable
    min_cost = None
    # Go through all the combinations (use itertools) of the input VNRs
    for vnlist in combinations(input_vns, accepted):
        print 'VNs',vnlist
        tmp_iv['vnlist'] = vnlist
        # If the score is improved, mark the result
        res = solve_min_cost_fixed_acceptance(tmp_iv)
        if res is None:
            print 'res is None'
            continue
        if len(res['mapping']) == accepted and\
                (min_cost is None or res['objective'] < min_cost['objective']):
            min_cost = res
            print 'Res=',res
    return min_cost

####################
## Obsolete
##
def solve_lp(input_vector):
    """Solve the VNE with an exact solution.
    Parameters
    input_vector    -- according to test plan
    """
    lp_input = 'tmp_lp_input_%s.dat'%input_vector['nwksize']

    solver = SolverFactory('cbc')
    solver.options['sec']=300

    nwk_gen = input_vector['substrate']
    vnrs = input_vector['vnlist']

    # Before starting, eliminate the VN reqs whose reliability can't be supported.
    # This is done as follows:
    # 1. Find the shortest hop path length, p, for the vlink
    # 2. Get the reliability of the p substrate links with highest reliability
    # 3. If the vlink reliability is higher than the product of the p links,
    #    the vlink cannot be mapped. Because any other paths will be longer and
    #    have lower reliability.
    #nx_graph = nx.from_edgelist([(src,dst)
    #                                for (src,dst,rel) in nwk_gen.substrate_edges])
    #substrate_rel = [rel for (src,dst,rel) in nwk_gen.substrate_edges]
    #substrate_rel.sort()
    #feasible_vns = []
    #for vn in vnrs:
    #    vn_src = vn.links[0][0]
    #    vn_dst = vn.links[0][1]
    #    vn_rel = vn.links[0][4]
    #    splen = nx.dijkstra_path_length(nx_graph, vn_src, vn_dst)
    #    max_rel = reduce(lambda x,y: x*y, substrate_rel[-splen:])
    #    if max_rel >= vn_rel:
    #        feasible_vns.append(vn)
    #    else:
    #        print "Eliminating", vn, "early"
    #vnrs = feasible_vns
    # Get time snapshot
    start_time = time()
    solutions = []
    k = len(vnrs)
    while len(solutions) == 0 and k > 0:
        #print "Eliminating",len(vnrs)-k, "vnrs"
        combs = combinations(vnrs, k)
        # Try subsets of the vnrs
        for vnr_list in combs:
            # Because we use a common sink, we can't have higher required
            # capacity than 1. We can therefore filter early failing solutions
            total_demand = sum(v.links[0][2] for v in vnr_list)
            if total_demand > 100:
                # This configuration has no solution, fail early
                continue
            #print "Trying to solve for ", vnr_list
            # Generate the lp input with this subset of vnrs
            if generate_lp_input(nwk_gen, vnr_list, lp_input) < 0:
                return None
            # Attempt to solve with this subset of vnrs
            inst = model.create_instance(lp_input)
            try:
                results = solver.solve(inst)
            except ApplicationError:
                # Solver failed. No solution, fail.
                continue
            term_cond = results.solver.termination_condition
            if term_cond == TerminationCondition.optimal:
                obj_value = value(inst.objective)
                solutions.append((vnr_list, obj_value))
                #print "Solution found with obj:", obj_value
            else:
                #print "No feasible solution"
                pass
        # If there was no feasible solution, try smaller vnr subsets
        k -= 1
    # Define solution parameters
    proc_time = None
    acceptance = None
    mapping = None
    objective = None
    if len(solutions) > 0:
        # We have a set of possible solutions
        # Sort according to objective sense:1=minimise, -1=maximise
        if inst.objective.sense == 1:
            solutions.sort(key=lambda x:x[1], reverse=True)
        else:
            solutions.sort(key=lambda x:x[1])
        # The best solution is now the last
        #print "Best solution is", solutions[-1]
        # Determine processing time
        proc_time = time() - start_time
        acceptance = float(k+1)/len(vnrs)
        # Retrieve mappings
        mapping = {}
        for k,y in inst.y_vl_vr.items():
            if y.value == 1:
                vlink = (k[1], k[2])
                slink = (k[3], k[4])
                if vlink in mapping:
                    mapping[vlink].append(slink)
                else:
                    mapping[vlink] = [slink]
        objective = solutions[-1][1]
    else:
        # This means no solution was feasible.
        #print "No solution found"
        proc_time = time() - start_time
        acceptance = 0
        objective = None
        mapping = None
    # Build and return solution
    solution = {'nwksize': input_vector['nwksize'],
                'numvn': input_vector['numvn'],
                'iteration': input_vector['iteration'],
                'proc_time': proc_time,
                'acceptance': acceptance,
                'mapping': mapping,
                'objective': objective
                }
    return solution

