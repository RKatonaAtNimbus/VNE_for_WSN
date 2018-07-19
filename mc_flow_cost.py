from pyomo.environ import *
from pyomo.opt import SolverFactory
from math import ceil

model = AbstractModel()

model.epsilon = 0.01
#### Substrate network

# Nodes - including meta nodes
model.V_s = Set()
# Edges - including meta links
model.E_s = Set(within=model.V_s*model.V_s)

def get_num_links(model):
    return len(model.E_s)
model.links = Param(within=NonNegativeIntegers, initialize=get_num_links)

# Link capacity
model.cap_l = Param(model.E_s)
# Link latency
model.lat = Param(model.E_s)
# Link reliability
model.rel = Param(model.E_s)

def derive_log_reliability(model, i, j):
    return log(model.rel[i,j])

model.log_rel = Param(model.E_s, initialize=derive_log_reliability)

#### Flow source dest

# The sources are meta nodes - interest points
model.E_r = Set(within=model.V_s*model.V_s)

# Link demand
model.req_l = Param(model.E_r)
# E2E required latency
model.req_lat = Param(model.E_r)
# E2E reliability
model.req_rel = Param(model.E_r)

def derive_log_req_reliability(model, i, j):
    return log(model.req_rel[i,j])

model.log_req_rel = Param(model.E_r, initialize=derive_log_req_reliability)

def is_real_edge(model, i, j):
    """Function to filter out meta-edges"""
    # A meta edge starts at a VNR source. So if the edge (i,0) is in E_r, it's
    # a meta edge.
    if (i,0) in model.E_r: return False
    else: return True
model.E_real = Set(initialize=model.E_s, filter=is_real_edge)

#### Variables

# Mapping of virtual link to substrate link
# y:E_r x E_s -> {0,1}, y(s,t, u,v) -> indicates if s,t is mapped on u,v
model.y_vl_vr = Var(model.E_r, model.E_s, domain=Boolean, initialize=0)

# Total traffic flowing through a link
def total_traffic(model, i,j):
    return sum(model.req_l[i_v,j_v] * model.y_vl_vr[i_v,j_v,i,j]
            for (i_v,j_v) in model.E_r)

# Traffic flowing through a link for a single VN
def vn_traffic(model, i, j, i_v, j_v):
    return model.req_l[i_v, j_v] * model.y_vl_vr[i_v, j_v, i, j]

# Traffic generated at a node
def generated_traffic(model, node):
    # Reflects the VNRs that are mapped to the node
    # Inner represents all VNRs starting at node (different sinks)
    # Outer represents all outgoing links of node, in case 
    # VNRs are mapped to different links.
    gen = []
    for (i_v,j_v) in model.E_r:
        if i_v == node:
            for (i,j) in model.E_s:
                if i == node:
                    gen.append(vn_traffic(model, node,j,node,j_v))
    return sum(gen)
            
    #return sum(vn_traffic(model, node, j, node, j_v)\
    #        for j_v in model.V_s if (node, j_v) in model.E_r
    #                for (node, j_v) in model.E_r for (node, j) in model.E_s)

# Traffic incoming into a node
def demanded_traffic(model, node):
    gen = []
    for (i_v,j_v) in model.E_r:
        if j_v == node:
            for (i,j) in model.E_s:
                if j == node:
                    gen.append(vn_traffic(model, i,node,i_v,node))
    return sum(gen)
    #return sum(vn_traffic(model, j, node, j_v, node)\
    #                for (j_v, node) in model.E_r for (j, node) in model.E_s)


#### Objective
##   Minimize the cost of the allocation

##   Maximise the traffic allocated over the substrate
def substrate_traffic_rule(model):
    return sum(total_traffic(model, i, j) for (i,j) in model.E_s)

##   Minimise the cost while maximising accepted resources
##   First calculate the number of links in the conflict set
def conflict_set(model, a, b):
    conflict_set_size = 1 \
        +sum(1 for (i,k) in model.E_real if i == b)\
        +sum(1 for (k,l) in model.E_real if (k,b) in model.E_real and k != a and l!= a)\
        +sum(1 for (j,i) in model.E_real if j == a and i != b)\
        +sum(1 for (i,j) in model.E_real if j == a and i != b)\
        +sum(1 for (k,l) in model.E_real if (a,l) in model.E_real and\
                        k != a and k != b and l != b and (k,b) not in model.E_real)
    return conflict_set_size

def roland_cost_with_conflict_rule(model):
    cost = sum(total_traffic(model,i,j)*conflict_set(model,i,j)*(100-model.rel[i,j]*100) for i,j in model.E_s)
    return cost

def embedding_cost(model):
    cost = sum(40*total_traffic(model,i,j)*conflict_set(model,i,j) + 3*50 + 2*(100-model.rel[i,j]*100) + 781 for i,j in model.E_real)
    return cost

model.objective_cost = Objective(rule=embedding_cost, sense=minimize)

#### Constraints

#### Single path constraint
# For each VNR there is a single, unsplittable path from source to sink.
# This means that for every combination snode, VNR there is at most one
# inbound link and one outbound link.
def single_path_inbound_rule(model, node, i_v, j_v):
    in_tr = [model.y_vl_vr[i_v, j_v, i, node] for i in model.V_s if (i,node) in model.E_s]
    if len(in_tr) == 0:
        return Constraint.Skip
    else:
        return sum(in_tr) <= 1
model.single_path_in_constraint = Constraint(model.V_s*model.E_r,\
                                    rule=single_path_inbound_rule)
def single_path_outbound_rule(model, node, i_v, j_v):
    out_tr = [model.y_vl_vr[i_v, j_v, node, i] for i in model.V_s if (node,i) in model.E_s]
    if len(out_tr) == 0:
        return Constraint.Skip
    else:
        return sum(out_tr) <= 1
model.single_path_out_constraint = Constraint(model.V_s*model.E_r,\
                                    rule=single_path_outbound_rule)

#### Flow conservation rule
def flow_cons_rule(model, node):
    inbound = sum(total_traffic(model, i, node) for i in model.V_s if (i,node) in model.E_s)
    outbound = sum(total_traffic(model, node, i) for i in model.V_s if (node,i) in model.E_s)
    generated = generated_traffic(model, node)
    demanded = demanded_traffic(model, node)
    return inbound - outbound == demanded - generated
model.flowcons = Constraint(model.V_s, rule=flow_cons_rule)

#### Capacity constraint
# This is applied only on the real edges, not the meta edges
def conflict_capacity(model, a, b):
    return total_traffic(model, a, b)\
            +sum(total_traffic(model, i, k) for (i,k) in model.E_real if i == b)\
            +sum(total_traffic(model, k, l) for (k,l) in model.E_real\
                            if (k,b) in model.E_real and k != a and l!= a)\
            +sum(total_traffic(model, j, i) for (j,i) in model.E_real\
                            if j == a and i != b)\
            +sum(total_traffic(model, i, j) for (i,j) in model.E_real\
                            if j == a and i != b)\
            +sum(total_traffic(model, k, l) for (k,l) in model.E_real\
                            if (a,l) in model.E_real\
                                and k != a\
                                and k != b\
                                and l != b\
                                and (k,b) not in model.E_real)

def conflict_capacity_rule(model, a, b):
    return conflict_capacity(model, a,b) <= model.cap_l[a,b]
model.link_capacity_constraint = Constraint(model.E_real, rule=conflict_capacity_rule)

#### QoS constraints
def e2e_delay(model, i_v, j_v):
    return sum(model.y_vl_vr[i_v, j_v, i, j]*model.lat[i,j] for (i,j) in model.E_s)\
            <= model.req_lat[i_v, j_v]
model.delay_constraint = Constraint(model.E_r, rule=e2e_delay)

def e2e_reliability(model, i_v, j_v):
    return sum(model.y_vl_vr[i_v, j_v, i, j]*model.log_rel[i,j] for (i,j) in model.E_s)\
            >= model.log_req_rel[i_v, j_v]
model.reliability_constraint = Constraint(model.E_r, rule=e2e_reliability)

#### Meta node constraints

# A VNR can use at most one substrate node, so there's only one connection bw
#   a meta node and a substrate node per VNR
def single_source_rule(model, i_v, j_v):
    return sum(model.y_vl_vr[i_v, j_v, i_v, k] for k in model.V_s if (i_v,k) in model.E_s) == 1
model.single_source_constraint = Constraint(model.E_r, rule=single_source_rule)

#### Completeness constraint

# At least one mapping for each VNR
def completeness_rule(model, i, j):
    return sum(model.y_vl_vr[i, j, i_s, j_s] for (i_s, j_s) in model.E_s) >= 1
model.completeness_constraint = Constraint(model.E_r, rule=completeness_rule)

#### Flow length constraint
#model.flow_length = Var(model.E_r)
def flow_length_rule(model, s, t):
    return sum(model.y_vl_vr[s, t, i_s, j_s] for (i_s, j_s) in model.E_s)\
            <= model.flow_length[s,t]
#model.flow_length_constraint = Constraint(model.E_r, rule=flow_length_rule)


