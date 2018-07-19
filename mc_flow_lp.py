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

#model.links_mapped = Var(model.E_r, domain=NonNegativeIntegers)
model.capacity_slack = Var(model.E_s, domain=NonNegativeIntegers)
model.latency_slack = Var(model.E_r, domain=NonNegativeIntegers)
model.reliability_slack = Var(model.E_r, domain=NonNegativeReals)


# Total traffic flowing through a link
def total_traffic(model, i,j):
    return sum(model.req_l[i_v,j_v] * model.y_vl_vr[i_v,j_v,i,j]
            for (i_v,j_v) in model.E_r)

# Traffic flowing through a link for a single VN
def vn_traffic(model, i, j, i_v, j_v):
    return model.req_l[i_v, j_v] * model.y_vl_vr[i_v, j_v, i, j]

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

def maximise_links_mapped(model):
    return (1./model.links)*sum(model.links_mapped[s,t] for (s,t) in model.E_r)\
            -sum(model.latency_slack[s,t] for (s,t) in model.E_r)\
            -sum(model.reliability_slack[s,t] for (s,t) in model.E_r)

def minimise_QoS_deviation(model):
    return sum(model.latency_slack[s,t] for (s,t) in model.E_r)\
        +sum(model.reliability_slack[s,t] for (s,t) in model.E_r)\
        +sum(model.capacity_slack[i,j] for (i,j) in model.E_s)

def capacity_priority_rule(model):
    weight_res = 10**(-6)
    weight_slack = 10
    return\
            weight_res * sum(total_traffic(model,i,j)*conflict_set(model,i,j) for i,j in model.E_real)\
            + weight_slack\
            *(sum(model.capacity_slack[i,j] for (i,j) in model.E_s)\
                #+sum(model.latency_slack[s,t] for (s,t) in model.E_r)\
                +sum(model.reliability_slack[s,t] for (s,t) in model.E_r))

def resource_priority_rule(model):
    weight_res = 1
    weight_slack = 0.1
    return\
            weight_res * sum(total_traffic(model,i,j)*conflict_set(model,i,j) for i,j in model.E_real)\
            + weight_slack\
            *(sum(model.capacity_slack[i,j] for (i,j) in model.E_s)\
                #+sum(model.latency_slack[s,t] for (s,t) in model.E_r)\
                +sum(model.reliability_slack[s,t] for (s,t) in model.E_r))

def feasible_reliability_rule(model):
    return sum(model.reliability_slack[s,t] for (s,t) in model.E_r)

model.objective_capacity_priority = Objective(rule=capacity_priority_rule, sense=minimize)
model.objective_resource_priority = Objective(rule=resource_priority_rule, sense=minimize)
# Objective to eliminate requests that cannot be installed due to reliability
model.objective_reliability = Objective(rule=feasible_reliability_rule, sense=minimize)

#### Constraints

#### Flow conservation rule
def flow_cons_rule(model, node, i_v, j_v):
    inbound = sum(vn_traffic(model, i, node, i_v, j_v)\
                        for i in model.V_s if (i,node) in model.E_s)
    outbound = sum(vn_traffic(model, node, i, i_v, j_v)\
                        for i in model.V_s if (node,i) in model.E_s)
    generated = model.req_l[i_v, j_v] if node == i_v else 0
    #generated = sum(vn_traffic(model, i_v, j, i_v, j_v)\
    #                    for j in model.V_s if (i_v, j) in model.E_s)\
    #                    if node == i_v else 0
    demanded = model.req_l[i_v, j_v] if node == j_v else 0
    #demanded = sum(vn_traffic(model, i, j_v, i_v, j_v)\
    #                    for i in model.V_s if (i, j_v) in model.E_s)\
    #                    if node == j_v else 0
    return inbound - outbound == demanded - generated

model.flowcons = Constraint(model.V_s*model.E_r, rule=flow_cons_rule)

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
    return conflict_capacity(model, a,b)\
            -model.capacity_slack[a,b]<= model.cap_l[a,b]
model.link_capacity_constraint = Constraint(model.E_real, rule=conflict_capacity_rule)

def max_capacity_rule(model, i, j):
    return sum(model.req_l[s,t]*model.y_vl_vr[s,t, i,j] for (s,t) in model.E_r)\
            -model.capacity_slack[i,j] <= model.cap_l[i,j]
#model.capacity = Constraint(model.E_s, rule=max_capacity_rule)

#### QoS constraints
def e2e_delay(model, i_v, j_v):
    return sum(model.y_vl_vr[i_v, j_v, i, j]*model.lat[i,j] for (i,j) in model.E_s)\
            - model.latency_slack[i_v, j_v] <= model.req_lat[i_v, j_v]
model.delay_constraint = Constraint(model.E_r, rule=e2e_delay)

def e2e_reliability(model, i_v, j_v):
    return sum(model.y_vl_vr[i_v, j_v, i, j]*model.log_rel[i,j] for (i,j) in model.E_s)\
            +model.reliability_slack[i_v, j_v] >= model.log_req_rel[i_v, j_v]
model.reliability_constraint = Constraint(model.E_r, rule=e2e_reliability)

#### Meta node constraints

# A VNR can use at most one substrate node, so there's only one connection bw
#   a meta node and a substrate node per VNR
def single_source_rule(model, i_v, j_v):
    return sum(model.y_vl_vr[i_v, j_v, i_v, k] for k in model.V_s if (i_v,k) in model.E_s) == 1
model.single_source_constraint = Constraint(model.E_r, rule=single_source_rule)

# Traffic can only flow _out_ of meta nodes, so no inbound traffic
# -- this is actually not needed because the augm graph is directional
#    with no inbound links to meta nodes
#def meta_traffic_rule(model, i_v, j_v):
#    return sum(total_traffic(model, i, i_v) for i in model.V_s if (i, i_v) in model.E_s) == 0
#model.meta_traffic_constraint = Constraint(model.E_r, rule=meta_traffic_rule)

#### Completeness constraints
def completeness_rule(model, i, j):
    return sum(model.y_vl_vr[i, j, i_s, j_s] for (i_s, j_s) in model.E_s) >= 1
def completeness_rule2(model, s, t):
    return sum(model.y_vl_vr[s, t, i_s, j_s] for (i_s, j_s) in model.E_s)\
            - model.links_mapped[s, t] >= 0
#model.completeness_constraint = Constraint(model.E_r, rule=completeness_rule2)

#### Flow length constraint
model.flow_length = Var(model.E_r)
def flow_length_rule(model, s, t):
    return sum(model.y_vl_vr[s, t, i_s, j_s] for (i_s, j_s) in model.E_s)\
            <= model.flow_length[s,t]
model.flow_length_constraint = Constraint(model.E_r, rule=flow_length_rule)


