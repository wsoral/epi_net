from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
import networkx as nx
import numpy as np
import random
import csv

HATER = 1
NO_HATER = 0

"""
The aim of this script is to generate a number of networks, following certain characteristics, and
running a simulation of hate speech outbreak, upon those networks. The network generating is dealt with
using the networkX library, whereas the interactions within the model - by MESA library.

Data will be collected in order to later determine, which network traits make the outbreak spread
quicker or slower.

The networks generated are according to following rules: 
Each Network is of same order N=1000
Each Network has similar (between 4 and 10) mean degree of nodes (((really? 4 and 10???)))
Each Network has to be connected

At this level, nets are generated by Barabasi-Albert algorithm in two versions:
- original
- with 'culling', by which I understand removing from the network those nodes,
which exceed a certain degree. At this moment, the threshold degree is set at 40, which
corresponds to 10% of the network's order. After removing, new nodes are added to compensate
for the drop in order.

The interactions between agents (represented by respectively, edges and nodes of our networks)
can be described as exposure to hate speech.
Each agent has his own tendency to utter hate speech. This can be understood as their contempt
towards a minority group, for example. The tendency corresponds to the probability of committing
an act of hate speech by a given agent, and is of range (0,1). Each time an agent's neighbour
uses hate speech, this agent's probability of using HS - raises by a small number.
This represents the diminishing of social norms, which inhibit us from using HS.

And the last element - an agent must have used HS himself in the past, or have witnessed their
neighbour using it, in order to be able to use it themself. 
"""

# Define a function for calculating the percentage of hating agents in each step.


def percent_haters(model):
    agent_behs = [agent.behavior for agent in model.schedule.agents]
    x = sum(agent_behs) / len(agent_behs)
    return x

def average_sensitivity(model):
    agent_sensitivity = [agent.sensitivity for agent in model.schedule.agents]
    x = np.mean(agent_sensitivity)
    return x


def net_avg_deg(model):
    x = model.avg_degree
    return x


def net_culling(model):
    x = model.big_nodes
    return x


def net_conn(model):
    x = model.connectivity
    return x


def net_clust(model):
    x = model.clustering
    return x


def final_step(model):
    x = model.step_counter
    return x


# Network generators
# Barabasi-Albert - number 1

def netgen_ba(n, m):
    I = nx.barabasi_albert_graph(n=n, m=m)
    degs = [I.degree[node] for node in I.nodes()]
    avg_deg = np.mean(degs)
    max_deg = np.max(degs)
    conn = nx.average_node_connectivity(I)
    clust = nx.average_clustering(I)
    return I, avg_deg, max_deg, conn, clust


# Erdos-Renyi - number 2
def netgen_er(n, p):
    I = nx.erdos_renyi_graph(n=n, p=p)
    degs = [I.degree[node] for node in I.nodes()]
    avg_deg = np.mean(degs)
    max_deg = np.max(degs)
    conn = nx.average_node_connectivity(I)
    clust = nx.average_clustering(I)
    return I, avg_deg, max_deg, conn, clust

# Random Regular - number 3
def netgen_rr(n, d):
    I = nx.random_regular_graph(d=d, n=n)
    degs = [I.degree[node] for node in I.nodes()]
    avg_deg = np.mean(degs)
    max_deg = np.max(degs)
    conn = nx.average_node_connectivity(I)
    clust = nx.average_clustering(I)
    return I, avg_deg, max_deg, conn, clust


# Create an empty dictionary for networks, and for each number in (1,2,3), generate a list of networks, and supp. data.
networks_for_use = {}
networks_for_use['1'] = []
for x in range(100):
    net, data = netgen_ba(100, 4)
    networks_for_use['1'].append((net, data))
    print(f'Finished generating network number {x+1} from set number 1.')

networks_for_use['2'] = []
for x in range(100):
    net, data = netgen_er(100, .078)
    networks_for_use['2'].apppend((net, data))
    print(f'Finished generating network number {x + 1} from set number 2.')

networks_for_use['3'] = []
for x in range(100):
    net, data = netgen_rr(100, 4)
    networks_for_use['3'].append((net, data))
    print(f'Finished generating network number {x + 1} from set number 3.')


class NormAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.behavior = self.random.choices([NO_HATER, HATER], weights=[9, 1])[0]
        self.contempt = self.random.betavariate(2, 5)
        self.sensitivity = self.random.betavariate(5,2)


    def step(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
        neigh_beh = [neigh.behavior for neigh in neighbors]

        self._nextBehavior = self.behavior
        self._nextSensitivity = self.sensitivity


        if (self.contempt > self.random.uniform(0,1)) and (self.sensitivity < 0.3):
            self._nextBehavior = HATER
        else: self._nextBehavior = NO_HATER

        if self.sensitivity > 0.1:
            self._nextSensitivity = self.sensitivity - np.mean(neigh_beh)*0.2


    def advance(self):
        self.behavior = self._nextBehavior
        self.sensitivity = self._nextSensitivity

class NormModel(Model):
    def __init__(self, size, set_no, subset_no):
        self.num_agents = size
        self.num_nodes = self.num_agents
        self.set = str(set_no)
        self.subset = subset_no
        self.I = networks_for_use[self.set][self.subset][0]
        self.type = str(self.subset)
        self.avg_deg = networks_for_use[self.set][self.subset][1]  # 1st parameter - average node degree
        self.big_nodes = networks_for_use[self.set][self.subset][2]  # 2nd parameter - degree of highest-degree node
        self.connectivity = networks_for_use[self.set][self.subset][3]  # 3rd parameter - mean network connectivity
        self.clustering = networks_for_use[self.set][self.subset][4]  # Network's clustering coeff.
        self.grid = NetworkGrid(self.I)
        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.step_counter = 1

        for i, node in enumerate(self.I.nodes()):
            a = NormAgent(i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, node)


        self.datacollector = DataCollector(
            model_reporters={"PerHate": percent_haters,
                             "AveSensitivity": average_sensitivity,
                             "MeanDeg": net_avg_deg,
                             "MaxDeg": net_culling,
                             "NetConnect": net_conn,
                             "NetClust": net_clust,
                             "FinalStep": final_step,
                             },
            agent_reporters={"Hate": "behavior"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.step_counter += 1
        if average_sensitivity(self) < 0.1 or self.step_counter > 250:
            self.running = False

fixed_params = {
    "size": 100,
}

variable_params = {
    "subset_no": np.arange(100),
    "set_no": np.arange(1,4),
}


batch_run = BatchRunner(
    NormModel,
    variable_params,
    fixed_params,
    iterations=2,
    max_steps=255,

    model_reporters={"PerHate": percent_haters,
                     "AveSensitivity": average_sensitivity,
                     "MeanDeg": net_avg_deg,
                     "MaxDeg": net_culling,
                     "NetConnect": net_conn,
                     "NetClust": net_clust,
                     "FinalStep": final_step,
                     },
    # agent_reporters={"Hate": "behavior",
    #                  "Step": "step_no"}
)

batch_run.run_all()

# agent_data = batch_run.get_agent_vars_dataframe()
run_data = batch_run.get_model_vars_dataframe()
run_data.to_csv('zbiorcze.csv')
# agent_data.to_csv('agentami.csv')


