from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random

# TODO Instead of double Barabasi-Albert generator, use regular Barabasi - Albert. with m=7. (?)
# TODO Additionally, Use one (?) of the following generators: a) random regular graph generator, b) random graph gen
# TODO In the server.py file, add possibility for user, to select the kind of network they want to use.
# TODO in server.py, make sure the current network's parameters are displayed.

HATER = 1
NO_HATER = 0

def percent_haters(model):
    agent_behs = [agent.behavior for agent in model.schedule.agents]
    x = sum(agent_behs)/len(agent_behs)
    return x


def average_sensitivity(model):
    agent_sensitivity = [agent.sensitivity for agent in model.schedule.agents]
    x = np.mean(agent_sensitivity)
    return x


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


class NormAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.behavior = self.random.choices([NO_HATER, HATER], weights=[9,1])[0]
        self.contempt = self.random.betavariate(2,5)
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
        # self.contempt = self._nextContempt
        self.behavior = self._nextBehavior
        self.sensitivity = self._nextSensitivity


class NormModel(Model):
    def __init__(self, size, net_type):
        self.num_agents = size
        self.num_nodes = self.num_agents
        self.type = net_type
        if self.type == 1:
            self.G, self.avg_degree, self.big_nodes, self.connectivity, self.clustering = netgen_ba(self.num_agents, 4)
        if self.type == 2:
            self.G, self.avg_degree, self.big_nodes, self.connectivity, self.clustering = netgen_er(self.num_agents, .078)
        if self.type == 3:
            self.G, self.avg_degree, self.big_nodes, self.connectivity, self.clustering = netgen_rr(self.num_agents, 4)
        self.grid = NetworkGrid(self.G)
        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.step_counter = 1
        for i, node in enumerate(self.G.nodes()):
            a = NormAgent(i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, node)


        self.datacollector = DataCollector(
            model_reporters={"PerHate": percent_haters,
                             "AverageSens": average_sensitivity,
                             },
            agent_reporters={"Hate": "behavior"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.step_counter += 1
        if percent_haters(self) > 0.35 or self.step_counter > 250:  # When the percentage of haters in the model exceeds 80,
            self.running = False  # the simulation is stopped, data collected, and next one is started.
        # Alternatively:
        if average_sensitivity(self) < 0.1 or self.step_counter > 250:
            self.running = False
