from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random

HATER = 1
NO_HATER = 0

def percent_haters(model):
    agent_behs = [agent.behavior for agent in model.schedule.agents]
    x = sum(agent_behs)/len(agent_behs)
    return x

def average_hate(model):
    agent_hate = [agent.hate for agent in model.schedule.agents]
    x = sum(agent_hate)/len(agent_hate)
    return x

# def netgen_dba(n=400, m1=4, m2=3, p=.64):
#     I = nx.dual_barabasi_albert_graph(n=n, m1=m1, m2=m2, p=p)
#     degs = [I.degree[node] for node in list(I.nodes)]
#     avg_deg = np.mean(degs)
#     avg_clust = nx.average_clustering(I)
#     connectivity = nx.node_connectivity(I)
#
#     return [I, avg_clust, connectivity]

def netgen_dba(n=1000, m1=4, m2=3, p=.55):
    I = nx.dual_barabasi_albert_graph(n=n, m1=m1, m2=m2, p=p)
    degs = [I.degree[node] for node in list(I.nodes)]
    avg_deg = np.mean(degs)
    while avg_deg > 7 or avg_deg <6:
        print(f"Wrong mean degree number. Getting {avg_deg}. Rerunning.")
        I = nx.dual_barabasi_albert_graph(n=n, m1=m1, m2=m2, p=p)
        degs = [I.degree[node] for node in list(I.nodes)]
        avg_deg = np.mean(degs)
    big_nodes = [node for node in list(I.nodes) if I.degree(node) >= 40]
    node_size = [I.degree(node) for node in big_nodes]
    return I

# def netgen_dba(n=1000, m1=4, m2=3, p=.64, maxDeg=40):
#     I = nx.dual_barabasi_albert_graph(n=n, m1=m1, m2=m2, p=p)
#     degs = [I.degree[node] for node in list(I.nodes)]
#     avg_deg = np.mean(degs)
#     big_nodes = [node for node in list(I.nodes) if I.degree(node) >= maxDeg]
#     node_size = [I.degree(node) for node in big_nodes]
#     I.remove_nodes_from(big_nodes)
#     for numerro in big_nodes:
#         I.add_node(numerro)
#         eN = random.choice([m1, m2])
#         for _ in range(eN):
#             to = random.choice(list(I.nodes()))
#             I.add_edge(numerro, to)
#     new_degs = [I.degree[node] for node in list(I.nodes)]
#     new_avg = np.mean(new_degs)
#     while new_avg > 7 or new_avg < 6:
#         I = nx.dual_barabasi_albert_graph(n=n, m1=m1, m2=m2, p=p)
#         degs = [I.degree[node] for node in list(I.nodes)]
#         avg_deg = np.mean(degs)
#         big_nodes = [node for node in list(I.nodes) if I.degree(node) >= maxDeg]
#         node_size = [I.degree(node) for node in big_nodes]
#         I.remove_nodes_from(big_nodes)
#         for numerro in big_nodes:
#             I.add_node(numerro)
#             eN = random.choice([m1, m2])
#             for _ in range(eN):
#                 to = random.choice(list(I.nodes()))
#                 I.add_edge(numerro, to)
#         new_degs = [I.degree[node] for node in list(I.nodes)]
#         new_avg = np.mean(new_degs)
#
#     return I


class NormAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.behavior = self.random.choices([NO_HATER, HATER], weights=[9,1])[0]
        self.hate = self.random.betavariate(2,5)
        self.knows_hatered = 0

    def step(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
        neigh_beh = [neigh.behavior for neigh in neighbors]

        self._nextBehavior = self.behavior
        self._nextHate = self.hate

        if (HATER in [neigh_beh] or self.behavior == HATER):
            self.knows_hatered = 1

        if (self.hate > self.random.uniform(0,1)) and (self.knows_hatered == 1):
            self._nextBehavior = HATER
        else: self._nextBehavior = NO_HATER

        if self.hate < 0.8:
            self._nextHate = self.hate + sum(neigh_beh)*0.01

    def advance(self):
        self.hate = self._nextHate
        self.behavior = self._nextBehavior


class NormModel(Model):
    def __init__(self, size):
        self.num_agents = size
        self.num_nodes = self.num_agents
        self.G = netgen_dba()
        self.grid = NetworkGrid(self.G)
        self.schedule = SimultaneousActivation(self)
        self.running = True

        i = 0
        list_of_random_nodes = self.random.sample(self.G.nodes(), self.num_agents)
        for i in range(self.num_agents):
            a = NormAgent(i, self)
            self.grid.place_agent(a, list_of_random_nodes[i])
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={"PerHate": percent_haters,
                             "AveHate": average_hate,
                            },
            agent_reporters={"Hate": "behavior"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()