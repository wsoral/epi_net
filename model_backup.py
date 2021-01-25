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


def netgen_dba(n, m1, m2, p, cull, maxDeg):
    I = nx.dual_barabasi_albert_graph(n=n, m1=m1, m2=m2, p=p)  # Call the basic function
    degs = [I.degree[node] for node in list(I.nodes)]  # Calculate the node degree list for all nodes
    avg_deg = np.mean(degs)  # Calculate the mean node degree for the whole network.
    # conn = nx.average_node_connectivity(I)
    conn = 1
    if cull:  # Only if the network generator should remove supernodes.
        big_nodes = [node for node in list(I.nodes) if I.degree(node) >= maxDeg]  # Assigned SuperNode size.
        I.remove_nodes_from(big_nodes)
        for numerro in big_nodes:  # My very simple function for removing SuperNodes.
            I.add_node(numerro)
            eN = random.choice([m1, m2])
            for _ in range(eN):
                to = random.choice(list(I.nodes()))
                I.add_edge(numerro, to)
        degs = [I.degree[node] for node in list(I.nodes)]
        avg_deg = np.mean(degs)  # This whole loop should be a function, but method will not produce any new SuperNodes
        # conn = nx.average_node_connectivity(I)
        conn = 1
    if not (nx.is_connected(I) and (4 <= avg_deg <= 10)):  # Test if network within parameters
        return netgen_dba(n=n, m1=m1, m2=m2, p=p, maxDeg=maxDeg, cull=cull)  # If not within parameters, call self.
    else:
        print(f"Successfully generated naetwork with parameters {m1, m2, p, cull, maxDeg}\n")
        return I, avg_deg, cull, maxDeg



the_network = netgen_dba(n=100, m1 = 3, m2 = 4, p =.64, maxDeg=50, cull=False)
another_network = nx.random_regular_graph(7, 100)

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
    def __init__(self, size):
        self.num_agents = size
        self.num_nodes = self.num_agents
        # self.G = the_network[0]
        self.G = another_network
        self.grid = NetworkGrid(self.G)
        self.schedule = SimultaneousActivation(self)
        self.running = True

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
        if percent_haters(self) > 0.8:  # When the percentage of haters in the model exceeds 80,
            self.running = False  # the simulation is stopped, data collected, and next one is started.
