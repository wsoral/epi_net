from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random

COUNTER_HATER = -1
NEUTRAL = 0
HATER = 1

def percent_haters(model):
    agent_behs = [agent.behavior for agent in model.schedule.agents]
    x = sum(agent_behs)/len(agent_behs)
    return x

def percent_hate_knowing(model):
    agent_knowledge = [agent.knows_hatered for agent in model.schedule.agents]
    x = np.mean(agent_knowledge)
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



the_network = netgen_dba(n=100  , m1 = 3, m2 = 4, p =.64, maxDeg=50, cull=False)
# the_network = nx.gnp_random_graph(n=1000, p=0.002)

class NormAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.behavior = self.random.choices([COUNTER_HATER, NEUTRAL, HATER], weights=[1,8,1])[0]
        self.contempt = self.random.betavariate(2,5)
        self.sensitivity = self.random.betavariate(5,2)

    def step(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
        neigh_beh = [neigh.behavior for neigh in neighbors]
        sum_hater = np.sum([x == 1 for x in neigh_beh])
        sum_counter = np.sum([x == -1 for x in neigh_beh])

        self._nextBehavior = self.behavior
        self._nextSensitivity = self.sensitivity

        if (self.contempt >= 0.7 and self.sensitivity <= 0.3):
            self._nextBehavior = HATER
        elif (self.contempt <= 0.3 and self.sensitivity >= 0.7):
            self._nextBehavior = COUNTER_HATER
        else:
            self._nextBehavior = NEUTRAL

        self._nextSensitivity = self.sensitivity + sum_hater*0.01 - sum_counter*0.01


    def advance(self):
        self.sensitivity = self._nextSensitivity
        self.behavior = self._nextBehavior



class NormModel(Model):
    def __init__(self, size):
        self.num_agents = size
        self.num_nodes = self.num_agents
        self.G = the_network[0]
        # self.G = the_network
        self.grid = NetworkGrid(self.G)
        self.schedule = SimultaneousActivation(self)
        self.running = True

        # list_of_random_nodes = self.random.sample(self.G.nodes(), self.num_agents)
        # for i in range(self.num_agents):
        #     a = NormAgent(i, self)
        #     self.grid.place_agent(a, list_of_random_nodes[i])
        #     self.schedule.add(a)

        for i, node in enumerate(self.G.nodes()):
            a = NormAgent(i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, node)


        self.datacollector = DataCollector(
            model_reporters={"PerHate": percent_haters,
                             "AveKnowing": percent_hate_knowing,
                             },
            agent_reporters={"Hate": "behavior"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if percent_haters(self) > 0.8:  # When the percentage of haters in the model exceeds 80,
            self.running = False  # the simulation is stopped, data collected, and next one is started.
