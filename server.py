from model import *
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.ModularVisualization import ModularServer

def network_portrayal(G):
    portrayal = dict()
    portrayal['nodes'] = [
        {"id": node_id,
         "size": agents[0].hate * 0.2,
         "color": "black" if agents[0].behavior == 1 else "#CCCCCC"}
        for (node_id, agents) in G.nodes.data('agent')
    ]

    portrayal['egdes'] = [
        {"source": source,
         "target": target,
         "color": "#000000",
         "width": .5
         } for (source, target) in enumerate(G.edges)
    ]

    return portrayal

network = NetworkModule(network_portrayal, 500, 500, library='sigma')


PerHate = ChartModule([{"Label": "PerHate",
                      "Color": "Black"}],
                    data_collector_name="datacollector")

chartHate = ChartModule([{"Label": "AveKnowing",
                      "Color": "Black"}],
                    data_collector_name="datacollector")

server = ModularServer(NormModel,
                       [network,
                        PerHate,
                        chartHate,
                        ],
                       "Hate Speech Model",
                       {"size": 100})
server.port = 8521