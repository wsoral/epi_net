from model import *
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer

def network_portrayal(G):
    portrayal = dict()
    portrayal['nodes'] = [
        {"id": node_id,
         "size": agents[0].sensitivity * 0.2,
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

network = NetworkModule(network_portrayal, 800, 800, library='sigma')


PerHate = ChartModule([{"Label": "PerHate",
                      "Color": "Black"}],
                    data_collector_name="datacollector")

AveSens = ChartModule([{"Label": "AverageSens",
                        "Color": "Black"}],
                      data_collector_name="datacollector")

class MyTextElement(TextElement):
    def render(self, model):
        ad = model.avg_degree
        md = model.big_nodes
        ncn = model.connectivity
        ncl = model.clustering
        if model.type == 1:
            nt = "Barabasi - Albert"
        if model.type == 2:
            nt = "Erdos - Renyi"
        if model.type == 3:
            nt = "Random Regular"

        return f"Network type: {nt}<br>" \
               f"Average node degree: {ad}<br>" \
               f"Maximum degree of a node: {md}<br>" \
               f"Network's connectivity: {ncn}<br>" \
               f"Network's clustering coefficient: {ncl}<br>" \
               f"Step number: {model.step_counter}"

model_params = {
    "size": 100,
    "net_type": UserSettableParameter(
        "slider",
        "Type of network",
        1,
        1,
        3,
        1,
        description="Choose the type of network generator you want to use."
    )
}
server = ModularServer(NormModel,
                       [network,
                        MyTextElement(),
                        PerHate,
                        AveSens,
                        ],
                       "Hate Speech Model",
                       model_params)
server.port = 8521
