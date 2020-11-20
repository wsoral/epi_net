from old_model.model import *
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.ModularVisualization import ModularServer



def network_portrayal(G):

    portrayal = dict()
    portrayal['nodes'] = [
        {"id": node_id,
         "size": agents[0].contempt,
         "color": "black" if agents[0].behavior == 1 else "#999999"}
        for (node_id, agents) in G.nodes.data('agent')
    ]

    portrayal['egdes'] = [
        {"id": edge_id,
         "source": source,
         "target": target,
         "color": "#111111"
         } for edge_id, (source, target) in enumerate(G.edges)
    ]

    return portrayal


grid = NetworkModule(network_portrayal, 750, 750, library='sigma')
# grid = CanvasGrid(agent_portrayal, 40, 40, 800, 800)

chartHate = ChartModule([{"Label": "PerHate",
                      "Color": "Black"}],
                    data_collector_name="datacollector")

chartSens = ChartModule([{"Label": "AveSens",
                      "Color": "Black"}],
                    data_collector_name="datacollector")

chartCont = ChartModule([{"Label": "AveCont",
                      "Color": "Black"}],
                    data_collector_name="datacollector")

chartCorr = ChartModule([{"Label": "CorHatCon",
                      "Color": "Black"}],
                    data_collector_name="datacollector")

server = ModularServer(HateModel,
                       [grid, chartHate, chartSens, chartCont, chartCorr],
                       "Hate Speech Model",
                       {"width": 20, "height": 20})
server.port = 8521
