from model import *
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import ModularServer

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "black",
                 "r": agent.contempt}
    if agent.behavior == 1:
        portrayal["Color"] = "black"
    else:
        portrayal["Color"] = "gray"

    return portrayal


grid = CanvasGrid(agent_portrayal, 40, 40, 800, 800)

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
                       {"width": 40, "height": 40})
server.port = 8521
