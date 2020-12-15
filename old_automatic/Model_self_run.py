from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from numpy import corrcoef, arange
from mesa.datacollection import DataCollector
import random
import pandas
from mesa.batchrunner import BatchRunner

HATER = 1
NO_HATER = 0
CONTEMPT_A = 2
CONTEMPT_B = 5
HSSENSI_A = 5
HSSENSI_B = 1


def percent_haters(model):
    agent_behs = [agent.behavior for agent in model.schedule.agents]
    x = sum(agent_behs)/len(agent_behs)
    return x


def average_sens(model):
    agent_sens = [agent.hs_sensi for agent in model.schedule.agents]
    x = sum(agent_sens)/len(agent_sens)
    return x


def average_contempt(model):
    agent_cont = [agent.contempt for agent in model.schedule.agents]
    x = sum(agent_cont)/len(agent_cont)
    return x


def cor_hate_cont(model):
    agent_behs = [agent.behavior for agent in model.schedule.agents]
    agent_cont = [agent.contempt for agent in model.schedule.agents]
    return corrcoef(agent_behs, agent_cont)[1, 0]


class HateAgent(Agent):
    # Inicjujemy agenta, przypisując mu miejsce na planszy i typ.
    def __init__(self, unique_id, model, behavior):
        super().__init__(unique_id, model)
        # początkowy poziom pogardy
        self.contempt = self.random.betavariate(CONTEMPT_A, CONTEMPT_B)
        # początkowy poziom wrażliwości na mowę nienawiści
        self.hs_sensi = self.random.betavariate(HSSENSI_A, HSSENSI_B)
        # Początkowe zachowanie
        self.behavior = behavior
        # poziom potrzeby poznania
        self.needForCog = self.random.gauss(0.5, 0.15)
        self.sociability = self.model.edges
        self.unique_id = unique_id
        self.step_no = [1]
        self.zachowanie = []

    def step(self):
        similar = 0

        a = self.sociability
        neighbors_coord = []
        x, y = self.pos

        self.step_no.append(self.step_no[-1] + 1)
        self.zachowanie.append(self.behavior)

        for dx in list(range(-a, a+1)):
            for dy in list(range(-a, a+1)):
                if 0 < x+dx < self.model.height and\
                 0 < y+dy < self.model.width:
                    neighbors_coord.append((x+dx, y+dy))
        neighbors = self.model.grid.get_cell_list_contents(neighbors_coord)
        neigh_beh = [neigh.behavior for neigh in neighbors]
        self._nextBehavior = self.behavior
        for neighbor in neighbors:
            if neighbor.behavior == self.behavior:
                similar += 1

        # Zachodzi faux pas, nie następuje zarażanie hate'em i agent hejtujący
        # zwiększa swoją wrażliwość (aka normę)

        if self.model.faux_pas == 1 and self._nextBehavior == HATER and\
                                        similar == 0:
            self._next_hs_sensi = self.hs_sensi
            self._next_contempt = self.contempt
            self._next_hs_sensi += 0.005 * len(neighbors)
            if self.contempt > self.random.uniform(0, 1):
                if sum(neigh_beh) > 3:
                    self._nextBehavior = HATER
                else:
                    if self.hs_sensi > self.random.uniform(0, 1):
                        self._nextBehavior = NO_HATER
                    else:
                        self._nextBehavior = HATER
            else:
                self._nextBehavior = NO_HATER
        else:
            # Nie zachodzi faux pas, wszystko przebiega normalnie
            if self.contempt > self.random.uniform(0, 1):
                if sum(neigh_beh) > 3:
                    self._nextBehavior = HATER
                else:
                    if self.hs_sensi > self.random.uniform(0, 1):
                        self._nextBehavior = NO_HATER
                    else:
                        self._nextBehavior = HATER
            else:
                self._nextBehavior = NO_HATER

            self._next_hs_sensi = self.hs_sensi
            if self.hs_sensi > 0.08:
                self._next_hs_sensi -= 0.005*sum(neigh_beh)

            self._next_contempt = self.contempt
            if self.contempt < 0.92:
                self._next_contempt += 0.001*sum(neigh_beh)

            # Jeśli liczba podobnych sąsiadów jest mniejsza,
            # niż sobie zdefiniowaliśmy,
            # przenosimy się nalosowe inne pole
        if similar < self.model.homophily or self.random.uniform(0, 1) < 0.1:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1

    def advance(self):
        self.behavior = self._nextBehavior
        self.hs_sensi = self._next_hs_sensi
        self.contempt = self._next_contempt


class Hate(Model):
    def __init__(self, height, width, density, hater_pc, homophily, faux_pas,
                 edges):
        self.height = height
        self.width = width
        self.density = density
        self.hater_pc = hater_pc
        self.homophily = homophily
        self.grid = SingleGrid(height, width, torus=True)
        self.schedule = SimultaneousActivation(self)
        # self.schedule = RandomActivation(self)
        self.happy = 0
        self.faux_pas = faux_pas
        self.edges = edges

        self.datacollector = DataCollector(
            model_reporters={"PerHate": percent_haters,
                             "AveSens": average_sens,
                             "AveCont": average_contempt,
                             "CorHatCon": cor_hate_cont,
                             "Density": density},
            agent_reporters={"Hate": "behavior",
                             "Step": "step_no",
                             "Hate": "zachowanie"}
        )

        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            if self.random.random() < self.density:
                if self.random.random() < self.hater_pc:
                    behavior = 1
                else:
                    behavior = 0

                agent = HateAgent((x, y), self, behavior)
                self.grid.position_agent(agent, (x, y))
                self.schedule.add(agent)

        self.running = True
        # self.datacollector.collect(self)

    def step(self):
        self.happy = 0
        self.schedule.step()
        if percent_haters(self) > 0.9:
            self.running = False


fixed_params = {
    "height": 20,
    "width": 20
    # "density": 0.5,
    # "hater_pc": 0.5
}

variable_params = {
    "density": arange(0.25, 1.0, 0.25),
    "hater_pc": arange(0.2, 1.0, 0.20),
    "homophily": arange(0, 8, 2),
    "faux_pas": [0, 1],
    "edges": [1, 2]
}

batch_run = BatchRunner(
    Hate,
    variable_params,
    fixed_params,
    iterations=2,
    max_steps=150,
    model_reporters={"PerHate": percent_haters,
                     "AveSens": average_sens,
                     "AveCont": average_contempt,
                     "CorHatCon": cor_hate_cont
                     },
    agent_reporters={"Hate": "behavior",
                     "Sociability": "sociability",
                     "Step": "step_no"}
)

batch_run.run_all()

agent_data = batch_run.get_agent_vars_dataframe()
run_data = batch_run.get_model_vars_dataframe()
run_data.to_csv('zbiorcze.csv')
agent_data.to_csv('agentami.csv')
