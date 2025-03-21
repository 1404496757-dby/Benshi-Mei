# from .user_interface import simulate
from .base import Controller, Action


class RECCo_Controller(Controller):
    def __init__(self, insulin=0):
        self.insulin = insulin  # init_state
        # self.state = init_state
        # super().__init__(init_state)

    def policy(self, observation, reward, done, **info):
        # self.state = observation
        action = Action(basal=self.insulin, bolus=0)
        return action

    def update_insulin(self, get_insulin):
        self.insulin = get_insulin

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        # self.state = self.init_state
        pass

# ctrller = MyController(0)
# simulate(controller=ctrller)
