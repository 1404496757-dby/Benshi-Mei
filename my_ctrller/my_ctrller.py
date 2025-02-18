from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename('simglucose',
                                                'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

class None_Controller(Controller):
    def __int__(self):
        pass

    def policy(self, observation, reward, done, **info):
        return Action(basal=0, bolus=0)

    def reset(self):
        pass