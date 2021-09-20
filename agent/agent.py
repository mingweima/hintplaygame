"""
Agent class (card game player)
All player-agents should inherit from this class
"""

import abc

class Agent(abc.ABC):
    @abc.abstractmethod
    def select_action(self, obs, **kwargs):


