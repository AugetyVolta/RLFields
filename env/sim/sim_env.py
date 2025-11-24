from abc import ABC, abstractmethod
import numpy as np

class SIM_ENV(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError("step method must be implemented by subclass.")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset method must be implemented by subclass.")