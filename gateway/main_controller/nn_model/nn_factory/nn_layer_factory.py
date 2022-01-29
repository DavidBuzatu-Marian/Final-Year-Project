from abc import ABC, abstractmethod


class NNAbstractLayerFactory(ABC):
    @abstractmethod
    def get_layer(self, layer_type, parameters):
        pass
