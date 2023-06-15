from abc import ABC, abstractmethod


class KFoldDataModule(ABC):
    """This class is abstract class for typing

    *THIS IS NOT A REAL DATA MODULE*
    *THIS IS EMPTY CLASS*
    """

    pass


class KFoldDataModuleContainer(ABC):
    """This class is abstract class, don't create this object directly"""

    @abstractmethod
    def kfold_data_module(self, fold: int) -> KFoldDataModule:
        pass
