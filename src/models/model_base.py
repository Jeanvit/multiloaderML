"""
Author: Jean Vitor de Paulo
Date  : 12/10/21
Last updated: 
"""

from abc import ABC, abstractmethod, abstractproperty
from enum import Enum, auto

"""Supported libraries by now"""
class ModelType(Enum):
    KERAS      = auto()
    ONNX       = auto()
    TORCH      = auto()
    THEANO     = auto()
    MXNET      = auto()
    PYTORCH    = auto()
    SKLEARN    = auto()
    TENSORFLOW = auto()
    TFKERAS    = auto()
    CNTK       = auto()
    CAFFE      = auto()
    UNKNOWN    = auto()

class Model(ABC):
    """Abstract class  to rule all model implementation on the lib. Please make sure to always use it"""

    @property
    @abstractmethod
    def model_path(self) -> str:
        """The file or folder where the model lies on your device"""
        pass
    
    @model_path.setter
    @abstractmethod
    def model_path(self, new_model_path: str) -> None:
        return

    @property
    @abstractmethod
    def model_instance(self) -> any:
        """The model instance itself"""
        pass

    @model_instance.setter
    @abstractmethod
    def model_instance(self, new_model_instance: any) -> None:
        return

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """model state, if it went through all loading process"""
        pass
    
    @is_loaded.setter
    @abstractmethod
    def is_loaded(self, is_loaded: bool) -> None:
        return

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Just a simple name to better identify the model"""
        pass
    
    @model_name.setter
    @abstractmethod
    def model_name(self, new_model_name: str) -> None:
        return

    @abstractmethod
    def load(self) -> bool:
        """load a model using only its path and return if it was successfull"""
        pass
    
    @abstractmethod
    def reload(sef) -> bool:
        """reload the model and return if the process was sucessful"""
        pass
    
    @abstractmethod
    def preprocess_files_on_path(self) -> None:
        """do file manipulations to make the model loadable"""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """check if the model has a valid structure"""
        pass

    @abstractmethod
    def get_complexity_level(self) -> float:
        """complexity level measure how much computing power the model will need"""
        pass

    @abstractmethod
    def get_number_of_params(self) -> int:
        """get the total number of parameters of a given model"""
        pass
    
    @abstractmethod
    def get_number_of_layers(self) -> int:
        """count the number of layers of a model"""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> str:
        """get the input shape of a given model"""
        pass

    @abstractmethod
    def get_input(self) -> str:
        """get the input shape of a given model"""
        pass

    @abstractmethod
    def get_input_dimensionality(self,consider_batch=False) -> int:
        """get the input dimensions of a given model, subtractring -1 when considering batch"""
        pass

    @abstractmethod
    def get_output_shape(self) -> str:
        """get the output shape of a given model"""
        pass

    @abstractmethod
    def get_optimizer(model: any) -> str:
        """get the optimizer used to train the model"""
        pass

    @abstractmethod
    def get_output(self) -> str:
        """get the output a given model"""
        pass

    @abstractmethod
    def get_output_dimensionality(self, consider_batch=False) -> int:
        """get the output dimensions of a given model, subtractring -1 when considering batch"""
        pass

    @abstractmethod
    def get_single_layer_by_index(self, index) -> list:
        """get a layer of a given model considering the index of it"""
        pass

    @abstractmethod
    def get_type(self) -> ModelType:
        """get the model type"""
        pass