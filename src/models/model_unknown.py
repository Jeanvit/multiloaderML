"""
Author: Jean Vitor de Paulo
Date  : 15/10/21
Last updated: 
"""

import logging
from models.model_base import Model, ModelType

class UnknownModel(Model):
    """This is the class for handling tf.keras models behaviour and properties"""
    model_path: str
    model_name: str
    is_loaded: bool
    model_instance: any

    def __init__(self, path: str, name: str, load_on_start: bool = True) -> None:
        logging.warning("An unkown model has been instanced, setting everything to 0")
        self.model_path = path
        self.model_name = name
        self.is_loaded = False

    """start of getters and setters"""

    def model_path(self) -> str:
        return self.model_path

    def model_path(self, new_model_path: str) -> None:
        self.model_path = new_model_path

    def model_name(self) -> str:
        return self.model_name

    def model_name(self, new_model_name: str) -> None:
        self.model_name = new_model_name

    def model_instance(self) -> any:
        return self.model_instance

    def model_instance(self, new_model_instance: any) -> None:
        self.model_instance = new_model_instance

    def is_loaded(self) -> bool:
        return self.is_loaded

    def is_loaded(self, is_loaded) -> None:
        self.is_loaded = is_loaded
    """end of getters and setters"""

    def load(self, forced_new_istance=None) -> bool:
        return False

    def is_valid(self) -> bool:
        """check if the model has a valid structure"""
        return False

    def get_number_of_layers(self) -> int:
        return 0

    def get_complexity_level(self) -> float:
        return 0

    def get_model_size(self) -> float:
        return 0

    def get_number_of_params(self) -> int:
        """get the total number of parameters of a given model"""
        return 0

    def get_input_shape(self) -> str:
        """get the input shape of a given model"""
        return 0

    def get_input(self) -> str:
        """get the input shape of a given model"""
        return 0

    def get_input_dimensionality(self, consider_batch=False) -> int:
        """get the input dimensions of a given model, subtractring -1 when considering batch"""
        return 0

    def get_output_shape(self) -> str:
        """get the output shape of a given model"""
        return 0

    def get_output(self) -> str:
        """get the output a given model"""
        return 0

    def get_output_dimensionality(self, consider_batch=False) -> int:
        """get the output dimensions of a given model, subtractring -1 when considering batch"""
        return 0

    def get_single_layer_by_index(self, index: int) -> list:
        """get a layer of a given model considering the index of it"""
        return None

    def get_type(self) -> ModelType:
        """get the model type"""
        return ModelType.UNKNOWN

    def reload(self) -> bool:
        """reload the model and return if the process was sucessful"""
        return self.load()

    def preprocess_files_on_path(self) -> None:
        """do file manipulations to make the model loadable"""
        raise NotImplementedError

    def get_optimizer(self) -> str:
        return 0
