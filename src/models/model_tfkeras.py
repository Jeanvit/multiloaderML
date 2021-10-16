"""
Author: Jean Vitor de Paulo
Date  : 12/10/21
Last updated: 15/10/21
"""

import logging
import tensorflow as tf
from models.model_base import Model, ModelType
from pathlib import Path
import os

class TFKerasModel(Model):
    """This is the class for handling tf.keras models behaviour and properties"""
    model_path: str
    model_name: str
    is_loaded: bool
    model_instance: any

    def __init__(self, path: str, name: str, load_on_start: bool = True) -> None:
        self.model_path = path
        self.model_name = name
        self.is_loaded = False
        if load_on_start:
            self.load()

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
        """load a model using only its path and return if it was successful"""
        if (forced_new_istance is not None):
            logging.warning("Forcing a new model instance")
            self.model_instance = forced_new_istance
            self.is_loaded = True
            return True
        try:
            self.model_instance = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
        except Exception as e:
            logging.error("Error loading model: " + str(e))
            self.model_instance = None
            self.is_loaded = False
            return False
        return True

    def is_valid(self) -> bool:
        """check if the model has a valid structure"""
        return self.is_loaded

    def get_number_of_layers(self) -> int:
        return len(self.model_instance.layers)

    def get_complexity_level(self) -> float:
        """complexity level measure how much computing power the model will need"""
        value: float
        try:
            value = self.get_number_of_layers() * self.get_number_of_params()
            value = (value/1000000) / self.get_model_size()
            return round(value,3)
        except Exception as e:
            logging.error("Cant measure model size " + str(e))
            return 0

    def get_model_size(self) -> float:
        size: float = 0
        model = Path(self.model_path)
        if (not model.is_dir()):
            size = os.path.getsize(model)
        else:
            for path, dirs, files in os.walk(self.model_path):
                for f in files:
                    fp = os.path.join(path, f)
                    size += os.path.getsize(fp)
        return size / (1024*1024.0)

    def get_number_of_params(self) -> int:
        """get the total number of parameters of a given model"""
        return self.model_instance.count_params()

    def get_input_shape(self) -> str:
        """get the input shape of a given model"""
        return self.model_instance.input_shape

    def get_input(self) -> str:
        """get the input shape of a given model"""
        return self.model_instance.input

    def get_input_dimensionality(self, consider_batch=False) -> int:
        """get the input dimensions of a given model, subtractring -1 when considering batch"""
        dimensionality = len(self.get_input_shape())
        if (not consider_batch):
            dimensionality -= 1
        return dimensionality

    def get_output_shape(self) -> str:
        """get the output shape of a given model"""
        return self.model_instance.output_shape

    def get_output(self) -> str:
        """get the output a given model"""
        return self.model_instance.input

    def get_output_dimensionality(self, consider_batch=False) -> int:
        """get the output dimensions of a given model, subtractring -1 when considering batch"""
        dimensionality = len(self.get_output_shape())
        if (not consider_batch):
            dimensionality -= 1
        return dimensionality

    def get_single_layer_by_index(self, index: int) -> list:
        """get a layer of a given model considering the index of it"""
        try:
            return self.model_instance.get_layer(None, index)
        except Exception as e:
            logging.error("Wrong index number: " + str(e))
            return None

    def get_type(self) -> ModelType:
        """get the model type"""
        return ModelType.TFKERAS

    def reload(self) -> bool:
        """reload the model and return if the process was sucessful"""
        return self.load()

    def preprocess_files_on_path(self) -> None:
        """do file manipulations to make the model loadable"""
        raise NotImplementedError

    def get_optimizer(self) -> str:
        return self.model_instance.optimizer

    """Custom TFKERAS methods"""

    def get_metric_names(self) -> any:
        return self.model_instance.metrics_names

    def get_output_names(self) -> list:
        """get name of all model's outputs"""
        return self.model_instance.output_names

    def get_input_names(self) -> list:
        """get name of all model's inputs"""
        return self.model_instance.input_names
