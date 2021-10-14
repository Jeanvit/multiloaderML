"""
Author: Jean Vitor de Paulo
Date  : 13/10/21
Last updated: 
"""

from model_base import Model, ModelType
from src.models.model_tfkeras import TFKerasModel
from fileprocessor import file_utils

"""Available implementations"""
Factories = {
    ModelType.TFKERAS:    TFKerasModel,
    ModelType.PYTORCH:    None
}


class ModelFactory:
    """
    Factory class for the heavy lifting of model type handling
    Model idetification is done at __init__ and get() does the job
    of return it to you
    """

    model_type: ModelType
    path: str
    name: str

    def __init__(self, model_path: str, model_name: str ) -> None:
        self.path = model_path
        self.name = model_name
        self.model_type = self.identify_model_type()

    def get(self) -> Model:
        """get the the model you are you loaded"""
        if (self.model_type in Factories and self.model_type != ModelType.UNKNOWN):
            return Factories[self.model_type](self.path, self.name)
        else:
            raise ValueError(
                "Invalid factory type choose one of these: " + str(Factories))

    def is_tfkeras(self):
        """
        check if the folder is a tf.keras related one it must be a savedModel from tensorflow2.0+
        It's based on file checking, it must have the files: .pb .index .data and the /variables dir 
        """
        return file_utils.is_tfkeras(self.path)

    def is_pytorch(self):
        return False

    def is_keras(self):
        """
        Almost the same as tfkeras, the difference is that there is an extra keras_metadata.pb
        """
        return file_utils.is_keras(self.path)

    def is_HDF5(self):
        return file_utils.is_HDF5(self.path)

    def identify_model_type(self) -> ModelType:
        if self.is_tfkeras() or self.is_HDF5() or self.is_keras():
            return ModelType.TFKERAS
        elif self.is_pytorch():
            return ModelType.PYTORCH
        else:
            return ModelType.UNKNOWN