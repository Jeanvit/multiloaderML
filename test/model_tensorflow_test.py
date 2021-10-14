"""
Author: Jean Vitor de Paulo
Date  : 13/10/21
Last updated: 
"""

from src.models.model_tensorflow import TensorflowModel
import tensorflow as tf


def dummy_model(save_path: str = "") -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(32,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    if (save_path != ""):
        model.save(save_path)
    else:
        return model


def test_model_istance():
    tensorflow_model = TensorflowModel("/test/", "name")
    assert tensorflow_model.model_name == "name"
    assert tensorflow_model.model_path == "/test/"
    assert tensorflow_model.model_instance == None
    assert tensorflow_model.is_loaded == False


def test_model_force_new_instance():
    tensorflow_model = TensorflowModel("", "", False)
    tensorflow_model.model_instance = dummy_model()
    assert tensorflow_model.load(dummy_model()) == True
    assert tensorflow_model.model_name == ""
    assert tensorflow_model.model_path == ""
    assert tensorflow_model.is_loaded == True


def test_model_simple_properties():
    tensorflow_model = TensorflowModel("", "", False)
    tensorflow_model.load(dummy_model())
    assert tensorflow_model.is_loaded == True
    assert tensorflow_model.get_number_of_layers() == 2
    assert tensorflow_model.get_input_shape() == (None, 32)
    assert tensorflow_model.get_number_of_params() == 33
    assert tensorflow_model.get_output_dimensionality() == 1
    assert tensorflow_model.get_output_names() == ['dense_2']
    assert tensorflow_model.get_input_names() == ['input_3']


def test_model_instances_types():
    tensorflow_model = TensorflowModel("", "", False)
    tensorflow_model.load(dummy_model())
    assert isinstance(tensorflow_model.get_single_layer_by_index(1),
                      tf.keras.layers.Dense)
    assert isinstance(tensorflow_model.get_optimizer(), tf.keras.optimizers.Adam)