"""
Author: Jean Vitor de Paulo
Date  : 13/10/21
Last updated: 
"""

from src.models.model_tfkeras import TFKerasModel
import tensorflow as tf


def dummy_model(save_path: str = "") -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(32,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    if (save_path != ""):
        tf.keras.models.save_model(model,save_path)
    else:
        return model


def test_model_istance():
    tf_keras_model = TFKerasModel("/test/", "name")
    assert tf_keras_model.model_name == "name"
    assert tf_keras_model.model_path == "/test/"
    assert tf_keras_model.model_instance == None
    assert tf_keras_model.is_loaded == False


def test_model_force_new_instance():
    tf_keras_model = TFKerasModel("", "", False)
    tf_keras_model.model_instance = dummy_model()
    assert tf_keras_model.load(dummy_model()) == True
    assert tf_keras_model.model_name == ""
    assert tf_keras_model.model_path == ""
    assert tf_keras_model.is_loaded == True


def test_model_simple_properties():
    tf_keras_model = TFKerasModel("", "", False)
    tf_keras_model.load(dummy_model())
    assert tf_keras_model.is_loaded == True
    assert tf_keras_model.get_number_of_layers() == 2
    assert tf_keras_model.get_input_shape() == (None, 32)
    assert tf_keras_model.get_number_of_params() == 33
    assert tf_keras_model.get_output_dimensionality() == 1
    assert tf_keras_model.get_output_names() == ['dense_2']
    assert tf_keras_model.get_input_names() == ['input_3']
    assert tf_keras_model.get_complexity_level() == 0

def test_model_instances_types():
    tf_keras_model = TFKerasModel("", "", False)
    tf_keras_model.load(dummy_model())
    assert isinstance(tf_keras_model.get_single_layer_by_index(1),
                      tf.keras.layers.Dense)
    assert isinstance(tf_keras_model.get_optimizer(), tf.keras.optimizers.Adam)