import os
import logging

logging.basicConfig(level=logging.DEBUG)
EXTRA_KERAS_FILE = "keras_metadata.pb"


def list_all_files(path: str) -> list:
    """list all files of any extension in all the path directories"""
    return [os.path.join(r, file) for r, d, f in os.walk(path) for file in f]


def list_all_folders(path: str) -> list:
    """list all directories and subdirectories in a given path"""
    return [os.path.join(r, directory) for r, d, f in os.walk(path) for directory in d]


def is_tfkeras(path: str) -> bool:
    """
    check if the folder is a tf.keras related one it must be a savedModel from tensorflow2.0+
    It's based on file checking, it must have the files: .pb .index .data and the /variables dir 
    """
    must_have_files = {".pb": False, ".index": False, "data": False}
    must_have_folders = {"variables": False}
    local_files = list_all_files(path)
    for key in must_have_files.keys():
        for file in local_files:
            if (str(key) in str(file) and not(EXTRA_KERAS_FILE in str(file))):
                must_have_files[key] = True
                break
    local_folders = list_all_folders(path)
    for key in must_have_folders.keys():
        for file in local_folders:
            if (str(key) in str(file)):
                must_have_folders[key] = True
                break
    logging.debug("Files:" + str(must_have_files) +
                  " dirs: " + str(must_have_folders))
    return (all(value == True for value in must_have_files.values())) and (all(value == True for value in must_have_folders.values()))


def is_HDF5_tensorflow(path: str) -> bool:
    HDF5str = ".hdf5"
    H5str = ".h5"
    if (path.lower().endswith(HDF5str) or path.lower().endswith(H5str)):
        return True
    return False


def is_keras(path: str) -> bool:
    # current version of tensorflow speficies this extra file for keras models
    if(not is_tfkeras(path)):
        """Latest versions of tf.keras savedMode just adds a keras_metadata file"""
        return False
    print(EXTRA_KERAS_FILE in list_all_files(path))
    for file in list_all_files(path):
        print (file, EXTRA_KERAS_FILE)
        if (EXTRA_KERAS_FILE in file):
            return True
    return False
