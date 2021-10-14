import os
import logging

logging.basicConfig(level=logging.DEBUG)


def list_all_files(path: str) -> list:
    """list all files of any extension in all the path directories"""
    return [os.path.join(r, file) for r, d, f in os.walk(path) for file in f]


def list_all_folders(path: str) -> list:
    """list all directories and subdirectories in a given path"""
    return [os.path.join(r, directory) for r, d, f in os.walk(path) for directory in d]


def is_tensorflow(path: str) -> bool:
    """
    check if the folder is a tensorflow related one it must be a savedModel from tensorflow2.0+
    It's based on file checking, it must have the files: .pb .index .data and the /variables dir 
    """
    must_have_files = {".pb": False, ".index": False, "data": False}
    must_have_folders = {"variables": False}
    local_files = list_all_files(path)
    for key in must_have_files.keys():
        for file in local_files:
            if (str(key) in str(file)):
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
