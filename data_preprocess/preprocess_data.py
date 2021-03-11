import os
import shutil
import copy


"""
./list_of_test_videos -> ./test & ./val & ./train with balanced classes
"""



def preproces_data(data, path,):
    if data == "Celebdf":
        pass