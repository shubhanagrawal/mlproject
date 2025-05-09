import os
import sys
import numpy as np
import pandas as pd
import os
import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        print(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise Exception(f"Error saving object to {file_path}: {e}")
