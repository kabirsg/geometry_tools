import vtk 
import pyvista as pv 
import numpy as np 
import vmtk
import os

def calculate_hydraulic_diameter(): 
    pass


if __name__ == "__main__":
    FILENAME = ""
    if not os.path.exists(FILENAME):
        raise FileNotFoundError(f"File not found at: {FILENAME}\nPlease check file path again and retry")
    calculate_hydraulic_diameter(FILENAME)