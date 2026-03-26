import numpy as np 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import vmtk
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from datetime import datetime

class LumpedParameter:
    def __init__(self, cline_file, Q, rho, Kt, mu):
        self.centerline_file = cline_file
        self.flow_rate = Q
        self.density = rho
        self.Kt = Kt
        self.dyn_viscosity = mu

        #Creating the polydata object
        if not Path(self.centerline_file).exists(): 
            raise FileNotFoundError("The centerline file at the path specified could not be found. Please double check the path provided")

        reader = vtk.vtkXMLPolyDataReader() 
        reader.SetFileName(self.centerline_file)
        reader.Update()
        self.polydata = reader.GetOutput()

        self.create_arrays() #Get the data from the centerline file

    def create_arrays(self):
        self.point_length_dict = {}#This is a dictionary with key : value pairs being -> point_id : (point_coords, length_along_centerline)
        self.radius_array_np = vtk_to_numpy(self.polydata.GetPointData().GetArray("MaximumInscribedSphereRadius"))
        self.point_array_np = vtk_to_numpy(self.polydata.GetPoints().GetData())
        self.length_array = [0] #Array holding the length along the centerline for all the points - To not mess up indexes, the value for index 0 is given a filler value

        length_along_cl = 0 #The distance along the centerline will be calculated as a sum of the distances between all the points along the centerline
        #Not including the first or last point - messes it up for some reason
        for i in range(1, len(self.point_array_np)-1):
            point_i = self.point_array_np[i] #Returns the [X Y Z] coordinates for the centerline point
            length_along_cl += np.linalg.norm(self.point_array_np[i+1] - point_i)
            self.length_array.append(length_along_cl)
            self.point_length_dict[i] =  length_along_cl #point_id : length_along_centerline
    
    def calculate_viscous_resistances(self):
        self.viscous_resistances = [] #List for viscous resistances
        CONST_TERM = 8*self.dyn_viscosity/np.pi #The constant term in the viscous resistance equation

        #Resistance contribution of all the centerline points until and excluding the last point
        #The length (L) is half the distance from the last point to this point and half the distance from this point to the next
        #Not using the first or last point since their radius values are a little funky and they are in the flow extension region anyways
        for i in range(1, len(self.point_array_np)-2):
            #Calculate the distances between the points
            back_L = self.length_array[i] - self.length_array[i-1]
            forward_L = self.length_array[i+1] - self.length_array[i]

            #Calculate the length of the segment for this centerline point
            L_i = back_L/2 + forward_L/2

            #Calculate the viscous resistance at this centerline point
            visc_res = CONST_TERM * L_i / (self.radius_array_np[i] ** 4)
            self.viscous_resistances.append(visc_res)

    def create_min_max_array(self):
        minima_indices = argrelextrema(self.radius_array_np, np.less, order=3)[0] #Gets the inidices of the local minima - order = 3 means that 3 points on each side used for comparison to reduce noise
        maxima_indices = argrelextrema(self.radius_array_np, np.greater, order=3)[0]
        start_min = minima_indices[0] < maxima_indices[0] #True if the index of the first minima is less than the index of the first maxima
        
        return minima_indices, maxima_indices, start_min

    def calculate_added_resistance(self, A_s, A_0):
        try:
            return (self.density * self.Kt/(2*A_0**2) * (A_0/A_s - 1) ** 2) * abs(self.flow_rate)
        except Exception as e:
            print(f'Exception encountered: {e}')
            return 0

    def calculate_expansion_resistances(self):
        # extrema, start_minima = self.create_min_max_array()
        min_indices, max_indices, start_min = self.create_min_max_array()
        extrema_array = np.sort(np.concatenate((min_indices, max_indices))) #List of the local min, max, min, max, etc. This works because they are always going to alternate min, max, etc. 
        pdrop_dict = {} #Empty for now - Eventually, Index : expansion pressure drop
        expansion_resistance = 0.0
        num_extrema = len(min_indices) + len(max_indices)

        if extrema_array[0] == min_indices[0]:
            first = "min"
        else:
            first = "max"
        
        if extrema_array[-1] == min_indices[-1]:
            last = "min"
        else:
            last = "max"

        #If the first element is a maximum, no issues
        #If it's a minimum 
        if first == "min":
            A_0 = np.pi * self.radius_array_np[extrema_array[0]] ** 2
            A_s = np.pi * self.radius_array_np[extrema_array[1]] ** 2

            delta_R = self.calculate_added_resistance(A_s, A_0)
            pdrop_dict[min_indices[i]] = delta_R
            expansion_resistance += delta_R

            #The first and last values aren't handled by the for loop
            for i in range(1, len(min_indices)-1):
                extrema_i = np.where(extrema_array == min_indices[i])[0][0]
                A_s = np.pi * self.radius_array_np[min_indices[i]] ** 2
                A_0 = np.pi * ((self.radius_array_np[extrema_array[extrema_i-1]] + self.radius_array_np[extrema_array[extrema_i+1]]) / 2) ** 2
                
                delta_R = self.calculate_added_resistance(A_s, A_0)
                pdrop_dict[min_indices[i]] = delta_R
                expansion_resistance += delta_R
            
            if last == "min":
                A_s = np.pi * self.radius_array_np[extrema_array[-1]] ** 2
                A_0 = np.pi * self.radius_array_np[extrema_array[-2]] ** 2
            else:
                A_s = np.pi * self.radius_array_np[extrema_array[-2]] ** 2
                A_0 = np.pi * ((self.radius_array_np[extrema_array[-3]] + self.radius_array_np[extrema_array[-1]]) / 2) ** 2
            
            delta_R = self.calculate_added_resistance(A_s, A_0)
            pdrop_dict[min_indices[i]] = delta_R
            expansion_resistance += delta_R
        
        #The first element is a maxima
        else:
            for i in range(0, len(min_indices)-1):
                print(min_indices[i])
                extrema_i = np.where(extrema_array == min_indices[i])[0][0]
                A_s = np.pi * self.radius_array_np[min_indices[i]] ** 2
                A_0 = np.pi * ((self.radius_array_np[extrema_array[extrema_i-1]] + self.radius_array_np[extrema_array[extrema_i+1]]) / 2) ** 2
                
                delta_R = self.calculate_added_resistance(A_s, A_0)
                pdrop_dict[min_indices[i]] = delta_R
                expansion_resistance += delta_R

            if last == "min":
                A_s = np.pi * self.radius_array_np[extrema_array[-1]] ** 2
                A_0 = np.pi * self.radius_array_np[extrema_array[-2]] ** 2
            else:
                A_s = np.pi * self.radius_array_np[extrema_array[-2]] ** 2
                A_0 = np.pi * ((self.radius_array_np[extrema_array[-3]] + self.radius_array_np[extrema_array[-1]]) / 2) ** 2
            
            delta_R = self.calculate_added_resistance(A_s, A_0)
            pdrop_dict[min_indices[-1]] = delta_R
            expansion_resistance += delta_R
        self.expansion_resistances = expansion_resistance #C: NAMING
        self.p_drop_dict = pdrop_dict
        print(expansion_resistance)
        return expansion_resistance, pdrop_dict
    
    '''
    This function uses a different interpretation of the expansion terms and calculates the expansion resistance between every two points if the radius has decreased at a point compared to the last
    Does not seem to be viable
    '''
    def calculate_expansion_resistances_v2(self):
        pdrop_dict = {} #C: This should be named something like "added_resistance_dict"
        expansion_resistance = 0.0 #C: Expansion resistance total
        radius_array = self.radius_array_np
        for i in range(1, len(radius_array)-10):
            if radius_array[i] > radius_array[i-1]:
                print(i)
                A_s = np.pi * radius_array[i] ** 2
                A_0 = np.pi * ((radius_array[i-1] + radius_array[i+1]) / 2) ** 2
                delta_R = self.calculate_added_resistance(A_s, A_0)
                pdrop_dict[i] = delta_R
                expansion_resistance += delta_R
        
        # if radius_array[-1] < radius_array[-2]:
        #     A_s = np.pi * radius_array[-1] ** 2
        #     A_0 = np.pi * radius_array[-2] ** 2
        #     delta_R = self.calculate_delta_added_resistance(A_s, A_0)
        #     pdrop_dict[i] = delta_R
        #     expansion_pressure += delta_R
        self.expansion_resistances = expansion_resistance #C: NAMING
        self.p_drop_dict = pdrop_dict
        return expansion_resistance, pdrop_dict

    def calculate_curvature_resistances(self):
        pass


    def calculate_pressures(self):
        self.pressure_drops = [] #List of pressure drops due to resistances of each segment
        self.pressures = [] #List of pressures at each point
        # Inlet flow rate * total resistance = inlet pressure (assuming pressure at outlet = 0 -> think of this as the difference in pressure between inlet and outlet)
        total_r = sum(self.viscous_resistances) + self.expansion_resistances #C: THIS WILL HAVE TO CHANGE LATER TO BE THE TOTAL RESISTANCES
        pressure = total_r * self.flow_rate #inlet pressure
        resistances = self.viscous_resistances #C: THIS WILL HAVE TO CHANGE LATER TO BE THE TOTAL RESISTANCES
        for key, val in self.p_drop_dict.items():
            resistances[key] += val

        for resistance in resistances:
            delta_P = self.flow_rate * resistance #Pressure drop over each segment due to the resistive elements in that segment
            self.pressure_drops.append(delta_P)

            #Calculating new pressure
            pressure -= delta_P
            #Adding new pressure to the list of pressures at each point
            self.pressures.append(pressure)

        return self.pressures, self.pressure_drops
    
    def calculate_pressures_no_exp(self):
        self.pressure_drops = [] #List of pressure drops due to resistances of each segment
        self.pressures = [] #List of pressures at each point
        # Inlet flow rate * total resistance = inlet pressure (assuming pressure at outlet = 0 -> think of this as the difference in pressure between inlet and outlet)
        total_r = sum(self.viscous_resistances) #C: THIS WILL HAVE TO CHANGE LATER TO BE THE TOTAL RESISTANCES
        pressure = total_r * self.flow_rate #inlet pressure
        resistances = self.viscous_resistances #C: THIS WILL HAVE TO CHANGE LATER TO BE THE TOTAL RESISTANCES

        for resistance in resistances:
            delta_P = self.flow_rate * resistance #Pressure drop over each segment due to the resistive elements in that segment
            self.pressure_drops.append(delta_P)

            #Calculating new pressure
            pressure -= delta_P
            #Adding new pressure to the list of pressures at each point
            self.pressures.append(pressure)
        
        return self.pressures, self.pressure_drops

    def generate_viscous_resistance_plots(sef, viscous_resistances, total_v):
        pass

    def generate_expansion_resistance_plots(self, exp_res, total_e):
        fig, ax = plt.figure(figsize=(10,8))

        ax.plot(self.expansion_resistances)

    def generate_pressure_plots(self, exp_v):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))

        points = np.arange(len(self.pressures))
        seg_points = np.arange(len(self.pressure_drops))

        # Pressure along vessel
        ax1.plot(self.length_array[1:-1], self.pressures, 'b-o', markersize=4)
        ax1.set_xlabel("Length Along Centerline (mm)")
        ax1.set_ylabel("Pressure (Pa)")
        ax1.set_title("Pressure Along Vessel")
        ax1.grid(True)

        # Pressure drops at each segment
        ax2.bar(self.length_array[11:-1], self.pressure_drops[10:], color='red', alpha=0.7)
        ax2.set_xlabel("Length Along Centerline (mm)")
        ax2.set_ylabel("Pressure Drop (Pa)")
        ax2.set_title("Pressure Drop at Each Point Along Vessel")
        ax2.grid(True, axis='y')

        plt.tight_layout()
        if exp_v == 1:
            plt.savefig("../dlp_output_figures/pressure_results_vis_exp.png", dpi=150)
        elif exp_v == 2:
            plt.savefig("../dlp_output_figures/pressure_results_vis_exp_v2.png", dpi=150)
        else:
            plt.savefig("../dlp_output_figures/pressure_results_vis.png", dpi=150)
        plt.show()

    def debug(self, txt_file_name, desc):
        text_lines = []
        text_lines.append(f"Description: {desc}\n")
        if hasattr(self, "viscous_resistances"):
            v_res_sum = sum(self.viscous_resistances)
            text_lines.append(f"Viscous Total Resistance: {v_res_sum}\n")
            text_lines.append(f"Pressure drop due to viscous losses: {v_res_sum * self.flow_rate}\n")
        if hasattr(self, "expansion_resistances"):
            text_lines.append(f"Expansion Total Resistance: {self.expansion_resistances}\n")
            text_lines.append(f"Pressure drop due to expansion losses: {self.expansion_resistances * self.flow_rate}\n")

        text_lines.append('\n')

        #Actually writing to the text file
        with open(txt_file_name, "a") as f:
            f.writelines(text_lines)

if __name__ == "__main__":
    #CONSTANTS
    BLOOD_DYNAMIC_VISCOSITY = 0.04 #dynamic viscosity mu value [Poise]
    INLET_FLOW_RATE = 5.58 #mL/s - same as Back to Bernoulli paper
    KT = 1.52 #Same as Mirramezani paper
    DENSITY = 1.06 #g/mL or g/cm^3

    CLINE_FILE = "/home/kabir/PT/PTSeg028_v3/PTSeg028_cl_centerline_graph_vmtk.vtp"
    EXPANSION = 1 #0, 1, or 2
    DEBUG = True

    lp = LumpedParameter(cline_file=CLINE_FILE, Q=INLET_FLOW_RATE, rho=DENSITY, Kt=KT, mu=BLOOD_DYNAMIC_VISCOSITY)
    
    #Calculating viscous resistance
    lp.calculate_viscous_resistances()
    
    #Calculate expansion resistance if necessary
    if EXPANSION == 0:
        lp.calculate_pressures_no_exp()
    elif EXPANSION == 1:
        lp.calculate_expansion_resistances()
        lp.calculate_pressures()
    elif EXPANSION == 2:
        lp.calculate_expansion_resistances_v2()
        lp.calculate_pressures()

    #Output to debug file if desired
    if DEBUG:
        desc = f"EXPANSION = {EXPANSION} \t->\t{datetime.now().strftime('%H:%M:%S')}"
        lp.debug(txt_file_name="debug_dlp.txt", desc=desc)

    #lp.generate_viscous_resistance_plots()
    #lp.generate_expansion_resistance_plots()
    lp.generate_pressure_plots(EXPANSION)
