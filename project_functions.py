import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
mpl.rcParams.update({"font.size": 14})  # default font size

# SimPEG functionality
import simpeg
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.utils import model_builder
from simpeg import maps, data
from simpeg.electromagnetics.static.utils.static_utils import (
    write_dcip2d_ubc,
    pseudo_locations,
    plot_pseudosection,
    apparent_resistivity_from_voltage,
    generate_survey_from_abmn_locations,
    geometric_factor
)

# discretize functionality
from discretize import TreeMesh
from discretize.utils import active_from_xyz

# SURVEY TOOLS
def create_topography_from_terrain_file(terrain_file_path, terrain_file_column_names=None):
    # Use default column names if none given
    if terrain_file_column_names is None:
        terrain_file_column_names = ['x','z']
    
    # Use pandas to read .trn file
    topography_2d = pd.read_table(terrain_file_path,
                        names=terrain_file_column_names, skiprows=3, delimiter=',', skipinitialspace=True)
    
    return topography_2d

def extract_survey_and_data_from_stg(topography_2d, stg_filepath, electrode_spacing=None, datatype='volt'):
    '''
    INPUTS
    - topography_2d: Pandas dataframe containing 'x' and 'z' coordinates. The column headings should explicitly
        be 'x' and 'z'.
    - stg_filepath: Path to raw data .stg file. The column names for the dipole-dipole data are assumed to be:
        ['datum', 'type', 'date', 'time',
        'V/I', 'error in per mille', 'current [mA]', 'apparent resistivity [Ohm-m]',
        'a', 'na', 'x_mp']
    - electrode_spacing: Float, theoretical electrode spacing (e.g. 3.0, 1.5, etc.) in meters.
    - datatype: String, the data_type of the observed data to be associated with the survey (e.g. 'volt' or 
        'apparent resistivity').
        -Note: The data predicted will always be in normalized voltages, even if the observed data you feed the
        survey is in apparent resistivities.

    This function assumes that the coordinates given in the terrain file correspond exactly with the electrodes.
    
    RETURNS
    - survey: SimPEG Survey object that essentially lists the source-receiver locations for each datum
    - field_normalized_voltages_sorted: Pandas dataframe of 'V/I' column data from the .stg file;
        sorted to match the survey. Units are Volts/Ampere.
    - field_apparent_resistivities_sorted: Pandas dataframe of 'apparent resistivity [Ohm-m]' column
        data from the .stg file; sorted to match the survey. Units are Ohm-meters.
    - field_error_estimates_sorted: Pandas dataframe of 'error per mille' column data from the .stg file;
        sorted to match the survey. Divided by 1,000 such that the data are now unitless.
    - sorting_indices: Array of new indices for each datum. Use this only if you need to match unsorted field data
        to the returned survey.
    '''
    
    # The necessary dipole-dipole data are dipole size (a), distance between dipoles (na), and x-coordinate of the midpoint
    # between the dipoles (x_mp).
    raw_data_column_names = [
    'datum', 'type', 'date', 'time', 'V/I', 'error in per mille', 'current [mA]', 'apparent resistivity [Ohm-m]', 'a', 'na', 'x_mp'
    ]
    
    # Use pandas to read .stg file
    raw_data = pd.read_table(stg_filepath, sep=',', header=0, names = raw_data_column_names,
                            skiprows=3, skipfooter=1, skipinitialspace=True, engine='python',
                            usecols=[0,1,2,3,4,5,6,7,8,9,10])
    
    # Remove rows with type not = "DIP-DIP"
    raw_data = raw_data[raw_data.loc[:,'type'].str.contains("DIP-DIP", na=False)]

    # Set datatypes of a, na, and x_mp to float so that we can do math on them
    raw_data.loc[:,'a'] = raw_data.loc[:,'a'].astype(float)
    raw_data.loc[:,'na'] = raw_data.loc[:,'na'].astype(float)
    raw_data.loc[:,'x_mp'] = raw_data.loc[:,'x_mp'].astype(float)

    # Given electrode spacing s,
    # the A index, i, is i = (x_mp/s) - (na/s)/2;
    # the B index, j, is j = i - a/s;
    # the M index, k, is k = (x_mp/s) + (na/s)/2; and
    # the N index, l, is l = k + a/s.

    # Set the electrode spacing, s:
    if electrode_spacing is None:
        electrode_spacing = raw_data.loc[:,'a'].min()

    # Calculate the indices and set their data types to integers
    locations_a_indices = (raw_data.loc[:,'x_mp'] / electrode_spacing) - (raw_data.loc[:,'na'] / electrode_spacing)/2
    locations_a_indices = locations_a_indices.astype(int)

    locations_b_indices = locations_a_indices - (raw_data.loc[:,'a']/electrode_spacing)
    locations_b_indices = locations_b_indices.astype(int)

    locations_m_indices = (raw_data.loc[:,'x_mp']/ electrode_spacing) + (raw_data.loc[:,'na'] / electrode_spacing) / 2
    locations_m_indices = locations_m_indices.astype(int)

    locations_n_indices = locations_m_indices + (raw_data.loc[:,'a'] / electrode_spacing)
    locations_n_indices = locations_n_indices.astype(int)

    # Also extract the field-measured data and their associated errors
    field_apparent_resistivities = raw_data.loc[:,'apparent resistivity [Ohm-m]']
    field_normalized_voltages = raw_data.loc[:,'V/I']
    field_error_estimates = raw_data.loc[:,'error in per mille']/1000

    # Find the size of the data, n, based on the number of field-measured apparent resistivities
    n = len(field_apparent_resistivities)

    # Initialize locations arrays, with dimensions n by 2 (n for the number of data points, 2 for x and z)
    locations_a = np.zeros((n,2))
    locations_b = np.zeros((n,2))
    locations_m = np.zeros((n,2))
    locations_n = np.zeros((n,2))

    # Extract the x and z values from the topography_2d dataframe, using the indices from the .stg file
    for i in range(n):
        locations_a[i] = [topography_2d.loc[np.array(locations_a_indices)[i],'x'], topography_2d.loc[np.array(locations_a_indices)[i],'z']]
        locations_b[i] = [topography_2d.loc[np.array(locations_b_indices)[i],'x'], topography_2d.loc[np.array(locations_b_indices)[i],'z']]
        locations_m[i] = [topography_2d.loc[np.array(locations_m_indices)[i],'x'], topography_2d.loc[np.array(locations_m_indices)[i],'z']]
        locations_n[i] = [topography_2d.loc[np.array(locations_n_indices)[i],'x'], topography_2d.loc[np.array(locations_n_indices)[i],'z']]

    # Finally, generate the survey from the locations calculated above.
    survey, sorting_indices = generate_survey_from_abmn_locations(locations_a=locations_a, locations_b=locations_b,
                                                locations_m=locations_m, locations_n=locations_n,
                                                data_type=datatype, output_sorting=True)

    # Reorder field data based on sorting indices
    field_apparent_resistivities_sorted = field_apparent_resistivities.iloc[sorting_indices]
    field_error_estimates_sorted = field_error_estimates.iloc[sorting_indices]
    field_normalized_voltages_sorted = field_normalized_voltages.iloc[sorting_indices]

    # # Extract pseudo-locations from survey object
    # pseudo_locations_xz = pseudo_locations(survey)

    print("Note: If you see 'UserWarning: Ordering of ABMN locations changed when generating survey,' you may disregard it \n" \
    "because the extract_survey_and_data_from_stg function sorts the data for you. To compare the survey and sorted data \n" \
    "with the unsorted data, this function also returns the sorting_indices array that was used to sort the raw data. \n")

    return survey, field_normalized_voltages_sorted, field_apparent_resistivities_sorted, field_error_estimates_sorted, sorting_indices

# MESH TOOLS
def create_mesh_from_survey(survey, base_cell_size=None, padding_configuration=[16,6,1,1,1,1,1], height_buffer_percentage=50.0):
    '''
    INPUTS
    survey: SimPEG survey object.
    base_cell_size: Float. Minimum cell size in meters. Default is 1/4 of the average horizontal electrode spacing,
        rounded to the nearest hundredth of a meter.
    padding_configuration: Array. Equivalent to the padding_cells_by_level parameter of the TreeMesh.refine_surface()
        method. Default is [16,6,1,1,1,1,1].
    height_buffer_percentage: Float. The percentage of the depth to the deepest pseudo-location that is ensured to
        be included in the domain height. Default is 50.

    RETURNS
    mesh: SimPEG TreeMesh object.
    active_cells: Array containing Booleans indicating if the mesh cells are active (True) or not (False).
    '''

    # Extract electrode coordinates and pseudo-locations from the survey
    electrode_coordinates = survey.unique_electrode_locations
    survey_pseudo_locations = pseudo_locations(survey)

    # Set default base_cell_size to 1/4 of the average horizontal electrode spacing, rounded to the nearest
    # hundredth of a meter.
    if base_cell_size is None:
        base_cell_size = np.round(np.mean(np.diff(electrode_coordinates[:,0]))/4, decimals=2)
    
    # Domain width
    domain_width = electrode_coordinates[:,0].max() - electrode_coordinates[:,0].min()
    # Domain height, with a buffer based on percentage of depth to deepest pseudo-location
    domain_height = (
        (electrode_coordinates[:,1].max()-survey_pseudo_locations[:,1].min())
        *(1 + height_buffer_percentage/100)
        )
    
    # Number of base cells in x and z
    nbcx = 2 ** (int(np.ceil(np.log(domain_width / base_cell_size) / np.log(2.0)))+1)
    nbcz = 2 ** (int(np.ceil(np.log(domain_height / base_cell_size) / np.log(2.0))))

    # Define the base mesh with top at z = 0m.
    hx = [(base_cell_size, nbcx)]
    hz = [(base_cell_size, nbcz)]
    mesh = TreeMesh([hx, hz], x0="CN", diagonal_balance=True)

    # Shift top to maximum topography, shift center to electrode center
    center=np.ceil(np.median(electrode_coordinates[:,0]))
    mesh.origin = mesh.origin + np.r_[center, electrode_coordinates[:,1].max()]

    # Mesh refinement based on topography
    mesh.refine_surface(
        electrode_coordinates,
        padding_cells_by_level=padding_configuration,
        finalize=True,
    )

    # Define active cells
    active_cells = active_from_xyz(mesh, electrode_coordinates)

    print(f"Base cell size: {base_cell_size} m")
    print(f"padding_cells_by_level parameter: {padding_configuration}")
    print(f"# of cells: {mesh.n_cells}")  # Number of cells
    print(f"# of nodes: {mesh.ntN}")
    print(f"Max cell volume: {mesh.cell_volumes.max()} sq.m.")  # Largest cell size
    return mesh, active_cells

# DATA TOOLS
# def add_noise_and_create_data_object(data_values, survey, noise_level=0.05, export=False, filename=None, data_type='volt', comment_lines=""):
#     '''
#     Creates a SimPEG Data object from data and the survey associated with it. Optionally, will export the Data object to
#     "./outputs/Data_objects/filename.obs"

#     INPUTS
#     data_values: Array. Normalized voltages or apparent resistivities. Specify which one with data_type.
#     survey: SimPEG Survey object associated with the data_values. The Survey data_type should match the data_values data_type.
#     noise_level: Float. The factor of the data value that will be used as the standard deviation of the added noise.
#         Default value is 0.05 (i.e., standard deviation of noise = 5% of data value)
#     export: Boolean. Set to True to export Data object to .obs file. If True, filename should be specified.
#     filename: String. Must be specified if export = True. The object will be saved to "./outputs/Data_objects/filename.obs"
#     data_type: 'volt' or 'apparent resistivity'. Default is 'volt'.
#     comment_lines: String. Optional lines printed to the beginning of the file
    
#     RETURNS
#     data_object: SimPEG Data object containing the data values.
#     '''
#     # To add noise, first create a random number generator
#     random_number_generator = np.random.default_rng(seed=225)
#     # Add the noise_level
#     standard_deviation = noise_level * np.abs(data_values)
#     noise = random_number_generator.normal(scale=standard_deviation, size=len(data_values))
#     data_observed = data_values + noise

#     data_object = simpeg.data.Data(survey, dobs=data_observed, standard_deviation=standard_deviation)

#     if export is True:
#         if filename is None:
#             print('Specify filename!')
#             return data_object
#         filepath = "./outputs/Data_objects/"+filename+".obs"
#         write_dcip2d_ubc(filepath,data_object,data_type=data_type,file_type='dobs',comment_lines=comment_lines)
#         print(f'The Data object was saved to {filepath}')

#     return data_object

def create_data_object(data_values, survey, 
                       add_noise, noise_level=0.03, uncertainties=None,
                       export=False, filename=None, data_type='volt', comment_lines=""):
    '''
    Creates a SimPEG Data object from data and the survey associated with it. Optionally, will export the Data object to
    "./outputs/Data_objects/filename.obs"

    INPUTS
    data_values: Array. Normalized voltages or apparent resistivities. Specify which one with data_type.
    survey: SimPEG Survey object associated with the data_values. The Survey data_type should match the data_values data_type.
    add_noise: Boolean. Set to True to add noise to data_values before creating the Data object.
    noise_level: Float. The factor of the data value that will be used as the standard deviation of the added noise.
        Default value is 0.03 (i.e., standard deviation of noise = 3% of data value), suggested by EarthImager.
    uncertainties: Array of factors of uncertainty, typically from raw data. Note that if the error is greater than 0 but less than
        0.01, it will be rounded up to 0.01. If there is no measure of uncertainty (i.e., the error is 0.000), the
        noise_level will be used.
    export: Boolean. Set to True to export Data object to .obs file. If True, filename should be specified.
    filename: String. Must be specified if export = True. The object will be saved to "./outputs/Data_objects/filename.obs"
    data_type: 'volt' or 'apparent resistivity'. Default is 'volt'.
    comment_lines: String. Optional lines printed to the beginning of the file
    
    RETURNS
    data_object: SimPEG Data object containing the data values.
    uncertainties: Numpy array of final errors used.
    '''
    if add_noise is True:
        # To add noise, first create a random number generator
        random_number_generator = np.random.default_rng(seed=225)
        # Add the noise_level
        standard_deviation = noise_level * np.abs(data_values)
        noise = random_number_generator.normal(scale=standard_deviation, size=len(data_values))
        data_to_save = data_values + noise

    elif add_noise is False:
        if uncertainties is None:
            # Assume that the standard deviation of uncertainties is the default value of noise_level
            standard_deviation = noise_level * np.abs(data_values)
        elif uncertainties is not None:
            for i in range(len(uncertainties)):
                # If the error is greater than 0 but less than 0.01, it will be rounded up to 0.01. 
                # If there is no measure of uncertainty (i.e., the error is 0.000), the noise_level will be used.
                if uncertainties[i] == 0:
                    uncertainties[i] = 0.03
                elif uncertainties[i] < 0.01:
                    uncertainties[i] = 0.01
                else:
                    pass

            standard_deviation = uncertainties * np.abs(data_values)
        
        data_to_save = data_values

    else:
        print('Specify add_noise parameter!')

    data_object = simpeg.data.Data(survey, dobs=data_to_save, standard_deviation=standard_deviation)

    if export is True:
        if filename is None:
            print('Specify filename!')
            return data_object
        else:
            filepath = "./outputs/Data_objects/"+filename+".obs"
            write_dcip2d_ubc(filepath,data_object,data_type=data_type,file_type='dobs',comment_lines=comment_lines)
            print(f'The Data object was saved to {filepath}')

    return data_object, uncertainties

# MODEL TOOLS
def save_model_to_txt(model, model_filename):
    '''
    Saves the input model to the './outputs/Models/' directory as a .txt file.
    
    INPUTS
    model: Array. Contains the model parameters.
    model_filename: String.
    
    RETURNS
    Nothing
    '''
    # Construct the full path
    output_directory = './outputs/Models/'
    full_path = os.path.join(output_directory, model_filename + '.txt')

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    else:
        print(f"Directory already exists: {output_directory}")

    # Try to save the file
    try:
        np.savetxt(full_path, model)
        print(f"Successfully saved the model to: {full_path}")
    except Exception as e:
        print(f"An error occurred while saving: {e}")
    
    return

def load_model_from_txt(file_to_import, import_directory='./outputs/Models/'):
    full_path = os.path.join(import_directory, file_to_import)
    return np.loadtxt(full_path)