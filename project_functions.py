import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# SimPEG functionality
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

def create_topography_from_terrain_file(terrain_file_path, terrain_file_column_names=None):
    # Use default column names if none given
    if terrain_file_column_names is None:
        terrain_file_column_names = ['x','z']
    
    # Use pandas to read .trn file
    topography_2d = pd.read_table(terrain_file_path,
                        names=terrain_file_column_names, skiprows=3, delimiter=',', skipinitialspace=True)
    
    return topography_2d

def plot_topography_from_dataframe(topography_dataframe, upper_buffer=5, lower_buffer_fraction=0.25, vertical_exaggeration=1, ax1=None):
    '''
    INPUTS
    topography_dataframe: Pandas dataframe with x on the first column and elevation on the second column
    upper_buffer: Upper ylimit above the highest elevation. Default is 5.
    lower_buffer_fraction: Fraction of the width along x that will be used to extend the y-axis down below
    the lowest elevation. Default is 1/4.
    vertical_exaggeration: Float value of vertical exaggeration.

    RETURNS
    ax1: Matplotlib figure Axes object of topography plotted with the input parameters.
    '''
    if ax1 is None:
        # Create figure
        fig1, ax1 = plt.subplots(1,1, figsize = (12,3))
    
    # Plot z-column vs x-column
    ax1.plot(topography_dataframe.iloc[:,0], topography_dataframe.iloc[:,1], color="b", linewidth=2)
    # Set vertical exaggeration to 1
    ax1.set_aspect(vertical_exaggeration, adjustable='box')
    # Extend ylimit down by a fraction of full x-width, and buffer up by 5 m
    fraction = 1/4
    depth = (topography_dataframe.iloc[:,0].max() - topography_dataframe.iloc[:,0].min()) * fraction
    ax1.set_ylim(topography_dataframe.iloc[:,1].min()-depth,topography_dataframe.iloc[:,1].max()+5)
    # Other plot elements
    ax1.grid()
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('Elevation [m]')
    ax1.set_title(f'Topography (VE={vertical_exaggeration})', fontsize=16, pad=10)
    plt.show()

    return ax1

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

def plot_survey(survey, vertical_exaggeration=1, ax=None):
    # Extract pseudo-locations from survey object
    survey_pseudo_locations = pseudo_locations(survey)

    # Extract topography
    survey_electrode_locations = survey.unique_electrode_locations

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(12,4))
    
    # Plot topography (i.e., electrode locations as a line).
    ax.plot(survey_electrode_locations[:,0], survey_electrode_locations[:,1], color="b", linewidth=2, label='surface', zorder=1)
    
    # Plot electrodes as points.
    ax.scatter(survey_electrode_locations[:,0], survey_electrode_locations[:,1], 20, "r", label='electrodes')
    
    # Plot pseudo-locations
    ax.scatter(survey_pseudo_locations[:,0], survey_pseudo_locations[:,-1], 8, "b", label='pseudo-locations')
    # Set vertical exaggeration to 1
    ax.set_aspect(vertical_exaggeration, adjustable='box')

    # Other plot elements
    ax.grid()
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('Elevation [m]')
    ax.set_title(f'Survey setup (VE={vertical_exaggeration})', fontsize=16, pad=10)
    plt.show()

    return ax