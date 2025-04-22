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

# discretize functionality
from discretize import TreeMesh
from discretize.utils import active_from_xyz

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

def create_mesh_from_survey(survey, base_cell_size=None, padding_configuration=[16,6,1,1,1,1,1], height_buffer_percentage=50.0):
    '''
    INPUTS
    survey: SimPEG survey object.
    base_cell_size: Float. Minimum cell size in meters. Default is 1/4 of the average horizontal electrode spacing.
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

    if base_cell_size is None:
        base_cell_size = np.mean(np.diff(electrode_coordinates[:,0]))/4
        print(f'Base cell size = {base_cell_size} m')
    
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
    center=np.median(electrode_coordinates[:,0])
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

def plot_mesh_and_survey(mesh, survey, full=False, buffer=5.0, vertical_exaggeration=1.0, ax=None):
    '''
    INPUTS
    mesh: SimPEG Mesh object.
    survey: SimPEG survey object.
    full: Boolean. Set to True to plot the whole mesh. Set to False and use buffer to plot around the survey set-up.
        Default is False.
    buffer: Float that defines the buffer, in meters, to plot around the highest electrode, lowest pseudo-location,
        and leftmost and rightmost electrodes. Dfault is 5 meters.
    vertical_exaggeration: Float. Default is 1.0.
    ax: Matplotlib figure Axes object.

    RETURNS
    ax: Matplotlib figure Axes object containing the plot of the mesh with the survey set-up overlain on top of it.
    '''

    # Extract survey electrodes and pseudo-locations
    electrode_coordinates = survey.unique_electrode_locations
    survey_pseudo_locations = pseudo_locations(survey)

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,4))

    # Plot mesh
    mesh.plot_grid(ax=ax, linewidth = 1, zorder = 1)

    # Overlay topography, electrodes, and pseudo-locations
    # Plot topography as blue line
    ax.plot(electrode_coordinates[:,0], electrode_coordinates[:,1], color="b", linewidth=2, label='surface', zorder=2)
    # Plot electrodes as red dots
    ax.scatter(electrode_coordinates[:,0], electrode_coordinates[:,1], 20, "r", label='electrodes')
    # Plot pseudo-locations as blue dots
    ax.scatter(survey_pseudo_locations[:,0], survey_pseudo_locations[:,1], 8, "b", label='pseudo-locations')
    # Set vertical exaggeration
    ax.set_aspect(vertical_exaggeration, adjustable='box')

    # Set x and y limits if full=False
    if full is False:
        xlim_max = electrode_coordinates[:,0].max()
        xlim_min = electrode_coordinates[:,0].min()
        ylim_max = electrode_coordinates[:,1].max()
        ylim_min = survey_pseudo_locations[:,1].min()
        ax.set_xlim(xlim_min-buffer, xlim_max+buffer)
        ax.set_ylim(ylim_min-buffer, ylim_max+buffer)

    # Other figure elements
    ax.grid(False)
    ax.set_ylabel('elevation [m]')
    ax.set_xlabel('x [m]')
    ax.set_title('Mesh and survey setup')
    ax.legend()

    return ax
