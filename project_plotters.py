import matplotlib as mpl
from matplotlib import pyplot as plt

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

def plot_mesh_and_survey(mesh, survey, full=False, buffer=5.0, vertical_exaggeration=1.0, ax=None):
    '''
    INPUTS
    mesh: SimPEG Mesh object.
    survey: SimPEG survey object.
    full: Boolean. Set to True to plot the whole mesh. Set to False and use buffer to plot around the survey set-up.
        Default is False.
    buffer: Float that defines the buffer, in meters, to plot around the highest electrode, lowest pseudo-location,
        and leftmost and rightmost electrodes. Default is 5 meters.
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

def plot_model_on_survey_and_mesh(mesh, logresistivity_model, plotting_map,
                                  survey=None, colormap_name='jet', title='Resistivity model',
                                  full=False, buffer=5.0, vertical_exaggeration=1,
                                  model_min=None, model_max=None, ax=None):
    '''
    INPUTS
    mesh: SimPEG Mesh object upon which to plot the model.
    logresistivity_model: Array of model parameters. The values are assumed to be in log-resistivity units.
    plotting_map: SimPEG Map object, used to hide the inactive cells of the mesh.
    survey: SimPEG Survey object. Optional; used to plot the surface topography, electrode positions, and pseudo-locations.
    colormap_name: String. The colormap to use for the log resistivity values. Default is jet.
    title: String. Figure title. Default is 'Log-resistivity model'.
    full: Boolean. Set to True to plot the whole mesh. Set to False and use buffer to plot around the survey set-up.
        Default is False.
    buffer: Float that defines the buffer, in meters, to plot around the highest electrode, lowest pseudo-location,
        and leftmost and rightmost electrodes. Default is 5 meters.
    vertical_exaggeration: Float. Default is 1.0.
    model_min: Float. Minimum model value for the colorbar, in terms of resistivity (NOT log resistivity).
    model_max: Float. Maximum model value for the colorbar, in terms of resistivity (NOT log resistivity).
    ax: Matplotlib figure Axes object.

    RETURNS
    ax: Matplotlib figure Axes object.
    '''
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(9,4))
    
    # If the minimum and maximum model values for the colorbar aren't given, take them from the provided model.
    if model_min is None:
        model_min = np.e**logresistivity_model.min()
    if model_max is None:
        model_max = np.e**logresistivity_model.max()

    # Map log of minimum and maximum resistivities to the colormap 0 to 1 range.
    norm = LogNorm(vmin=model_min, vmax=model_max)

    # Plot resistivity values on mesh
    mesh.plot_image(
        plotting_map * np.e**logresistivity_model,
        ax=ax,
        grid=False,
        pcolor_opts={"norm": norm, "cmap": colormap_name}
    )

    # Overlay mesh
    mesh.plot_grid(ax=ax, linewidth = 0.5, alpha=0.5, color='white')

    # If the survey is given, overlay topography, electrodes, and pseudo-locations.
    if survey is not None:
        # Extract survey electrodes and pseudo-locations
        electrode_coordinates = survey.unique_electrode_locations
        survey_pseudo_locations = pseudo_locations(survey)

        # Plot topography as blue line
        ax.plot(electrode_coordinates[:,0], electrode_coordinates[:,1], color="b", linewidth=2, label='surface', zorder=2)
        # Plot electrodes as red dots
        ax.scatter(electrode_coordinates[:,0], electrode_coordinates[:,1], 20, "r", label='electrodes')
        # Plot pseudo-locations as small grey dots
        ax.scatter(survey_pseudo_locations[:,0], survey_pseudo_locations[:,1], 1, "grey", label='pseudo-locations')

    # Set vertical exaggeration
    ax.set_aspect(vertical_exaggeration, adjustable='box')

    # Set x and y limits if full=False and a Survey object is given.
    if full is False and survey is not None:
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
    ax.set_title(title)
    
    # Add colorbar
    scalarMappable = plt.cm.ScalarMappable(cmap=colormap_name, norm=norm)
    scalarMappable.set_array([])
    fig = ax.figure
    cbar = fig.colorbar(scalarMappable, ax=ax)
    cbar.set_label(r"resistivity ($\Omega \cdot m$)", rotation=270, labelpad=15, size=12)

    return ax

def plot_normalized_volts_and_apparent_resistivity(data, survey, mesh=None, apparent_resistivity=None, colormap_name='jet',
                                                   title="", full=False, buffer=5.0, vertical_exaggeration=1.0,
                                                   ax=None):
    '''
    INPUTS
    data: Array. Data in the form of normalized voltages.
    survey: SimPEG Survey object.
    mesh: SimPEG Mesh object.
    apparent_resistivity: Array. Optional. Predicted data in the form of apparent resistivities. Calculated
        from dpred and Survey if not given.
    colormap_name: String. The colormap to use for the log resistivity values. Default is jet.
    title: String. Prefix to be used for each subplots' title.
    full: Boolean. Set to True to plot the whole mesh. Set to False and use buffer to plot around the survey set-up.
        Default is False.
    buffer: Float that defines the buffer, in meters, to plot around the highest electrode, lowest pseudo-location,
        and leftmost and rightmost electrodes. Default is 5 meters.
    vertical_exaggeration: Float. Default is 1.0.
    ax: Matplotlib figure Axes object with two subplots.
    
    RETURNS
    '''
    if apparent_resistivity is None:
        apparent_resistivity = apparent_resistivity_from_voltage(survey=survey, volts=data)

    if ax is None:
        fig, ax = plt.subplots(2,1, figsize=(8,8))

    # Extract survey electrodes and pseudo-locations
    electrode_coordinates = survey.unique_electrode_locations
    survey_pseudo_locations = pseudo_locations(survey)
    
    # TOP PLOT
    # Plot mesh, topography, and electrodes, or grid in the absence of given mesh.
    if mesh is not None:
        # Plot mesh as grey grid
        mesh.plot_grid(ax=ax[0], linewidth = 0.5, alpha=0.5, color='grey', zorder=1)
        ax[0].grid(False)
    else:
        ax[0].grid(True)

    # Plot surface topography as blue line
    ax[0].plot(electrode_coordinates[:,0], electrode_coordinates[:,1], color="b", linewidth=2, label='surface', zorder=2)
    # Plot electrodes as red dots
    ax[0].scatter(electrode_coordinates[:,0], electrode_coordinates[:,1], 20, "r", label='electrodes')

    plot_pseudosection(
        data=survey,
        dobs= data,
        plot_type="scatter",
        ax=ax[0],
        scale="log",
        cbar_label="Resistance [V/A]",
        scatter_opts={"cmap": colormap_name, "marker": 's', "label": 'pseudo-locations'}
    )
    # Set vertical exaggeration
    ax[0].set_aspect(vertical_exaggeration, adjustable='box')

    # Other figure elements
    ax[0].set_xlabel("")
    ax[0].set_ylabel("Pseudo-elevation [m]")
    ax[0].set_title(title+" Normalized Voltages")
    ax[0].legend()

    # BOTTOM PLOT 
    plot_pseudosection(
        survey,
        dobs= apparent_resistivity,
        plot_type="contourf",
        ax=ax[1],
        scale="log",
        cbar_label=r"Resistivity [$\Omega \cdot m$]",
        mask_topography=True,
        contourf_opts={"levels": 20, "cmap": colormap_name},
    )
    
    # Plot mesh, or grid in the absence of given mesh.
    if mesh is not None:
        # Plot mesh as grey grid
        mesh.plot_grid(ax=ax[1], linewidth = 0.5, alpha=0.5, color='grey', zorder=1)
        ax[1].grid(False)
    else:
        ax[1].grid(True)

    # Plot surface topography as blue line
    ax[1].plot(electrode_coordinates[:,0], electrode_coordinates[:,1], color="b", linewidth=2, label='surface', zorder=2)
    # Plot electrodes as red dots
    ax[1].scatter(electrode_coordinates[:,0], electrode_coordinates[:,1], 20, "r", label='electrodes')

    # Set vertical exaggeration
    ax[1].set_aspect(vertical_exaggeration, adjustable='box')

    # Other figure elements
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("Pseudo-elevation [m]")
    ax[1].set_title(title+" Apparent Resistivity")

    # Set x and y limits if full=False.
    if full is False:
        xlim_max = electrode_coordinates[:,0].max()
        xlim_min = electrode_coordinates[:,0].min()
        ylim_max = electrode_coordinates[:,1].max()
        ylim_min = survey_pseudo_locations[:,1].min()
        ax[0].set_xlim(xlim_min-buffer, xlim_max+buffer)
        ax[0].set_ylim(ylim_min-buffer, ylim_max+buffer)
        ax[1].set_xlim(xlim_min-buffer, xlim_max+buffer)
        ax[1].set_ylim(ylim_min-buffer, ylim_max+buffer)
    
    plt.tight_layout()
    return ax

def plot_initial_and_recovered_models(mesh, logresmodel_initial, logresmodel_recovered,
                                      plotting_map, survey, title,
                                      colormap_name='jet', full=False, buffer=5.0, vertical_exaggeration=1,
                                      model_min=None, model_max=None,
                                      ax=None):
    if ax is None:
        fig, ax = plt.subplots(2,1, figsize=(16,6))
    
    full_title_initial = 'Initial ' + title
    full_title_recovered = 'Recovered ' + title

    # Set minimum and maximum values for the colorbar.
    if model_min is None:
        model_min = min(np.e**logresmodel_initial.min(), np.e**logresmodel_recovered.min())
    if model_max is None:
        model_max = max(np.e**logresmodel_initial.max(), np.e**logresmodel_recovered.max())

    plot_model_on_survey_and_mesh(mesh=mesh, logresistivity_model=logresmodel_initial, plotting_map=plotting_map,
                              survey=survey, title=full_title_initial,
                              colormap_name=colormap_name, full=full, buffer=buffer, vertical_exaggeration=vertical_exaggeration,
                              model_min=model_min, model_max=model_max, ax=ax[0])
    plot_model_on_survey_and_mesh(mesh=mesh, logresistivity_model=logresmodel_recovered, plotting_map=plotting_map,
                              survey=survey, title=full_title_recovered,
                              colormap_name=colormap_name, full=full, buffer=buffer, vertical_exaggeration=vertical_exaggeration,
                              model_min=model_min, model_max=model_max, ax=ax[1])

    plt.tight_layout()
    return ax
