import numpy as np
from PIL import Image
import re
import pandas as pd
import pyro
from sklearn.preprocessing import LabelEncoder
import torch
import matplotlib.pyplot as plt


DECIPHER_GLOBALS = dict()
DECIPHER_GLOBALS["save_folder"] = "./_decipher_models/"


def create_decipher_uns_key(adata):
    """
    Create the `decipher` uns key if it doesn't exist.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    """
    if "decipher" not in adata.uns:
        adata.uns["decipher"] = dict()
    if "trajectories" not in adata.uns["decipher"]:
        adata.uns["decipher"]["trajectories"] = dict()


def is_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook.

    Returns
    -------
    bool
        True if running in a Jupyter notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class GIFMaker:
    """Make a GIF from a list of images."""

    def __init__(self, dpi=100):
        self.images = []
        self.dpi = dpi

    def add_image(self, fig):
        """Add an image to the GIF.

        Parameters
        ----------
        fig : matplotlib.pyplot.figure
            The figure to add.
        """
        fig.set_dpi(self.dpi)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.images.append(Image.fromarray(image))

    def save_gif(self, filename):
        """Make and save a GIF from the images.

        Parameters
        ----------
        filename : str
            The filename to save the GIF to. Add `.gif` if not present.
        """
        images = self.images
        if not filename.endswith(".gif"):
            filename += ".gif"

        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            loop=0,
        )


def load_and_show_gif(filename):
    """Load and show a GIF in a Jupyter notebook.

    Parameters
    ----------
    filename : str
        The filename of the GIF.
    """
    from IPython.display import Image, display

    with open(filename, "rb") as f:
        display(Image(data=f.read(), format="png"))


def display_cat(lut, figsize=(3, 4), vertical=True, title="", title_x_offset=-4):
    """
    Display a color-coded categorical legend using matplotlib.

    Parameters:
    ----------
    lut : dict
        A lookup table (LUT) where keys represent category labels, and values are colors for each category.
    figsize : tuple, optional
        A tuple defining the figure size in inches (width, height). Default is (3, 4).
    vertical : bool, optional
        If True, display the color boxes in a vertical layout; if False, display them horizontally.
        Default is True (vertical layout).
    title : str, optional
        Title for the legend. Default is an empty string.
    title_x_offset : float, optional
        Offset for the x position of the title. Helps control the placement of the title. Default is -4.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object of the plot, useful for further customization.
    """
    
    # Number of color boxes (categories)
    n_boxes = len(lut)
    
    labels = list(lut.keys())  # Extract labels for each category
    
    # Create the figure and axis with the given size
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_aspect('equal')  # Set aspect ratio to equal for uniform box sizes
    ax.set_title(title, loc="left", x=title_x_offset)  # Set the title with left alignment

    if vertical:
        # Initialize y position for the first box
        start_y = 0
        
        # Create color boxes in a vertical layout
        for key in labels:
            ax.add_patch(plt.Rectangle((0, start_y), 1, 1, facecolor=lut[key]))  # Draw a rectangle for each category
            start_y += 1.5  # Move y position down for the next box
        
        # Set y-axis limits to fit all the boxes
        ylim = [-0.5, 1.5 * len(lut)]
        ax.set_ylim(ylim)
        ax.set_xlim([0, 1])

        # Create custom y-tick positions and labels
        yticks = np.arange(0.5, 0.5 + len(lut) * 1.5, 1.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(labels)
        
        # Hide unnecessary axis elements for a cleaner look
        ax.tick_params(axis='y', which='both', bottom=False, top=False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([])
        ax.set_xticklabels([])

    else:
        # Initialize x position for the first box
        start_x = 0

        # Create color boxes in a horizontal layout
        for key in labels:
            ax.add_patch(plt.Rectangle((start_x, 0), 1, 1, facecolor=lut[key]))  # Draw a rectangle for each category
            start_x += 1.5  # Move x position right for the next box
            
        # Set x-axis limits to fit all the boxes
        xlim = [-0.5, 1.5 * len(lut)]
        ax.set_xlim(xlim)
        ax.set_ylim([0, 1])

        # Create custom x-tick positions and labels
        xticks = np.arange(0.5, 0.5 + len(lut) * 1.5, 1.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, rotation=90)  # Rotate labels for readability
        
        # Hide unnecessary axis elements for a cleaner look
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Close the plot and return the axes for further customization
    plt.close()
    return ax


def clear_val_params():
    """
    Remove all parameters from Pyro's parameter store that contain the substring 'val' in their names.

    This function is useful when you want to clear validation-specific parameters from the parameter
    store, typically after or before running a validation phase in a machine learning pipeline using Pyro.

    The function identifies all parameter keys containing 'val' and deletes them from the parameter store.

    Example usage:
    --------------
    clear_val_params()

    Notes:
    ------
    This function modifies the global parameter store in Pyro, so use with caution if other parameters
    depend on those containing 'val' in their name.
    """
    
    # Find all parameter keys in Pyro's parameter store that contain the substring "val"
    keys_to_delete = [key for key in pyro.get_param_store().keys() if "val" in key]
    
    # Delete each of the identified parameters from the parameter store
    for key in keys_to_delete:
        del pyro.get_param_store()[key]
        

def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "values": v,
        }
    return site_stats

#Unfinished Converting to a function
# data = pd.read_csv("/Users/pressm/Documents/AziziLab/GVHD/immunoseq_r/AllClones.csv", low_memory=False)
def process_data(data, meta_data, last_timepoint = 180, n_filter = 2):
    """
    Process the data to create a sparse tensor representation of clone patterns.
    Parameters:
    data : pd.DataFrame
        DataFrame containing the clone data with columns for frequency, templates, timepoint, PatientID, cdr3_amino_acid, v_resolved, and j_resolved.
    meta_data : list
        List of additional metadata columns to include in the output DataFrame.
    last_timepoint : int, optional
        The last timepoint to consider in the data. Default is 180.
    n_filter : int, optional
        The minimum number of non-zero entries required for a clone to be included in the final tensor. Default is 2.
    Returns:
    clone_patterns_dense_filtered : torch.Tensor
        A dense tensor representation of the clone patterns, filtered based on the specified criteria.
    data : pd.DataFrame
        The processed DataFrame containing clone information, including frequency, templates, timepoint, tag, PatientID, cdr3_amino_acid, v_resolved, j_resolved, and clone_encoding.
    filter_idx : np.ndarray
        A boolean array indicating which clones passed the filtering criteria.
    """
    data["tag"] = data["PatientID"]+"_"+data["cdr3_amino_acid"]+data["v_resolved"]+data["j_resolved"]
    data = data[["frequency", "templates", "timepoint", "tag","PatientID", "cdr3_amino_acid", "v_resolved", "j_resolved"] + meta_data]
    data = data[data.loc[:, "timepoint"].apply(lambda x:x.isnumeric())]
    data.loc[:, "timepoint"] = data["timepoint"].astype(int)
    data = data[data["timepoint"] <=last_timepoint]
    clone_encoder = LabelEncoder().fit(data["tag"])
    clone_encoding = clone_encoder.transform(data["tag"])
    data["clone_encoding"] = clone_encoding
    data["time_encoding"] = data["timepoint"]
    data = data.drop_duplicates(subset=["clone_encoding", "timepoint", "frequency"])
    data = data.dropna(subset = ["tag"], axis = 0)
    data.loc[data["frequency"] == 0,"frequency"] = -1
    
    indices = torch.tensor([data["clone_encoding"].values, data["time_encoding"].values], dtype=torch.int64)
    values = torch.tensor(data["frequency"].values, dtype=torch.float64)
    clone_patterns_sparse = torch.sparse_coo_tensor(indices, values,dtype=torch.float64)
    clone_patterns_dense = clone_patterns_sparse.to_dense()
    filter_idx = ((clone_patterns_dense > 0).sum(axis = 1) >=n_filter)
    clone_patterns_dense_filtered = clone_patterns_dense[filter_idx,:]
    clone_pattern_max = np.expand_dims(np.max(clone_patterns_dense_filtered.numpy(), axis = 1), axis = 1)
    clone_patterns_dense_filtered[clone_patterns_dense_filtered==0] = float('nan')
    clone_patterns_dense_filtered[clone_patterns_dense_filtered < 0.0] = 0.0
    # clone_patterns_dense_filtered = torch.log(clone_patterns_dense_filtered + 1)
    clone_patterns_dense_filtered = clone_patterns_dense_filtered/clone_pattern_max
    
    return (clone_patterns_dense_filtered, data, filter_idx)


def find_sequence_rows_regex(df, sequence, column_name): 
    """
    Find rows in a DataFrame where a specific column matches a sequence pattern using regex.
    Parameters:
    df : pd.DataFrame
        The DataFrame to search within.
    sequence : str
        The sequence pattern to match. Use '%' as a wildcard for any character.
    column_name : str
        The name of the column in the DataFrame to search for the sequence pattern.
    Returns:
    pd.DataFrame
        A DataFrame containing rows where the specified column matches the sequence pattern.
    """
    # Convert the sequence pattern to regex
    regex_pattern = sequence.replace('%', '.')
    
    # Function to check if the pattern matches
    def has_match(seq):
        return bool(re.search(regex_pattern, seq))
    
    # Apply the function to the DataFrame
    matching_rows = df[df[column_name].apply(has_match)]
    return matching_rows


def clustermap(df_res:pd.DataFrame, df_cat:pd.DataFrame, col_categories:list, offset:list, legend_positions:list, legend_colors:list, legend_sizes:list, title:str):
    """
    Create a clustermap with colored rows based on categorical data.
    Parameters:
    df_res: pd.DataFrame
        DataFrame with the data to be plotted in the clustermap.
    df_cat: pd.DataFrame
        DataFrame with categorical data for coloring the rows. 
    col_categories: list
        List of column names in df_cat that will be used for coloring.
    offset: list
        List of x-offsets for the legends corresponding to each category.
    legend_positions: list
        List of positions for the legends corresponding to each category.
    legend_colors: list
        List of colors for the legends corresponding to each category.
    legend_sizes: list
        List of sizes for the legends corresponding to each category.
    title: str
        Title for the clustermap.
    Returns:
    ax: sns.clustermap
        The seaborn clustermap object.
    """
    
    
    row_colors = [] 
    luts = []
    #Generate a color mapping for each category in df_cat
    for idx, cat in enumerate(col_categories):
        lut = dict(zip(np.unique(df_cat[cat].astype(str).sort_values()), legend_colors[idx]))
        row_colors_temp = pd.Series(df_cat[cat].astype(str)).map(lut)
        row_colors.append(row_colors_temp)
        luts.append(lut)
        
    row_colors = pd.concat(row_colors, axis = 1).reset_index(drop = True)

    ax = sns.clustermap(df_res.T, col_colors = row_colors, xticklabels = df_res.columns.values + 1, dendrogram_ratio=0.25, figsize = (10,10), cmap = sns.color_palette("light:#19647E", as_cmap=True), row_cluster = False)
    ax.ax_col_dendrogram.set_title(title)
    legs = []
    for idx, lut in enumerate(luts):
        cur_leg = display_cat(lut, figsize = legend_sizes[idx],title = col_categories[idx], title_x_offset= offset[idx]) #Create legend displaing categories
        ax.ax_cbar.add_child_axes(cur_leg).set_position(legend_positions[idx])
    
    ax.ax_row_dendrogram.set_visible(False)
    ax.ax_col_dendrogram.set_visible(False)
    
    col_cb_ax = ax.ax_col_colors
    col_cb_ax.yaxis.set_label_position('left')
    ax.ax_heatmap.set_xticklabels([])
    ax.ax_heatmap.yaxis.set_ticks_position('left')
    ax.ax_col_colors.yaxis.tick_left()

    col_dendrogram = ax.dendrogram_col.calculated_linkage
    plt.show()
    
    return ax
