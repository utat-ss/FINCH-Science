import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

def plot_abundance_comparison(true_ab_df: pd.DataFrame, optimized_ab_df: pd.DataFrame, title: str = "Abundance Comparison", goalline: bool = True, best_fit: bool = False):
    """
    Creates a scatter plot comparing optimized abundance (y-axis) with true abundances (x-axis).
    """

    types = ['gv_fraction', 'npv_fraction', 'soil_fraction']
    colors = ['green', 'blue', 'brown']
    darkened_colors = ['darkgreen', 'darkblue', 'saddlebrown']
    # Create a scatter plot for each column (abundance type)

    fig, ax = plt.subplots(figsize=(8, 6))

    for column in optimized_ab_df.columns:
        ax.scatter(true_ab_df[column], optimized_ab_df[column], label=types[column], color=colors[column])

    if goalline:
        # Add a y=x line for the goal
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Goal Line (y=x)')

    if best_fit:
        for column in optimized_ab_df.columns:
            # Calculate the best fit line
            slope, intercept, r_value, p_value, std_err = linregress(true_ab_df[column], optimized_ab_df[column])
            r_value = r_value **2
            print(f"Best fit line for {types[column]}: slope = {slope}, intercept = {intercept}, r^2 = {r_value**2}")
            ax.plot(true_ab_df[column], slope * (true_ab_df[column]) + intercept, color=darkened_colors[column], linestyle='--', label=f'Best Fit {types[column]}')
    
    ax.set_ylabel('Optimized Abundance')
    ax.set_xlabel('True Abundance')
    ax.set_title(title)

    plt.show()

def plot3_abundance_comparison(ax, true_ab_df: pd.DataFrame, optimized_ab_df: pd.DataFrame,title: str = "Abundance Comparison", goalline: bool = True, best_fit: bool = False):
    """
    Creates a scatter plot comparing optimized abundance (y-axis) with true abundances (x-axis).
    
    Note: if using multiplot, ax should be passed as an argument, otherwise it will create a new figure and axis.
    And fig, ax = plt.subplots(1, 3, figsize=(15, 5)) must be used to create the subplots.
    """

    types = ['gv_fraction', 'npv_fraction', 'soil_fraction']
    colors = ['green', 'blue', 'brown']
    darkened_colors = ['darkgreen', 'darkblue', 'saddlebrown']
    
    for column in optimized_ab_df.columns:
        ax.scatter(true_ab_df[column], optimized_ab_df[column], label=types[column], color=colors[column])

    if goalline:
        # Add a y=x line for the goal
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Goal Line (y=x)')

    if best_fit:
        for column in optimized_ab_df.columns:
            
            # Calculate the best fit line
            slope, intercept, r_value, p_value, std_err = linregress(true_ab_df[column], optimized_ab_df[column])
            r_value = r_value **2
            print(f"Best fit line for {types[column]}: slope = {slope}, intercept = {intercept}, r^2 = {r_value**2}")
            ax.plot(true_ab_df[column], slope * (true_ab_df[column]) + intercept, color=darkened_colors[column], linestyle='--', label=f'Best Fit {types[column]}')
    
    ax.set_ylabel('Optimized Abundance')
    ax.set_xlabel('True Abundance')
    ax.set_title(title)

def plot_single_abundance_comparison(abundance_type: int, true_ab_df: pd.DataFrame, optimized_ab_df: pd.DataFrame, title: str = "Abundance Comparison", goalline: bool = True, best_fit: bool = True):
    """
    Creates a scatter plot comparing optimized abundance (y-axis) with true abundances (x-axis).
    """

    types = ['gv_fraction', 'npv_fraction', 'soil_fraction']
    colors = ['green', 'blue', 'brown']
    
    # Create a scatter plot for each column (abundance type)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(optimized_ab_df[abundance_type], true_ab_df[abundance_type], label=types[abundance_type], color=colors[abundance_type])

    if goalline:
        # Add a y=x line for the goal
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Goal Line (y=x)')

    if best_fit:
        # Calculate the best fit line
        slope, intercept, r_value, p_value, std_err = linregress(true_ab_df[abundance_type], optimized_ab_df[abundance_type])
        r_value = r_value **2
        print(f"Best fit line for {types[abundance_type]}: slope = {slope}, intercept = {intercept}, r^2 = {r_value**2}") 
        ax.plot(true_ab_df[abundance_type], slope * (true_ab_df[abundance_type]) + intercept, color='magenta', linestyle='--', label=f'Best Fit {types[abundance_type]}')

        return slope, intercept, r_value, p_value, std_err

    
    ax.set_xlabel('Optimized Abundance')
    ax.set_ylabel('True Abundance')
    ax.set_title(title)

    plt.show()

def plot3_single_abundance_comparison(ax, abundance_type: int, true_ab_df: pd.DataFrame, optimized_ab_df: pd.DataFrame, title: str = "Abundance Comparison", goalline: bool = True, best_fit: bool = True):
    """
    Creates a scatter plot comparing optimized abundance (y-axis) with true abundances (x-axis).
    """
    types = ['gv_fraction', 'npv_fraction', 'soil_fraction']
    colors = ['green', 'blue', 'brown']
    
    ax.scatter(optimized_ab_df[abundance_type], true_ab_df[abundance_type], label=types[abundance_type], color=colors[abundance_type])

    if goalline:
        # Add a y=x line for the goal
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Goal Line (y=x)')

    if best_fit:
        # Calculate the best fit line
        slope, intercept, r_value, p_value, std_err = linregress(true_ab_df[abundance_type], optimized_ab_df[abundance_type])
        r_value = r_value **2

        ax.plot(true_ab_df[abundance_type], slope * (true_ab_df[abundance_type]) + intercept, color='magenta', linestyle='--', label=f'Best Fit {types[abundance_type]}')

        return r_value, p_value, std_err, slope, intercept, 
    
    ax.set_ylabel('Optimized Abundance')
    ax.set_xlabel('True Abundance')
    ax.set_title(title)

