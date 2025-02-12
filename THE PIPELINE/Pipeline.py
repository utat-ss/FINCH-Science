'''
THE PIPELINE

This pipeline is crucial as it gives capability to do testing easily.

It is 

'''

#region imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

from scipy.optimize import minimize

from pymoo.core.problem import Problem

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#endregion 


#region Part 1: Data Seperation

'''
Jadon's code for Data Seperation Goes Here.

Requires:

Seperator(input_db (the input), seperation (other options than 60-40), abundance_thresh (thresh ), lambda_range):
    returns training_db, validation_db

'''

def separate_spectral_library(library_file, output_dir):
    #Open file
    try:
        df = pd.read_csv(library_file)
    except FileNotFoundError:
        print(f"File '{library_file}' not found.")
        return

    #Extract wavelength values from the header (assuming the first row is the header). They should be just the number (no units)
    try:
      wavelengths = [float(col) for col in df.columns[7:]] #DEPENDS on the input file (replace the "7" with the first numeric column)
    except ValueError:
        print("Look for the first numeric column and put that column number in df.columns[HERE:].")
        return

    valid_wavelengths_indices = [i for i, wl in enumerate(wavelengths) if 900 <= wl <= 1700] #change numbers here if you want to change the range of wavelengths

    if not valid_wavelengths_indices: 
        print("No wavelengths found within the 900-1700nm range.")
        return

    #Filter the DataFrame to include only the valid wavelengths
    df_filtered = df.iloc[:, [0] + [i+1 for i in valid_wavelengths_indices]] #Include the first column (sample names (if it's not, change the "0")) and then the valid wavelength columns

    #Rename columns with correct wavelength values
    new_columns = [df.columns[0]] + [wavelengths[i] for i in valid_wavelengths_indices]
    df_filtered.columns = new_columns

    for index, row in df_filtered.iterrows():
        spectrum_name = row.iloc[0] #Assumes first column is spectrum name/title (IF NOT, replace "0" with whichever column it is)
        spectrum_data = row.iloc[1:]  #Data (the 1 doesn't really matter because you're parsing through only the relevant wavelengths)

        #Create a DataFrame for each spectrum
        spectrum_df = pd.DataFrame({"Wavelength (nm)": spectrum_data.index, "Reflectance/Intensity": spectrum_data.values}) #change the table titles here
        spectrum_df = spectrum_df.set_index("Wavelength (nm)") #MAKE SURE THIS NAME IS THE SAME AS THE LINE ABOVE
        
        #output file
        output_filename = f"{output_dir}/{spectrum_name}.csv" #Use spectrum name as the filename

        try:
            spectrum_df.to_csv(output_filename)
            #print(f"Spectrum '{spectrum_name}' saved as {output_filename}") #uncomment out if you want like 300 confirmation messages
        except Exception as e:
            print(f"Error saving spectrum '{spectrum_name}': {e}")


if __name__ == "__main__":
    library_file = "C:\\Users\\Jadon\\Downloads\\original_data.csv"  #REPLACE with your own path (keep double backslashes)
    output_dir = "C:\\Users\\Jadon\\Downloads\\output_spectra"  #REPLACE with your own directory (keep double backslashes)
    #OPTIONAL!!!!! uncomment out if you wanna let it make the directory automatically
    # import os
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    separate_spectral_library(library_file, output_dir) #calling function




#region Definitions

#endregion

#endregion 

#region Part 2: Supervised Classification

#region Definitions

'''
This section includes the training process for the EM Optimizer algorithms that require it. Example: KL Div does not 
require it so it will just exist in Part 3, but KMC and KNN need it.
'''

def PCAnalysis(Df_Init: pd.DataFrame):

    '''
    Perform PCA to be used on KNN and KMC

    Parameters
        Df_Init (pd.DataFrame): Initial dataframe to perform PCA on

    Inter-Parameters
        i (int): Amount of Principal Components (PCs)
        n (int): How many columns to omit from the start
        k (int): How many columns to omit from the end
        plot_pca (bool): If True , plots the performance of PCAs

    Returns
        Df_PCA (pd.DataFrame): Output, PCA applied dataframe

    Aidan's Work, modified.
    '''

    i_input = int(input("How many PCs do you want? Default = 7 \n"))
    i = i_input if i_input else 7

    n_input = int(input("How many columns from the start do you want to omit? Default = 56 \n"))
    n = n_input if n_input else 56

    k_input = int(input("How many columns from the end do you want to omit? Default = 80 \n"))
    k = k_input if k_input else 80

    plot_pca_input = bool(input("Do you want to plot the PCs performance? Default = False"))
    plot_pca = plot_pca_input if plot_pca_input else False

    Df_Input_omitted = Df_Init.iloc[:,n:(len(Df_Init.columns)-k)]
    #Picks the necessary columns from the dataframe.

    scaling = StandardScaler()
    Df_Scaled = scaling.fit_transform(Df_Input_omitted)
    #Scales the dataframe.

    pca = PCA(n_components=i)
    Df_PostPca = pca.fit_transform(Df_Scaled)
    Df_PCA = pd.DataFrame(Df_PostPca)
    #Applies PCA on the dataframe.

    if plot_pca == True:

        x_data = range(1,i+1)
        y_data = pca.explained_variance_ratio_.cumsum()

        plt.plot(x_data, y_data, 'b-', label = "Var Ratio w.r.t. PC#")
        plt.xlabel("PC#")
        plt.ylabel("Var Ratio")
        plt.title("Var Ratio w.r.t. PC#")
        plt.show()

    return Df_PCA #returns post-PCA dataframe.

#implement VarThresh here as a pre-classification method as an alternative to PCAnalysis

def KNN_Fit(Df_PCA: pd.DataFrame, Df_Init: pd.DataFrame, k: int, thresh: int, ):

    '''
    Does KNN on PCA data and predicts the classes

    Parameters
        Df_PCA (pd.DataFrame): Post-PCA training dataframe
        Df_Init (pd.DataFrame): Pre-PCA training dataframe

    Inter-Parameters
        thresh (float): Threshold for classification division
        algorithm (str): Type of algorithm
        Various other Inter-Parameters, depending on the algorithm

    Returns
        Algo: KNN Algorithm
        Predicts (array): Predictions on Df_PCA using Algo
    '''

    algorithm_input = input("Enter the algorithm. Options: 'auto', 'ball_tree', 'kd_tree', 'brute' default = 'auto' \n")
    algorithm = algorithm_input if algorithm_input else 'auto'

    '''
    We first have to seperate our PCA performed data into training and to be fit. Doing this, we will be able
    to perform KNN on the same PCA'd data, doing KNN and predicting on different PCA'd data leads to weird results.
    Here, we will be creating these two PCA'd sets.
    '''

    #condition check
    if Df_Init.shape[0] == Df_PCA.shape[0]:
        init_df_shape=Df_Init.shape
    else: 
        raise Exception("PCA applied and initial dataframe are not the same")

    Thresh_Df = pd.DataFrame()
    #initialize the threshed PCA applied df

    threshed_index_list = []
    #initialize the list of rows that fulfill thresh

    thresh = float(input("What do you want to set your threshold as? Recommended: above .50\n"))

    #this collects all the row numbers that are bigger than the thresh. Using this, we will form the threshed PCA Df
    for i in range(init_df_shape[0]):

        if (float(Df_Init.iloc[i,1]) >= thresh) or (float(Df_Init.iloc[i,2]) >= thresh) or (float(Df_Init.iloc[i,3]) >= thresh):
            threshed_index_list.append(i)
            
    #this creates the dataframe of PCA applied EMs that also are above the thresh, 
    for i in threshed_index_list:
        Thresh_Df = pd.concat([Thresh_Df, Df_PCA.iloc[[i]]])
        
    #Now, we will set up the class_array of labels
    Threshed_shape = Thresh_Df.shape

    class_array= np.full(shape=(Threshed_shape[0],), fill_value = -1 ,dtype=int)

    #create a non-shaped array of equal length to the row number of initial data

    def Classify_as_GV():
        if float(Df_Init.iloc[i,1]) >= thresh: # 0
            class_array[counter] = 0
            #ie gv
    
    def Classify_as_NPV():
        if  float(Df_Init.iloc[i,2]) >= thresh: # 1
            class_array[counter] = 1
            #ie npv

    def Classify_as_Soil():
        if float(Df_Init.iloc[i,3]) >= thresh: # 2
            class_array[counter] = 2
            #ie soil

    counter = 0
    for i in threshed_index_list:
        #for all the entries in the initial dataset, we input if it is gv, npv, or soil

        options = [Classify_as_GV(), Classify_as_NPV(), Classify_as_Soil()]
        random.shuffle(options)

        for j in options:
            j()

        #with this complete, we now have the initial classification array

        counter = counter+1

    if algorithm == 'auto':

        k_input = int(input("How many neighbors do you want? Recommended: <sqrt(n), n being the amount of data, Default: " + str((math.sqrt(Threshed_shape[0]))//1)) +"\n")
        k = k_input if k_input else (math.sqrt(Threshed_shape[0]))//1

        weights_input = str(input("What do you want the weights to be? Options: 'distance', 'uniform'. Default: 'distance'. \n ")) 
        weights = weights_input if weights_input else 'distance'

        p_entry = int(input("What is the power parameter? Default is 2, Euclidian \n"))
        p = p_entry if p_entry else 2

        Algo = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, p=p)

    if algorithm == 'ball_tree':

        leaf_size_entry = int(input("what leaf size do you want? Default is 10 \n"))
        leaf_size = leaf_size_entry if leaf_size_entry else 10

        p_entry = int(input("What is the power parameter? Default is 2, Euclidian \n"))
        p = p_entry if p_entry else 2

        Algo = KNeighborsClassifier(algorithm=algorithm, X=Df_PCA.shape, leaf_size=leaf_size, p=p)

    if algorithm == 'kd_tree':

        leaf_size_entry = int(input("what leaf size do you want? Default is 10 \n"))
        leaf_size = leaf_size_entry if leaf_size_entry else 10

        p_entry = int(input("What is the power parameter? Default is 2, Euclidian \n"))
        p = p_entry if p_entry else 2

        Algo = KNeighborsClassifier(algorithm=algorithm, X=Df_PCA.shape, leaf_size=leaf_size, p=p)
    
    if algorithm == 'brute': #please don't. Actually, it might be a good idea

        k_input = int(input("How many neighbors do you want? Recommended: <sqrt(n), n being the amount of data, Default: " + str((math.sqrt(Threshed_shape[0]))//1)) +"\n")
        k = k_input if k_input else (math.sqrt(Threshed_shape[0]))//1

        weights_input = str(input("What do you want the weights to be? Options: 'distance', 'uniform'. Default: 'distance'. \n ")) 
        weights = weights_input if weights_input else 'distance'

        p_entry = int(input("What is the power parameter? Default is 2, Euclidian \n"))
        p = p_entry if p_entry else 2

        Algo = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, p=p)


    print(str(class_array.shape) + " of Spectra have been used for classification. \n")

    Algo.fit(Thresh_Df,class_array)
    #using the Post-PCA array and the initial classification array, train a model so that we can use it to predict on new data

    Predicts = Algo.predict(Df_PCA)

    return Algo, Predicts #return the algorithm, use this to further predict some more data, and calculate the distances

def KMC_Fit(Df_PCA: pd.DataFrame, n_clusters: int = 3):

    '''
    Does KMC on PCA data, predicts clusters and cluster centers

    Parameters
        Df_PCA (pd.DataFrame): Post-PCA training dataframe
        n_clusters (int): Number of clusters, default is 3, do not need to enter

    Inter-Parameters
        algorithm (str): Algorithm for KMC

    Returns
        Algo: The algo
        Cluster_Centers (np.array): Array including the cluster centers
        Predicts (np.array): Array (n,) of the possibly systematically wrong labelled predictions
    '''

    algorithm_input = input("Enter the algorithm. Options: 'lloyd', 'elkan' default = 'lloyd' \n")
    algorithm = algorithm_input if algorithm_input else 'lloyd'

    Algo = KMeans(n_clusters=n_clusters, algorithm=algorithm).fit(Df_PCA)
    Cluster_Centers = Algo.cluster_centers_
    Predicts = Algo.labels_

    return Algo, Cluster_Centers, Predicts

#endregion

#endregion

#region Part 3: EM Optimization

#region Definitions

def Collect_Class(Df: pd.DataFrame, Predicts: np.array, class_: int): 

    """
    Returns the dataframe only belonging to a single class

    Parameters
        Df (pd.DataFrame): Post/Pre-PCA training dataframe
        Predicts (np.array): Post-Prediction classification array
        class_ (int): Class to collect; 0 (gv), 1 (npv), 2 (soil)

    Returns
        Df_Class (pd.DataFrame): Dataframe of sifted classes
    """

    target_rows = np.where(Predicts == class_)[0]
    Df_Class = Df.iloc[target_rows]

    return Df_Class

def Correct_Class_KMC(Df_PCA: pd.DataFrame, Df_Init: pd.DataFrame, cluster_centers: np.array, labels: np.array):

    """
    This function corrects the classes and centers of KMC. The output of KMC shuffles the classes which needs fixing, this function fixes the class labels.

    Parameters
        Df_PCA (pd.DataFrame): Post-PCA training dataframe
        Df_Init (pd.DataFrame): Pre-PCA training dataframe
        cluster_centers (np.array): The cluster centers
        labels (np.array): Labels of each row of what they are

    Returns
        cluster_centers_index (np.array): The indices of cluster centers, 0:gv, 1:npv, 2:soil
        Predicts (np.array): The correct labels (predictions) of each row
    """

    #We first need to find the rows cluster centers are in, we do this by looping through the cluster_centers array and the post PCA dataset.
    
    index_list = []

    for i in cluster_centers.shape[0]:

        for j in Df_PCA:

            if cluster_centers[i,] == j:

                index_list.append(i)

    #This loop collects the indices for all the determined cluster centers

    #Now, we will correct the labels array. To do this, we will sum up the abundances of each class' rows, then rename the labellings.

    average_matrix = np.zeros(shape=(3,3))

    Df_Intermediary = Df_Init.drop(Df_Init.columns.difference(['gv_fraction','npv_fraction','soil_fraction']),1,inplace=True)
    #We first form the dataframe of fractions only
    
    for i in range(3):
        #For a label,
        bufferlist = []
        #We form a buffer list to store the rows of this label

        for j in labels:
            if i == j:
                #if j in labels is equal to the label, store it in the bufferlist
                bufferlist.append(j)

        #now, we will sum up its gv, npv, soil abundance columns, to do this, we need to define an intermediary df

        #for each column in the intermediary dataframe
        for k in range(3):
            
            #find the average of abundances of the rows in the bufferlist
            average_matrix[i,k] = (Df_Intermediary.iloc[bufferlist, k].sum())/(len(bufferlist))
            #append it to the ith row and kth column, so rows are the labels, columns are the averages of abundances
 
    
    #Now that we have formed the matrix of averages, we will go through to actually find which labels mean which EM

    label_dictionary = {}

    for i in range(3):
        #starting from one of the label rows

        j = np.argmax(average_matrix[i])
        #finds the column with the highest average, which is our true label (most possibly)

        label_dictionary[i] = j
        #append this to the dictionary
    
    #We will now update all of the values in our labels matrix

    Predicts = np.zeros(shape=(labels.shape[0],))

    for i in range(labels.shape[0]):
        #with this, we change all of the false labels with true labels using the relation dictionary
        Predicts[i] = label_dictionary[labels[i]]

    cluster_centers_index = []

    for i in range(3):
        for j in index_list:
            if i == Predicts[j]:
                cluster_centers_index.append[j]

    return cluster_centers_index, Predicts

def Find_mrEMs_KNN_or_KMC(Df_Init: pd.DataFrame, Df_PCA: pd.DataFrame, Predicts: np.array):

    """
    Tries to find the centroids, most representative EMs (mrEM) post KNN or KMC. For KMC, it is an alternative to KMC.cluster.centers_

    Parameters
        Df_Init (pd.DataFrame): Pre-PCA training dataframe
        Df_PCA (pd.DataFrame): Post-PCA training dataframe
        Predicts (np.array): Post-Prediction classification array

    Returns 
        mrEM_Indices (list): The indices of the EMs, [0]: gv, [1]: npv, [2]: soil
        Df_mrEM (pd.DataFrame): The dataframe of the endmembers, [0]: gv, [1]: npv, [2]: soil
    """

    algorithm_input = input("What method do you want to use? Options: 'centroid' (default), other options wip")
    algorithm = algorithm_input if algorithm_input else 'centroid'

    if algorithm == 'centroid':

        def Find_Centroid(Df_PCA: pd.DataFrame, Predicts: np.array, class_: int):

            """
            Intermediary function, finds the EM closest to the centroid determined by euclidian distance of that class, using PCA data
            
            Parameters
                Df_PCA (pd.DataFrame): Post-PCA training dataframe
                Predicts (np.array): Post-Prediction classification array
                class_ (int): Which class to find the centroid; [0]: gv, [1]: npv, [2]: soil
            
            Returns
                Representative_EM_index (int): The index of the mrEM in that class
                Representative_EM (pd.DataFrame): Dataframe of the single mrEM
                Centroid (np.array): The centroid matrix (single row, n columns)
                Distance_Collection (np.array): Collection of distances from the elements of class to the centroid by euc. dist.
            """

            Df_Class = Collect_Class(Df_PCA=Df_PCA, Predicts=Predicts, class_=class_)

            centroid = np.mean(Df_Class, axis=0)
            #Finds the centre point along each row (axis=0) which gives the centroid

            Distance_Collection = np.linalg.norm(Df_Class - centroid, axis =1)
            #Collects the distances to the centroid

            Representative_EM_index = np.argmin(Distance_Collection)
            #finds the index

            Df_Class_noPCA = Collect_Class(Df = Df_Init, Predicts=Predicts, class_=class_)

            Representative_EM = Df_Class_noPCA.iloc[Representative_EM_index]
            #finally chooses the EM from the non-PCA applied data

            return Representative_EM_index, Representative_EM, centroid, Distance_Collection
        
        mrEM_Indices = []
        Df_mrEMs = pd.DataFrame()

        for i in range(3):

            EM_index_buffer, Df_mrEM,_,_ = Find_Centroid(Df_PCA=Df_PCA, Predicts=Predicts, class_=i)

            mrEM_Indices.append(EM_index_buffer)
            Df_mrEMs = pd.concat([Df_mrEMs, Df_mrEM], ignore_index=True)

    return mrEM_Indices, Df_mrEMs


    #if further methods are wanted, define them here


#endregion

#endregion

#region Part 4: Perform Abundance Estimation

#region Definitions

def GradientDescent_SOO_RMSE(Df_Init: pd.DataFrame, Df_mrEMs: pd.DataFrame, plot_results: bool = False):
    """
    Estimate fractional abundances with sum-to-one and greater-than-zero constraints using a linear spectral mixing model.

    Parameters:
        Df_Init (pd.DataFrame): A DataFrame where each row represents a spectral measurement
                            and each column represents a wavelength or spectral band.
        Df_mrEMs (pd.DataFrame): A DataFrame where each row represents an endmember spectrum
                            [0]: GV, [1]: NPV, [2]: Soil
                            and each column represents the corresponding wavelength or spectral band.

    Returns:
        Df_Abundances (pd.DataFrame): A DataFrame with estimated fractional abundances for each spectrum
        overall_rmse (float): The overall RMSE between the original spectra and the reconstructed spectra
        individual_rmse_series (pd.Series): A Series containing the RMSE for each individual spectrum

    
    This function performs sequential optimization (gradient descent) for each spectrum using the RMSE of the entire spectrum of the difference in original by reconstructed. 

    Aidan's work, modified by Ege to fit the nomenclature of THE PIPELINE.
    """
    def objective(fractions, spectrum, endmembers):
        # Objective function: Mean squared error for a single spectrum
        reconstructed_spectrum = np.dot(fractions, endmembers)
        mse = np.mean((spectrum - reconstructed_spectrum) ** 2)
        return mse
    '''
    The objective function provides the optimizer (result = minimize) with a measure of how far the guessed spectra is from the inputted spectra.
    It does this by minimizing the MSE between the two. 
    
    First the fractions array is reshaped so that its dimensions are combatible with the number of endmembers defined.
    The index of -1 tells numpy to automatically reshape the array. Which in this case is the number of spectra multiplied by the number of endmembers.
    Therefore you can optimize the selection of endmembers based on many sample mixed spectrums and have the dot product operation be valid.
    The dot product is what actually computes the estimates (reconstructed) spectrum to fit the true spectrum. 
    This comes from the assumption that the mixture is linear and thus a weighted sum of the sprectra in an endmember by their abundances.
    
    The MSE is then calculated to feed back into the optimization.
    
    '''

    def constraint_sum_to_one(fractions):
        # Constraint: sum of fractions should be 1
        return np.sum(fractions) - 1
    
    def constraint_greater_than_zero(fractions):
        # Constraint: fractions should be greater than zero
        return fractions - epsilon
    
    '''
    The constraints are necessary for the optimization process so that the optimized abundances make physical sense.
    The contribution of an endmember is  a spectrum cannot be negative.
    Furthermore, all endmembers in a scene should contribute to the reconstructed spectra and the total abundances must equal 100%.
    
    The sum to one constraint function works by checking to see if the residual between the sum of the fractional abundaces and 1 is equal to 0.
    The result is passed on to the minimize operator which has an 'eq' clause in the constraints parameter. This means only solutions where the sum_to_one function passes a 0 are valid solutions.
    
    The greater_than_zero function works by subtracting epsilon, which is an arbitrarily small number, from the estimated fraction.
    If the value is negative it, the corresponding solution will be rejected in the minimize operator via the "ineq" clause in the constrainst parameter.
    '''
    
    # Convert DataFrames to NumPy arrays for matrix operations
    spectra_array = Df_Init.values
    endmembers_array = Df_mrEMs.values
    num_spectra = spectra_array.shape[0]
    num_endmembers = endmembers_array.shape[0]

    # Epsilon to ensure strictly positive values
    epsilon = 1e-6
    
    # Store the fractional abundances and RMSE for each spectrum
    all_fractions = np.zeros((num_spectra, num_endmembers))
    individual_rmse = []

    # Loop through each spectrum and solve the optimization problem individually
    for i in range(num_spectra):
        spectrum = spectra_array[i, :]
        
        # Initial guess: Equal fractions for the current spectrum
        initial_guess = np.ones(num_endmembers) / num_endmembers
        
        '''
        The initial guess just makes the contributions of all endmembers equal for each band in the spectrum.
        This is a good starting point for the minimize operator since it satisfies all constraints and can be any size depending on the number if endmember and spectra to fit for.
        '''

        # Constraints for the current spectrum
        constraints = [{'type': 'eq', 'fun': constraint_sum_to_one},
                       {'type': 'ineq', 'fun': constraint_greater_than_zero}]
        
        '''
        How to enforce the constraints are defined above and are given as parameters in the minimize operator.
        '''

        # Optimization for the current spectrum
        result = minimize(objective, initial_guess, args=(spectrum, endmembers_array),
                          constraints=constraints, method='SLSQP', options={'disp': False})
        
        '''
        This minimize operator is the core of the function and uses the scipy.optimize library.
        In this function we want to find the fractional abundances that minimize the RMSE between the observed and reconstructed spectra using n endmembers.
        
        The objective is the function we want to minimize which is described above.
        
        The initial guess gives us the starting point for optimization described above
        
        The args are passed to the objective function and come from the spectra and endmembers pd.Dataframes which are supplied in the LMM_FAE parameters.
       
        The constraints are given.
        
        The Sequential Least Squares Programming (method = SLSQP) iterively checks combinations of fractional abundances and chooses new abundances.
        This is basically a gradient descent which allows for constraints to be placed so that non physically possible variables are not chosen (such as negative fractional abundances).
        This way, we are not randomly changing the fill fractions but instead are calculating the gradient across nearby combinations of fractions.
        The "topography" of the RMSE space is shown through the gradient and the fractions which correspond to the largest change in slope minimizing the RMSE are chosen for the next solution option.
        The gradient itself is calculated numerically through the finite difference method.
        The numeric solution is calculated whilst considering the constraints.
        
        
        options={'disp': False} does not print each step of the minimization process.
        '''

        # Extract the optimized fractions for the current spectrum
        fractions = result.x
        all_fractions[i, :] = fractions

        # Reconstruct the spectrum and compute RMSE
        reconstructed_spectrum = np.dot(fractions, endmembers_array)
        mse = np.mean((spectrum - reconstructed_spectrum) ** 2)
        rmse = np.sqrt(mse)
        individual_rmse.append(rmse)
        
        # Optional plotting
        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(Df_Init.columns, spectrum, label='True Spectrum', color='blue')
            plt.plot(Df_Init.columns, reconstructed_spectrum, label='Reconstructed Spectrum', color='red', linestyle='--')
            plt.title(f'Spectrum {i+1}: True vs Reconstructed')
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            plt.legend()
            plt.show()
    
    # Convert the final estimated fractions back to a DataFrame
    Df_Abundances = pd.DataFrame(all_fractions, index=Df_Init.index, columns=Df_mrEMs.index)
    
    # Create a Series for individual RMSE values
    individual_rmse_series = pd.Series(individual_rmse, index=Df_Init.index, name='RMSE')

    # Calculate the overall RMSE across all spectra
    overall_rmse = np.mean(individual_rmse)
    
    return Df_Abundances, overall_rmse, individual_rmse_series

def GradientDescent_SOO_SFA():


    return #Df_Abundances, overall_rmse, individual_rmse_series

def MOO_SFA():


    return #Df_Abundanes, overall_rmse, individual_rmse_series 




#endregion

#endregion

#region Part 5: Export Metrics

#region Definitions

#endregion

#endregion



print("Done")

