'''
This file is for the determination of optimal endmembers using KNN. To do this, we'll first 
reduce the dimensions of the data by using PCA. I have no idea how many dimensions we should
preserve, so I will go through dimensions of 10-30 and see how the variance changes, using this
we can determine how many dimensions we require. Additionally, this code will include a section
of performing KNN and KMC on the reduced data, from this data, we will choose the endmember that is the
most representative of each class using methods as: centroid, median, and density.

Author: Ege Artan, UTAT-Finch Science
'''

#region Imports

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

#endregion

#region Initial

input_bigdata = r"C:\University\science utat\endmember_perfect_1.csv" #input the data here
Df_BigData = pd.read_csv(input_bigdata)
input_bigdata = r"C:\University\science utat\endmember_not_perfect_0.99.csv" #input the data here
Df_BigDataImperfect = pd.read_csv(input_bigdata)

#endregion

#region Analysis Using PCA
'''here, we do analysis on how much PC's we need, this will return a plot of variance w.r.t
how many PC's we have.'''

def PCAnalysis(i, n, k, Df_BigData):
    '''i is the amount of PCs, n/k is how many first/last columns we are omitting, and the Df_BigData is the initial dataframe.
    Aidan's Work'''

    Df_BigData_nonames = Df_BigData.iloc[:,n:(len(Df_BigData.columns)-k)]

    scaling = StandardScaler()
    Df_scaled = scaling.fit_transform(Df_BigData_nonames)

    pca = PCA(n_components=i)
    Df_PostPca = pca.fit_transform(Df_scaled)

    Df_PCA = pd.DataFrame(Df_PostPca)

    return Df_PCA

def PCAnalysis_DetailedOutput(i, n, k, Df_BigData):
    '''i is the amount of PCs, n/k is how many first/last columns we are omitting, and the Df_BigData is the initial dataframe.
    Aidan's Work'''

    Df_BigData_nonames = Df_BigData.iloc[:,n:(len(Df_BigData.columns)-k)]

    scaling = StandardScaler()
    Df_scaled = scaling.fit_transform(Df_BigData_nonames)

    pca = PCA(n_components=i)
    Df_PostPca = pca.fit_transform(Df_scaled)

    x_data = range(1,i+1)
    y_data = pca.explained_variance_ratio_.cumsum()

    plt.plot(x_data, y_data, 'b-', label = "Var Ratio w.r.t. PC#")
    plt.xlabel("PC#")
    plt.ylabel("Var Ratio")
    plt.title("Var Ratio w.r.t. PC#")
    plt.show()

    Df_PCA = pd.DataFrame(Df_PostPca)
    outhandle = r"C:\University\science utat\PostPCA_EMs.csv"
    Df_PCA.to_csv(outhandle)

    return Df_PCA

def perform_pca_svd(df, n_components=None):
    """
    Perform PCA via SVD and return the variance, 
    the principal component scores, and the loadings. This also plots the singular values
    and cumulative singular values on a log scale.

    Parameters:
    - df (pd.DataFrame): The input dataframe with spectral measurements.
    - n_components (int): Number of principal components to compute. If None, all components are computed.

    Returns:
    - explained_variance (np.ndarray): The variance explained by each principal component.
    - principal_components (pd.DataFrame): The transformed data with principal component scores.
    - loadings (pd.DataFrame): The loadings of the original variables (wavelengths) on the principal components.

    Aidan's Work
    """

    #Taken from Aidan's Work
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(df_scaled, full_matrices=False)
    
    # Determine the number of components
    if n_components is None:
        n_components = min(df.shape[0], df.shape[1])
    
    # Select the top n_components
    U = U[:, :n_components]
    S = S[:n_components]
    Vt = Vt[:n_components, :]
    
    # Compute the explained variance
    explained_variance = (S**2) / (df.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    
    # Compute the principal component scores
    principal_components = np.dot(U, np.diag(S))
    principal_components_df = pd.DataFrame(principal_components, 
                                           columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    
    # Compute the loadings
    loadings = Vt.T
    loadings_df = pd.DataFrame(loadings, 
                               columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
                               index=df.columns)
    
    # Plot singular values and cumulative singular values
    plt.figure(figsize=(12, 6))
    
    # Plot singular values
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(S) + 1), S, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.xlabel('Component Number')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Singular Values')
    
    # Plot cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.xlabel('Component Number')
    plt.ylabel('Cumulative Explained Variance (log scale)')
    plt.title('Cumulative Explained Variance')
    
    plt.tight_layout()
    plt.show()
    
    return explained_variance_ratio, principal_components_df, loadings_df

'''
Found out that this does not work at all, the post PCA dataset has values all over the place, outside 0-1 range. I don't know yet
if there is a way to 'rescale' this new data for something that would resemble reflectance. Additionally, the newly formed PCA
are not defined to be specific wavelengths but rather new arbitrary 'wavelengths'. Chances are using PCA wouldn't make much sense.

On a different note, we can just retrieve the most contributing columns to the calculation of PCs of the original data. I will
attempt it below:
'''

def PC_ComponentWeight_Analysis(i, n, Df_BigData):
    '''i is the amount of PCs, n is how many first columns we are omitting, and the Df_BigData is the initial dataframe.'''

    Df_BigData_nonames = Df_BigData.iloc[:,n:]
    scaling = StandardScaler()
    Df_scaled = scaling.fit_transform(Df_BigData_nonames)

    pca = PCA(n_components=i)
    Df_PostPca = pca.fit_transform(Df_scaled)

    Df_PCA = pd.DataFrame(Df_PostPca)

    components = pca.components_

    Df_ComponentWeight = pd.DataFrame(components)

    outhandle = r"C:\University\science utat\ComponentWeights_EMs.csv"
    Df_ComponentWeight.to_csv(outhandle, index=True)

    return 

'''
Can't seem to get this to work, this is because it returns specific contributions values of each PC's wavelengths, i.e. in what way
each wavelength contributes to the PC. Couldn't make much sense of this. Tried doing this with 48 PCs (preserving everything) but
again, couldn't make much sense, it was way beyond me to be honest.
'''

#endregion
  
#region Analysis Using VarThresh

'''
Instead of using PCA, I'll attempt to formulate a way to reduce the data using a variance threshold. This code will attempt to 
calculate the variance across a single column (i.e. wavelength) and then reduce down the columns that do not meet a certain threshold.
I am predicting that this is kind of what we are looking for.
'''

def VarThresh_Analysis(thresh, n, Df_Bigstuff):
    '''Here, thresh is the threshold for variances, n is how many columns we remove 
    from beginning and Df_Bigstuff is the input dataframe.'''

    Df_BigData_nonames = Df_Bigstuff.iloc[:,n:]

    VarSelector = VarianceThreshold(threshold=thresh)
    Sifted_Data = VarSelector.fit_transform(Df_BigData_nonames)

    Selected_Wavelengths = Df_BigData_nonames.columns[VarSelector.get_support()]
    Df_Sifted = pd.DataFrame(Sifted_Data, columns=Selected_Wavelengths)

    return Df_Sifted

'''
Works as intended, pretty dope. The function below will now work without threshold but specifically
try to find the threshold which will lead to n columns, this makes things more operable.
'''

def VarThresh_Analysis_Output(k, n, increase_rate, Df_Bigstuff):
    '''Currently does not work.'''


    thresh = increase_rate * 1000
    i=1

    Df_Sifted = VarThresh_Analysis(0, n, Df_Bigstuff)

    while len(Df_Sifted.head()) > k:

        print(i)
        i= i+ 1

        thresh = thresh - increase_rate

        try:
            Df_Sifted = VarThresh_Analysis(thresh, n, Df_BigData)

        except: 
            None

    outhandle = r"C:\University\science utat\VarThreshedEMs.csv"
    Df_Sifted.to_csv(outhandle)

    return Df_Sifted

#endregion

#region KNN Analysis

'''
In this section, we'll actually perform the KNN analysis, to find the 'most representetive' endmembers. So far, we have readied the
data by performing PCA Analysis and reducing the number of wavelengths. We do not necessarily know what specific wavelengths mean
but it does not matter as long as we still have 'relative meaning' with respect to different endmembers.
'''

def KNN_Fit(X, init_df, k, thresh):

    """This takes X is the input with PCA performed (including all EMs to be worked on), y is the initial dataset that we will 
    use to classify these entries with, k is how many neighbors we want to use. Thresh is the threshold for EMs, values >= will
    be included in training."""

    KNN_Algo = KNeighborsClassifier(n_neighbors=k, weights='distance',algorithm='auto',p=1)
    #This initializes the requirements of the algo.


    """We first have to seperate our PCA performed data into training and "to be worked"(not used). Doing this, we will be able
    #to perform KNN on the same PCA'd data, doing KNN and predicting on different PCA'd data leads to weird results.
    #Here, we will be creating these two PCA'd sets."""

    if init_df.shape[0] == X.shape[0]:
        init_df_shape=init_df.shape
    else: 
        raise Exception("PCA applied and initial dataframe are not the same")

    Thresh_Df = pd.DataFrame()
    #initialize the threshed PCA applied df

    threshed_index_list = []
    #initialize the list of rows that fulfill thresh

    for i in range(init_df_shape[0]):
        if (float(init_df.iloc[i,1]) >= thresh) or (float(init_df.iloc[i,2]) >= thresh) or (float(init_df.iloc[i,3]) >= thresh):
            threshed_index_list.append(i)
            #this collects all the row numbers that are bigger than the thresh. Using this, we will form the threshed PCA Df

    for i in threshed_index_list:
        Thresh_Df = pd.concat([Thresh_Df, X.iloc[[i]]])
        #this creates the dataframe of PCA applied EMs that also are above the thresh, 

    #Now, we will set up the class_array of labels
    Threshed_shape = Thresh_Df.shape
    class_array= np.full(shape=(Threshed_shape[0],), fill_value = -1 ,dtype=int)
    
    #create a non-shaped array of equal length to the row number of initial data

    counter = 0
    for i in threshed_index_list:
        #for all the entries in the initial dataset, we input if it is gv, npv, or soil
        
        if float(init_df.iloc[i,1]) >= thresh:
            class_array[counter] = 0
            #ie gv

        if  float(init_df.iloc[i,2]) >= thresh:
            class_array[counter] = 1
            #ie npv
        
        if float(init_df.iloc[i,3]) >= thresh:
            class_array[counter] = 2
            #ie soil
        #with this complete, we now have the initial classification array
        counter = counter+1

    KNN_Algo.fit(Thresh_Df,class_array)
    #using the Post-PCA array and the initial classification array, train a model so that we can use it to predict on new data

    return KNN_Algo #return the algorithm, use this to further predict some more data, and calculate the distances

def Collect_Class(Df_BigData, Predicts, class_number):
    """
    Returns the dataframe only belonging to a single class.

    Df_BigData is the PCA applied dataframe, predicts are the prediction vector, class_number is self explanatory.
    """
    target_rows = np.where(Predicts == class_number)[0]
    Df_class = Df_BigData.iloc[target_rows]
    return Df_class

#endregion

#region KMC Analysis

def KMC_Fit(X, n_clusters, algorithm_type):

    KMC = KMeans(n_clusters=n_clusters, algorithm=algorithm_type).fit(X)
    
    return KMC.cluster_centers_

#endregion

#region Representative Endmember

"""
Here, we will finally find the most representative endmembers. This is essentially the same as finding the data closest to
the centroid of all other data. 
"""

def Find_Centroid(Class_Dataframe):
    """
    This will take in the class and then spews out representative endmember index, using euclidian dist. Takes in class dataframe
    This function will see more changes, like other methods to find the most representative EM. Other methods to look into are:
    Medoid, Point Closest to Median, Farthest Point Sammpling, Maximum Density Point (kernel density estimation), Principal
    Component Projection (selecting the point that has most largest/smallest proj to the first pc, or collection of pcs), or
    using K Means Clustering alltogether and picking the cluster center.
    """

    centroid = np.mean(Class_Dataframe, axis=0)
    #Finds the centre point along each row (axis=0) which gives the centroid

    Distance_Collection = np.linalg.norm(Class_Dataframe - centroid, axis =1)
    #Collects the distances to the centroid

    Representative_EM_index = np.argmin(Distance_Collection)
    #finds the index

    Representative_EM = Class_Dataframe.iloc[Representative_EM_index]
    #finally chooses the EM

    return Representative_EM_index, Representative_EM, centroid, Distance_Collection
    
#endregion

#Testing grounds:

"""First, fit the KNN model to a training sample, that we know the labels of (perfect EMs).
Then, perform KNN on the wanted set of endmembers, then find the most representative by finding
the centroid of the data.
"""

Df_PCA = PCAnalysis(7,40,40,Df_BigDataImperfect)
print(KMC_Fit(Df_PCA, 3, "lloyd"))

"""
KNN_Algo = KNN_Fit(Df_PCA, Df_BigDataImperfect, 4, 1)

Predictions = KNN_Algo.predict(Df_PCA)

Df_Class = Collect_Class(Df_PCA, Predictions, 1)

Find_Centroid(Df_Class)
"""

#Df_PCA = PCAnalysis_DetailedOutput(10,20,20,Df_BigData)
#Array_PCA = Df_PCA.to_numpy()

print("Done")
