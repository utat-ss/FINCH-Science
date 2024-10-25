#region Imports

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math  as gigachad

#endregion

#region Initialize Data
input_bigdata = r"C:\University\science utat\better_data.csv" #the dataset with mixed spectra, does not only contain EMs
dataframe_bigdata = pd.read_csv(input_bigdata)

input_emdata = r"C:\University\science utat\better_em_data.csv" #the dataset that only contains EMs, imported for easier implementation
dataframe_emdata = pd.read_csv(input_emdata) #rather than working with different columns of the same dataframe, if we use different
#dataframe in the first place, easier to make sense of what the code is trying to do.

n= 2 #how many endmembers we have -1, have not been used yet will be implemented as per Rik's reccomendation of having this code
#work for any number of inputted n-different endmembers

#endregion

#region Calculate and Solve

"""
Below, we will write the functions to determine the area under the observed graph. 
This function is the initial step to the system of eq.
"""

def dope_area_calculatinator_3000(dataframe, j, m, k):
    #will calculate the area under the m and kth columns of the jth row in the input file.

    if j<0 or j>= dataframe.shape[0] - 1: #check if the rows are not goofy
        raise ValueError(f"Invalid row entered bozo, {j} must be between 0 and {dataframe.shape[0]}")
    if m<=3 or m>k or k > dataframe.shape[1]: #check if the columns are not goofy
        raise ValueError(f"Invalid column entered bozo, {m} must be smaller than {k} but bigger than 0 and {k} must be smaller than {dataframe.shape[1]}")
    
    sumresult = 0 #initialize the summation result

    for i in range(m, k +1): #for all columns 
        average = (float(dataframe.iloc[j,i]) + float(dataframe.iloc[j, i])) /2 #since the data is discrete and area is not defined 
        #for a singular point, we take the average among different columns and multiply by the difference of wavelengths, 10.
        #after second thought, maybe I could've just multiplied the value of the column with the wavelength. Trying that and 
        #seeing if it fixes up some problems might be a good idea.
        discretearea = average * 10 #multiply average by delta-wavelength
        sumresult = sumresult + discretearea #add it to the sum result
    
    return sumresult #give the sum as a number.

def system_solver_gigachad_sigma_alpha_male_tiktok_skibidi_toilet(gv_interval1, npv_interval1, soil_interval1, solution1, gv_interval2, npv_interval2, soil_interval2, solution2, gv_interval3, npv_interval3, soil_interval3, solution3):
    #joke name, I am not a zoomer
    
    #set up the matrix to be solved:
    Matrix = np.array([[gv_interval1, npv_interval1, soil_interval1], 
                       [gv_interval2, npv_interval2, soil_interval2],
                       [gv_interval3, npv_interval3, soil_interval3]])
    
    #set up the vector of results
    Vector = np.array([solution1, solution2, solution3])

    try:
        result_of_abundances = np.linalg.solve(Matrix, Vector) #solves the abundances, returns a vector of abundances
        return result_of_abundances
    except:
        return "There's something wrong with the system. Check if you have the matrix correctly set up, it has to be 3x3 and the vector has to be 1x3."

#endregion

#region Optimal Interval 

"""
This function finds the optimal interval to be used in the taking integral process.
Using this function, we get specific intervals where the integral results return an 
even distribution like: 100-200-300, rather than 10-20-570. This allows us to have
less possible error that would result from the round offs. Since we would have to
round off the found intervals, under a small interval we will lose much more data.
So, with these optimal intervals, we won't lose as much data when rounding off.
"""

def RVMUSI_5k(guess, dataframe, rows, a, d): 
    """
    Takes inputs as initial guess of "guess", input dataframe of "dataframe",
    rows to calculate variance of the areas of the intervals given by guess, a, d.
    Note here that the first column begins with 5th, and the last is 214.
    Input this into the code as start and end. When using a derivative of 
    better_data use these as start and end.
    """

    b, c = int(round(guess[0])), int(round(guess[1]))
    variances_for_intervals=[]
    for start, end in [(a,b), (b,c), (c,d)]:
        #picks an interval, 

        sums = [
            dope_area_calculatinator_3000(dataframe, row, start, end) #calculates the integral for that interval.
            for row in rows #under different rows
        ]
        variances_for_intervals.append(np.var(sums)) #appends the variance of those sums under specific intervals for all rows.

    overall_variance = np.var(variances_for_intervals) #returns the overall variance of the intervals 
    
    return overall_variance #returns each variance for the interval

#may try to calculate the integrals of different intervals in a row first, and then across different rows
#this may be a better way to define variance across intervals 
def different_RVMUSI_5k(guess, dataframe, rows, a, d): #this function is still WIP, nothing to see here.

    b, c = int(round(guess[0])), int(round(guess[1]))
    variances_for_intervals=[]

    return None

def find_interval(df, rows, a, d): #takes in the dataframe, the rows to find the best interval for, and the a,d boundary columns

    initial_tahmin = [round((a+d)/3), round(2*((a+d)/3))] #tahmin stands for guess in turkish, we are setting up initial guesses

    constraint = [ #the constraint function we are ought to use since the intervals are well ordered,
        #for a set of intervals: [a,b],[b,c],[c,d], we need relations:
        {'type': 'ineq', 'fun': lambda guess: guess[0] - a}, #a<=b
        {'type': 'ineq', 'fun': lambda guess: guess[1] - guess[0]}, #b<=c
        {'type': 'ineq', 'fun': lambda guess: d - guess[1]}, #c<=d
    ]

    optimize = minimize( #using the minimize function,
        RVMUSI_5k, #RVMUSI_5k is the thing we are trying to minimize, minimizing the variance
        initial_tahmin, #the stuff that are going to change
        args=(df,rows,a,d), #the arguments are given, check documentation of minimize function
        constraints=constraint, #the constraints are given
        method='SLSQP'#use method
    )

    if optimize.success: #since optimize is an object, we use .success:
        
        b, c = int(round(optimize.x[0])), int(round(optimize.x[1])) #if the thing is successful, round the column numbers, duh
        optimized_interval = (b,c) #pass this into a tuple
        return optimized_interval #return tuple of optimized b,c, since a,d is known, we do not give it
        

    else:
        raise ValueError("Something went wrong: " + optimize.message) #if something went wrong, display why.

#endregion

#region Trials

"""
We first have to determine the optimal intervals for the chosen endmembers.
We choose our endmembers to be Resop139, Res031, and NF_068. Using these
endmembers, we will first calculate the optimal intervals:
"""

#Giving the initial guess rows: 

em_rows = (1,3,4)

optimized_interval = find_interval(dataframe_emdata, em_rows, 5, 214) #find the optimized interval that has least variance throughout the EMs
intervals  = [5, optimized_interval[0], optimized_interval[1], 214] #specify the intervals we're going to use

greenveg_integration_values = [] #greenveg row is 1
nphotoveg_integration_values = [] #non-photosyn. row is 3
soil_integration_values = [] #soil row is 4
#we will have better values as we set up the knn algorithm and determine em's

for i in range(len(intervals) -1): #find the area values for each EM
    greenveg_integration_values.append(dope_area_calculatinator_3000(dataframe_emdata, 1, intervals[i], intervals[i+1])) 
    nphotoveg_integration_values.append(dope_area_calculatinator_3000(dataframe_emdata, 3, intervals[i], intervals[i+1]))
    soil_integration_values.append(dope_area_calculatinator_3000(dataframe_emdata, 4, intervals[i], intervals[i+1]))

#I am assuming here that we take the data into a csv file yada yada yada and we have a specific column to test out, we gotta
#first calculate the areas of the function. 

"""
We'll write a line here so that we'll test out the relative difference in the calculated abundances of ems and the given values.
Somehow, there's something wrong with the code, when we do the calculations on the sole endmembers, we get perfect values, but for 
mixed spectra, we get really absurd abundance combinations like -1.6 0.2 1.6. This definetly cannot be true. The hypothesis is that
for these data spewing out weird results, the actual endmember is actually different than the ones we are using. So, doing this,
we can at least see if there's any mixed spectra that are true with our calculations.
"""

difference_matrix = np.empty(shape=(1,3)) #set up an empty matrix

for j in range(200): #for the first 200 rows
    values= [] #initialize the values 
    for i in range(len(intervals)-1):
        values.append(dope_area_calculatinator_3000(dataframe_bigdata, j, intervals[i], intervals[i+1])) #find values of area for row
    
    value_vector = np.array(values) #make it a vector

    gv_col = np.array(greenveg_integration_values).reshape(-1,1) #make it into a column
    npv_col = np.array(nphotoveg_integration_values).reshape(-1,1) #make it into a column
    soil_col = np.array(soil_integration_values).reshape(-1,1) #make it into a column
    coeff_matrix = np.hstack((gv_col, npv_col, soil_col)) #combine the columns into a matrix

    solution_row = np.linalg.solve(coeff_matrix, value_vector) #finding the solution row

    actual_values = [] #initialize the actual abundance values we are intending to find

    for i in range(3):
        actual_values.append(float(dataframe_bigdata.iloc[j, i+1])) #make it into a list
        actual_values_row = np.array(actual_values) #make it into a row

    difference_row  =actual_values_row- solution_row #subtract the found abundances from expected

    difference_matrix = np.vstack((difference_matrix, difference_row)) #vertically stack the found differences

print(difference_matrix) #print the difference matrix, the first line will be 0 0 0 since we are initializing the diff_matrix as [0 0 0]


#endregion