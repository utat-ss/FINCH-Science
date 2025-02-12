"""
The code below was made by Jadon Tsai, 

This function seperates the rows of the spectral library file into distinct files. This will then be fed into ENVI to get .sli
"""

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
