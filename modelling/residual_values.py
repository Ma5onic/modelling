""""
Contains functionality specific for the RV project
"""

import numpy as np
import pandas as pd
from statistics import mean
import os
import pandas as pd
import numpy as np
from itertools import product


def mape(y_true, y_pred):
    """"
    Calculates the mean absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def duplicate_testset(testset,mileage_range,duration_range,splitting_variable,splitting_values):
    
    """
    
    Multiplies the testset into duplicate values with al the mileage, duration and splitting value combinations.
    
    Parameters
    ----------
    testset: pandas DataFrame
        The set of observations to be duplicated
    mileage_range: list
        The range of mileages that should be in the matrix
    duration_range: list
        The range of durations that should be in the matrix
    splitting_variable: str
        The name of the variable on which the matrices should be split for all make_models
    splitting_values: list
        The values that occur in the splitting variable that should be used for splitting
    
    Returns
    -------
    pandas DataFrame
        The duplicated testset
    
    """
    
    # Remove the variables that need to be duplicated from the testset
    testset_clean = testset.drop(['mileage','duration'],axis=1)
    for value in splitting_values:
        splitting_varname = splitting_variable+'_'+value
        testset_clean = testset_clean.drop([splitting_varname],axis=1)
    
    # Create a list of all possible combinations of the unique observations, durations, mileages and splitting values
    duplicated_columns = pd.DataFrame(list(product(testset.index,mileage_range, duration_range,splitting_values)), columns=['index_id','mileage', 'duration','split_var'])
    for value in splitting_values:
        splitting_varname = splitting_variable+'_'+value
        duplicated_columns[splitting_varname] = np.where(duplicated_columns['split_var'] == value, 1, 0)
    
    # Merge the list of duplicated observations back to the rest of the variables that match the unique observations
    duplicated_testset=pd.merge(duplicated_columns,testset_clean,left_on='index_id',right_on=testset_clean.index, how='left')
    duplicated_testset = duplicated_testset.drop(['index_id', 'split_var'], axis=1)
    
    return duplicated_testset




def duplicated_testset_to_matrix(duplicated_testset_predicted,min_matrix_size,splitting_variable,splitting_values,save_true_false,saving_directory,filename,colorcoding=False):

    """
    
    Transforms the duplicated testset into matrices with predicted RV percentages
    
    Parameters
    ----------
    duplicated_testset_predicted: pandas DataFrame
        The set of duplicated observations with added predictions from the model
    min_matrix_size: int
        Minimum size of a make_model in the testset to be assigned a separate matrix
    splitting_variable: str
        The name of the variable on which the matrices should be split for all make_models
    splitting_values: list
        The values that occur in the splitting variable that should be used for splitting
    save_true_fale: bool
        Boolean to indicate whether matrices should be written to excel file
    saving_directory: str
        Path to the directory in which the excel file should be saved
    filename: str
        Name of the excel file to save
    colorcoding: bool
        Indicates whether colorcoding should be added to the excel matrices, default is False
    
    """
    
    # Remove unnecessary variables to increase performance
    dummies=list()
    for value in splitting_values:
        dummies.append(splitting_variable+'_'+value)
    duplicated_testset_predicted[splitting_variable] = duplicated_testset_predicted[dummies].idxmax(axis=1)
    duplicated_testset_clean = duplicated_testset_predicted[['make','model_cons','mileage','duration','pred','van','catalog_price_car_options', splitting_variable]]
    
    # Count the size of each make_model in the testset and add this as a variable to each observation
    duplicated_testset_clean['make_model']= duplicated_testset_clean['make'] + ' ' + duplicated_testset_clean['model_cons']
    make_model_size = pd.DataFrame({'size' : duplicated_testset_clean.groupby(by = 'make_model').size()})
    duplicated_testset_clean=pd.merge(duplicated_testset_clean,make_model_size,left_on='make_model',right_on=make_model_size.index)
    
    #Find the mileage range, duration range and splitting values used in duplication
    mileage=sorted(duplicated_testset_clean.mileage.unique())
    duration=sorted(duplicated_testset_clean.duration.unique())
    
    # Test correct grouping of input data
    if len(mileage)>100 or len(duration)>100:
        print('input mileage or duration not grouped correctly')
        return
    
    # Compare each make_model size to the minimal matrix size and decide in which matrix observations should go
    min_matrix_size_duplicated = min_matrix_size*len(mileage)*len(duration)*len(splitting_values)
    conditions = [
        (duplicated_testset_clean['size'] < min_matrix_size_duplicated) & (duplicated_testset_clean['van'] == 0),
        (duplicated_testset_clean['size'] < min_matrix_size_duplicated) & (duplicated_testset_clean['van'] == 1),
        (duplicated_testset_clean['size'] >= min_matrix_size_duplicated)]
    choices = ['OTHER PASSENGER', 'LCV', duplicated_testset_clean['make_model']]
    duplicated_testset_clean['matrix'] = np.select(conditions, choices, default='error')
    duplicated_testset_clean['matrix'] = duplicated_testset_clean['matrix'] + ' ' + duplicated_testset_clean[splitting_variable]
    
  
    # Prepare filling of matrices
    predicted_testset = duplicated_testset_clean[['mileage','duration','pred','matrix','catalog_price_car_options']]
    matrix_dict = {}
    listmatrices=sorted(list(predicted_testset.matrix.unique()))
    if len(listmatrices)>100:
        print('too many different matrices')
        return
    elif len(listmatrices)<5:
        print('too little different matrices')
        return
    
    # Fill in every cell in all the matrices
    for matrix in listmatrices:
        matrix_dict[matrix] = np.zeros((len(mileage)+1,len(duration)+1))
        row_index=1
        subset1 = predicted_testset[predicted_testset.matrix == matrix]
        for miles in mileage:
            column_index=1
            subset2 = subset1[subset1.mileage == miles]
            for months in duration:
                subset3 = subset2[subset2.duration == months]
                matrix_dict[matrix][row_index,column_index]=mean(subset3['pred'])
                column_index += 1
            row_index += 1
        matrix_dict[matrix]=matrix_dict[matrix]/mean(subset1['catalog_price_car_options'])
        matrix_dict[matrix][0,1:len(duration)+1]=duration
        matrix_dict[matrix][1:len(mileage)+1,0]=mileage
    
    
    # Write the matrices to a single excel file, with every sheet containing one make_model
    if save_true_false:
        os.chdir(saving_directory)
        writer=pd.ExcelWriter(filename,engine='xlsxwriter')
        workbook=writer.book
        bold_fmt = workbook.add_format({'bold': True})
        for matrix in listmatrices:
            matrix_clean = matrix.replace(splitting_variable, '')
            matrix_clean = matrix_clean.replace('_', '')
            carname = matrix_clean
            split_order = 0
            for split_value in splitting_values:
                carname = carname.replace(split_value, '')
                if split_value in matrix_clean:
                    start_row = 1 + split_order * (len(mileage) + 4)
                split_order += 1
            pd.DataFrame(matrix_dict[matrix]).to_excel(writer,
                        sheet_name = carname,
                        startrow = start_row,
                        header = False,
                        index = False)
            writer.sheets[carname].write(start_row-1,0,matrix_clean,bold_fmt)
            if colorcoding:
                writer.sheets[carname].conditional_format('B'+str(start_row+2)+':'+chr(ord('B')+len(duration)-1)+str(start_row+len(mileage)+1), {'type': '3_color_scale'})
        writer.save()
    
    return matrix_dict

