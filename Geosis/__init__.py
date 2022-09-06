## section 0: read libraries 
import pandas as pd 
import numpy as np 

## section I: read and transform the data 

def data_read(path):
    """ read the Scopus dataset and return only the required columns
    """
    columns = ["Auhtors", "Title", "Year", "Affiliation", "Abstract", "Document Type"]
    data = pd.read_csv(f'path')
    data.columns = [col.strip() for col in data.columns]
    data = data[['']]

    return data

## section II: 

def subtract_numbers(num1, num2):
    return num1 - num2 

