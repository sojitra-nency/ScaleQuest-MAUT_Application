import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def MAUT_1():
    st.title("MAUT = Multi Attribute Utility Theory")
    
    col1,col2=st.columns([2,2])
    
    with col1:
        st.image("maut_1-MAUT_1.drawio.png")
    with col2:
        st.image("maut_1-MAUT_1-WORKING.drawio.png")
        
    
    # Read the Excel file and skip the first row
    data = pd.read_excel(upload_file, skiprows=1)

    # Remove first column
    data = data.iloc[:, 1:]

    # Remove the last two rows
    data = data.iloc[:-2]

    # Rename the columns
    data.columns = ['S.no', 'Company', 'Vendor', 'IRR', 'Strategic fit', 'Technical Feasibility',
                    'Uniqueness of R&D', 'Reputational risk', 'Market and Business risk',
                    'Scalability', 'Regulatory risk', 'Market factors'] + data.columns[12:].tolist()

    # Convert numeric columns to numeric data type
    numeric_columns = ['IRR', 'Strategic fit', 'Technical Feasibility', 'Uniqueness of R&D',
                    'Reputational risk', 'Market and Business risk', 'Scalability',
                    'Regulatory risk', 'Market factors']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Apply Min-Max normalization to the numeric columns
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Assign weights to each attribute
    weights = {
        'IRR': 0.2,
        'Strategic fit': 0.1,
        'Technical Feasibility': 0.15,
        'Uniqueness of R&D': 0.1,
        'Reputational risk': 0.1,
        'Market and Business risk': 0.1,
        'Scalability': 0.1,
        'Regulatory risk': 0.1,
        'Market factors': 0.05
    }

    # Calculate weighted scores for each attribute
    for attribute in weights:
        data[attribute + '_weighted'] = data[attribute] * weights[attribute]

    # Calculate overall scores
    data['overall_score'] = data.filter(like='_weighted').sum(axis=1)

    # Rank options based on overall scores
    ranked_data = data.sort_values('overall_score', ascending=False)

    # Abbreviate vendor names
    abbreviated_names = ['Vendor {}'.format(i + 1) for i in range(len(ranked_data))]
    ranked_data['Abbreviated Vendor'] = abbreviated_names

    # Save ranked options to CSV
    ranked_data[['S.no', 'Company', 'Vendor', 'Abbreviated Vendor', 'overall_score']].to_csv('Options.csv', index=False)
    output = pd.read_csv('Options.csv')
    
    
    # Scale the overall scores between 0 and 1 using Min-Max normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    output['normalized_score'] = scaler.fit_transform(output[['overall_score']])

    # Plot utility curve
    fig, ax = plt.subplots()
    ax.plot(output['Abbreviated Vendor'], output['normalized_score'])
    ax.set_xlabel('Vendor')
    ax.set_ylabel('Utility Score')
    ax.set_title('Utility Curve')
    ax.tick_params(axis='x', rotation=90, labelsize=2)
    st.pyplot(fig)

def MAUT_AHP():
    
    st.title("AHP = Analytical Hierarchy Process")
    col1,col2=st.columns([2,2])
    
    with col1:
        st.image("maut_1-AHP.drawio.png")
    with col2:
        st.image("maut_1-AHP- WORKING.drawio.png")

    # Read the Excel file and skip the first row
    data = pd.read_excel(upload_file, skiprows=1)

    # Remove first column
    data = data.iloc[:, 1:]

    # Remove the last two rows
    data = data.iloc[:-2]

    # Rename the columns
    data.columns = ['S.no', 'Company', 'Vendor', 'IRR', 'Strategic fit', 'Technical Feasibility',
                    'Uniqueness of R&D', 'Reputational risk', 'Market and Business risk',
                    'Scalability', 'Regulatory risk', 'Market factors'] + data.columns[12:].tolist()

    # Convert numeric columns to numeric data type
    numeric_columns = ['IRR', 'Strategic fit', 'Technical Feasibility', 'Uniqueness of R&D',
                    'Reputational risk', 'Market and Business risk', 'Scalability',
                    'Regulatory risk', 'Market factors']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Apply Min-Max normalization to the numeric columns
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Assign weights to each attribute
    weights = {
        'IRR': 0.2,
        'Strategic fit': 0.1,
        'Technical Feasibility': 0.15,
        'Uniqueness of R&D': 0.1,
        'Reputational risk': 0.1,
        'Market and Business risk': 0.1,
        'Scalability': 0.1,
        'Regulatory risk': 0.1,
        'Market factors': 0.05
    }

    # Calculate weighted scores for each attribute
    for attribute in weights:
        data[attribute + '_weighted'] = data[attribute] * weights[attribute]

    # Calculate overall scores
    data['overall_score'] = data.filter(like='_weighted').sum(axis=1)

    # Rank options based on overall scores
    ranked_data = data.sort_values('overall_score', ascending=False)

    # Abbreviate vendor names
    abbreviated_names = ['Vendor {}'.format(i + 1) for i in range(len(ranked_data))]
    ranked_data['Abbreviated Vendor'] = abbreviated_names

    # Save ranked options to CSV
    ranked_data[['S.no', 'Company', 'Vendor', 'Abbreviated Vendor', 'overall_score']].to_csv('Options.csv', index=False)
    output = pd.read_csv('Options.csv')

    # Scale the overall scores between 0 and 1 using Min-Max normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    output['normalized_score'] = scaler.fit_transform(output[['overall_score']])
    
    # Plot utility curve
    fig, ax = plt.subplots()
    ax.plot(output['Abbreviated Vendor'], output['normalized_score'])
    ax.set_xlabel('Vendor')
    ax.set_ylabel('Utility Score')
    ax.set_title('Utility Curve')
    ax.tick_params(axis='x', rotation=90, labelsize=2)
    st.pyplot(fig)

    
    
    
def MAUT_LOSS():
    st.title("LOSS function = Level of Service Satisfaction")
    col1,col2=st.columns([2,2])
    
    with col1:
        st.image("maut_1-LOSS.drawio.png")
    with col2:
        st.image("maut_1-LOSS-WORKING.drawio.png")


    # Read the Excel file and skip the first row
    data = pd.read_excel('Consolidated data.xlsx', skiprows=1)

    # Remove first column
    data = data.iloc[:, 1:]

    # Remove the last two rows
    data = data.iloc[:-2]

    # Rename the columns
    data.columns = ['S.no', 'Company', 'Vendor', 'IRR', 'Strategic fit', 'Technical Feasibility',
                    'Uniqueness of R&D', 'Reputational risk', 'Market and Business risk',
                    'Scalability', 'Regulatory risk', 'Market factors'] + data.columns[12:].tolist()

    # Convert numeric columns to numeric data type
    numeric_columns = ['IRR', 'Strategic fit', 'Technical Feasibility', 'Uniqueness of R&D',
                    'Reputational risk', 'Market and Business risk', 'Scalability',
                    'Regulatory risk', 'Market factors']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Apply Min-Max normalization to the numeric columns
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Convert the data to CSV format
    data.to_csv('Consolidated data_normalized.csv', index=False)

    # Read the normalized CSV file
    data = pd.read_csv('Consolidated data_normalized.csv')

    # Define the pairwise comparison matrix for criteria
    criteria_matrix = np.array([
        [1, 3, 5, 3, 7, 9, 5, 7, 3],  # IRR compared to other criteria
        [1/3, 1, 3, 3, 7, 7, 5, 7, 3],  # Strategic fit compared to other criteria
        [1/5, 1/3, 1, 1/3, 3, 3, 3, 5, 1],  # Technical Feasibility compared to other criteria
        [1/3, 1/3, 3, 1, 5, 7, 5, 7, 3],  # Uniqueness of R&D compared to other criteria
        [1/7, 1/7, 1/3, 1/5, 1, 3, 3, 5, 1],  # Reputational risk compared to other criteria
        [1/9, 1/7, 1/3, 1/7, 1/3, 1, 1/3, 1, 1/3],  # Market and Business risk compared to other criteria
        [1/5, 1/5, 1/3, 1/5, 1/3, 3, 1, 3, 1/3],  # Scalability compared to other criteria
        [1/7, 1/7, 1/5, 1/7, 1/5, 1, 1/3, 1, 1/5],  # Regulatory risk compared to other criteria
        [1/3, 1/3, 1, 1/3, 1, 3, 3, 5, 1]  # Market factors compared to other criteria
    ])

    # Perform weight calculation
    weights = np.power(np.prod(criteria_matrix, axis=1), 1 / criteria_matrix.shape[0])
    weights = weights / np.sum(weights)

    # Assign the calculated weights to the attributes
    weights_dict = dict(zip(data.columns[3:12], weights))

    # Calculate weighted scores for each attribute using the exponential utility function
    a = 1  # Rate of increase in the utility function
    for attribute in weights_dict:
        data[attribute + '_utility'] = np.exp(a * data[attribute]) * weights_dict[attribute]

    # Calculate overall scores
    data['overall_score'] = data.filter(like='_utility').sum(axis=1)

    # Rank options based on overall scores
    ranked_data = data.sort_values('overall_score', ascending=False)

    # Abbreviate vendor names
    abbreviated_names = ['Vendor {}'.format(i + 1) for i in range(len(ranked_data))]
    ranked_data['Abbreviated Vendor'] = abbreviated_names

    # Save ranked options to CSV
    ranked_data[['S.no', 'Company', 'Vendor', 'Abbreviated Vendor', 'overall_score']].to_csv('Ranked Options.csv', index=False)
    output = pd.read_csv('Ranked Options.csv')
    
    # Scale the overall scores between 0 and 1 using Min-Max normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    output['normalized_score'] = scaler.fit_transform(output[['overall_score']])


    # Plot utility curve
    fig, ax = plt.subplots()
    ax.plot(output['Abbreviated Vendor'], output['normalized_score'])
    ax.set_xlabel('Vendor')
    ax.set_ylabel('Utility Score')
    ax.set_title('Utility Curve')
    ax.tick_params(axis='x', rotation=90, labelsize=2)
    st.pyplot(fig)


def MAUT_TOPSIS():
    st.title("TOPSIS = Technique for Order Preference by Similarity to Ideal Solution.")
    col1,col2=st.columns([2,2])
    
    with col1:
        st.image("maut_1-TOPSIS.drawio.png")
    with col2:
        st.image("maut_1-TOPSIS-WORKING.drawio.png")

    # Read the Excel file and skip the first row
    data = pd.read_excel(upload_file, skiprows=1)

    # Remove first column
    data = data.iloc[:, 1:]

    # Remove the last two rows
    data = data.iloc[:-2]

    # Rename the columns
    data.columns = ['S.no', 'Company', 'Vendor', 'IRR', 'Strategic fit', 'Technical Feasibility',
                    'Uniqueness of R&D', 'Reputational risk', 'Market and Business risk',
                    'Scalability', 'Regulatory risk', 'Market factors'] + data.columns[12:].tolist()

    # Convert numeric columns to numeric data type
    numeric_columns = ['IRR', 'Strategic fit', 'Technical Feasibility', 'Uniqueness of R&D',
                    'Reputational risk', 'Market and Business risk', 'Scalability',
                    'Regulatory risk', 'Market factors']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Apply Min-Max normalization to the numeric columns
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Define the weights for each criterion
    weights = [0.1, 0.15, 0.1, 0.15, 0.05, 0.1, 0.05, 0.1, 0.2]

    # Multiply the normalized data by the weights
    weighted_data = data[numeric_columns] * weights

    # Calculate the ideal positive and negative solutions
    ideal_positive = weighted_data.max()
    ideal_negative = weighted_data.min()

    # Calculate the Euclidean distances to ideal positive and negative solutions
    positive_distances = np.sqrt(np.sum((weighted_data - ideal_positive) ** 2, axis=1))
    negative_distances = np.sqrt(np.sum((weighted_data - ideal_negative) ** 2, axis=1))

    # Calculate the relative closeness to the ideal positive solution
    closeness = negative_distances / (positive_distances + negative_distances)

    # Add the overall scores to the data
    data['overall_score'] = closeness

    # Rank options based on overall scores
    ranked_data = data.sort_values('overall_score', ascending=False)

    # Abbreviate vendor names
    abbreviated_names = [f'Vendor {i+1}' for i in range(len(ranked_data))]

    # Save ranked options with abbreviated vendor names to CSV
    ranked_data['Abbreviated Vendor'] = abbreviated_names
    ranked_data[['S.no', 'Company', 'Abbreviated Vendor', 'overall_score']].to_csv('Ranked Options.csv', index=False)

    # Read the ranked options CSV
    output = pd.read_csv('Ranked Options.csv')
    
    # Scale the overall scores between 0 and 1 using Min-Max normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    output['normalized_score'] = scaler.fit_transform(output[['overall_score']])

    # Plot utility curve
    fig, ax = plt.subplots()
    ax.plot(output['Abbreviated Vendor'], output['normalized_score'])
    ax.set_xlabel('Vendor')
    ax.set_ylabel('Utility Score')
    ax.set_title('Utility Curve')
    ax.tick_params(axis='x', rotation=90, labelsize=2)
    st.pyplot(fig)




def SENSITIVITY_TEST():
    # Read the Excel file and skip the first row
    data = pd.read_excel(upload_file, skiprows=1)

    # Define a range of weights to test
    weight_range = [0.01, 0.05]

    # Perform sensitivity test for each weight
    for weight in weight_range:
        # Update the weights dictionary with the new weight
        weights = {
            'IRR': weight,
            'Strategic fit': 0.1,
            'Technical Feasibility': 0.15,
            'Uniqueness of R&D': 0.1,
            'Reputational risk': 0.1,
            'Market and Business risk': 0.1,
            'Scalability': 0.1,
            'Regulatory risk': 0.1,
            'Market factors': 0.05
        }

        # Calculate weighted scores for each attribute
        for attribute in weights:
            if attribute in data.columns:
                data[attribute + '_weighted'] = data[attribute] * weights[attribute]

        # Calculate overall scores
        weighted_columns = [col for col in data.columns if col.endswith('_weighted')]
        data['overall_score'] = data[weighted_columns].sum(axis=1)

        # Rank options based on overall scores
        ranked_data = data.sort_values('overall_score', ascending=False)

        # Abbreviate vendor names
        abbreviated_names = ['Vendor {}'.format(i + 1) for i in range(len(ranked_data))]
        ranked_data['Abbreviated Vendor'] = abbreviated_names

        # Save ranked options to CSV
        ranked_data[['S.no', 'Company', 'Vendor', 'Abbreviated Vendor', 'overall_score']].to_csv('Options.csv', index=False)
        output = pd.read_csv('Options.csv')

        # Scale the overall scores between 0 and 1 using Min-Max normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        output['normalized_score'] = scaler.fit_transform(output[['overall_score']])

        # Plot utility curve
        fig, ax = plt.subplots()
        ax.plot(output['Abbreviated Vendor'], output['normalized_score'])
        ax.set_xlabel('Vendor')
        ax.set_ylabel('Utility Score')
        ax.set_title('Utility Curve')
        ax.tick_params(axis='x', rotation=90, labelsize=2)
        st.pyplot(fig)








# Create a sidebar with options for each IPYNB file
option = st.sidebar.selectbox('Select an IPYNB file', ['MAUT_1', 'MAUT_AHP', 'MAUT_LOSS', 'MAUT_TOPSIS','Sensitivity Analysis'])
upload_file = st.file_uploader("Upload a CSV file", type=["xlsx"])
if upload_file is not None:
    # Run the selected IPYNB file function
    if option == 'MAUT_1':
        MAUT_1()  # Call the function corresponding to MAUT_1.IPYNB
    elif option == 'MAUT_AHP':
        MAUT_AHP()  # Call the function corresponding to MAUT_AHP.IPYNB
    elif option == 'MAUT_LOSS':
        MAUT_LOSS()  # Call the function corresponding to MAUT_LOSS.IPYNB
    elif option == 'MAUT_TOPSIS':
        MAUT_TOPSIS()  # Call the function corresponding to MAUT_TOPSIS.IPYNB
    elif option == 'Sensitivity Analysis':  
        SENSITIVITY_TEST()

      
            

            
