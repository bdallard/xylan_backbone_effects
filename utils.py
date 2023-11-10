import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster



def convert_percentage_column(df, column_name):
    """
    Remove the '%' sign and convert to float
    """
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce') / 100.0
    return df


def prepare_data(df):
    """
    Prepare the data by adding a new column 'plant_ID'.
    """
    df['plant_ID'] = df['transgenic_line'] + '_' + df['plant_ID_treatment']
    return df


def filter_date_treatment(df, start, end, treat):
    """
    Filters the input dataframe based on a specified date range and treatment.

    This function performs the following steps:
    1. Filters the input dataframe to only include rows where the 'Day_after_sowing' 
       falls within the specified date range (inclusive of the end date, exclusive of the start date).
    2. Further filters the dataframe to only include rows that match the specified treatment.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to be filtered.
    
    start : int
        The starting day (exclusive) for the date range filter.
    
    end : int
        The ending day (inclusive) for the date range filter.
    
    treat : str
        The treatment to filter by.

    Returns
    -------
    pandas.DataFrame
        A filtered dataframe that only includes rows that match the specified date range and treatment.

    Examples
    --------
    >>> df = pd.read_csv('path_to_df.csv')
    >>> start_day = 27
    >>> end_day = 37
    >>> treatment = 'severe drought'
    >>> filtered_df = filter_date_treatement(df, start_day, end_day, treatment)
    """
    
    tmp = df[(df['Day_after_sowing']>start-1) & (df['Day_after_sowing']<end+1)]
    res = tmp[tmp['treatement']==treat]
    return res 


def merge_all(df, drr, wli):
    """
    Performs feature engineering, merges multiple dataframes, computes ratios of specified columns,
    and prepares a final merged dataframe for further analysis.

    The function performs the following steps:
    1. Basic feature engineering on the main dataframe.
    2. Merging the main dataframe with wli dataframe.
    3. Filtering, renaming, and datatype conversion in the second dataframe (drr data).
    4. Computing mean values of specified columns for control group in main dataframe.
    5. Merging mean values with main dataframe.
    6. Computing special WW ratios of specified columns against mean values.
    7. Creating a new plant_ID column to align with the format in second dataframe.
    8. Merging the processed main dataframe with the second dataframe on plant_ID.

    Parameters
    ----------
    df : pandas.DataFrame
        The main dataset.
    
    drr : pandas.DataFrame
        Second dataset, containing additional features to be merged with the main dataframe.
    
    wli : pandas.DataFrame
        Third dataset, to be merged with the main dataframe in the early stages of processing.

    Returns
    -------
    pandas.DataFrame
        A dataframe that results from merging and processing the input dataframes, with 
        additional computed ratio columns and merged features from all input dataframes.

    Examples
    --------
    >>> df = pd.read_csv('path_to_df.csv')
    >>> drr = pd.read_csv('path_to_drr.csv')
    >>> wli = pd.read_csv('path_to_wli.csv')
    >>> final_merged_df = merge_all(df, drr, wli)
    """

    df = feature_eng_basic(df)
    df = merge_wli(df,wli)
    
    df_other = drr.copy()
    df_main = df.copy()
    df_other = df_other[(df_other['drr1'].str.strip() != '') & (df_other['drr2'].str.strip() != '')]
    
    df_other['drr1'] = df_other['drr1'].astype(float)
    df_other['drr2'] = df_other['drr2'].astype(float)
    
    # Columns to consider for ratio computation
    columns_to_consider = ['Fo', 'Fm', 'Fv', 'QY_max', 'NDVI2', 'PRI', 'SIPI', 'NDVI', 'PSRI', 
                       'MCARI1', 'OSAVI','Slenderness','WLI/NDVI','%RGB(72,84,58)','%RGB(73,86,36)',
                      '%RGB(57,71,46)','%RGB(59,71,20)']

    # Filter dataframe for 'treatement' == 'control' for ratio computing
    control_df = df_main[df_main['treatement'] == 'control']

    mean_control_df = control_df.groupby(['transgenic_line'])[columns_to_consider].mean().reset_index()
    mean_control_df = mean_control_df.rename(columns={col: f"mean_{col}" for col in columns_to_consider})
    merged_df_main = df_main.merge(mean_control_df, on=['transgenic_line'], how='left')

    # Compute the ratio for each specified column
    for col in columns_to_consider:
        merged_df_main[f"ratio_{col}"] = merged_df_main[col] / merged_df_main[f"mean_{col}"]
    
    # Create a new column in the main dataframe to match the format of plant_ID in the second dataframe
    merged_df_main['plant_ID'] = merged_df_main['transgenic_line'] + '_' + merged_df_main['plant_ID_treatment']
    final_merged_df = pd.merge(merged_df_main, df_other, on='plant_ID', how='inner')
    
    #drop the intermediates columns
    final_merged_df = final_merged_df.drop(['Fo',
                 'Fm', 'Fv', 'QY_max', 'NDVI2', 'PRI', 'SIPI', 'NDVI', 'PSRI', 'MCARI1', 'OSAVI',
                 'Area_(mm2)', 'Slenderness', '%RGB(72,84,58)', '%RGB(73,86,36)',
                 '%RGB(57,71,46)', '%RGB(59,71,20)', 'plant_ID','WLI/NDVI','mean_Fo','n', 'mean_Fm',
                 'mean_Fv','mean_QY_max', 'mean_NDVI2','mean_PRI',
                 'mean_SIPI', 'mean_NDVI', 'mean_PSRI', 'mean_MCARI1',
                 'mean_OSAVI', 'mean_Slenderness', 'mean_WLI/NDVI', 'mean_%RGB(72,84,58)', 'mean_%RGB(73,86,36)',
                 'mean_%RGB(57,71,46)', 'mean_%RGB(59,71,20)','Tray_ID','Tray_Info','Position'], axis=1)


    #return final datasets from the three inputs
    return final_merged_df


def fit_polynomial_regression(df, plant_id,deg):
    """
    Fit a polynomial regression of degree 2 for a given plant_ID.
    Returns the coefficients (a, b, c), R-squared, and MSE.
    """
    subset = df[df['plant_ID'] == plant_id].dropna(subset=['Day_after_sowing', 'Area_(mm2)'])
    
    # If after dropping NaN values, the subset is empty, return None values
    if subset.empty:
        return None, None, None, None, None
    
    X = subset['Day_after_sowing'].values.reshape(-1, 1)
    y = subset['Area_(mm2)'].values
    
    # Transform X to polynomial features
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X)
    
    # Fit the polynomial regression
    model = LinearRegression().fit(X_poly, y)
    
    # Predict y values
    y_pred = model.predict(X_poly)
    
    # Extract coefficients and compute metrics
    coef = model.coef_
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    if deg==2:
        c,b,a = model.coef_
        return a,b,c,r2,mse
    if deg==3:
        d,c,b,a = model.coef_
        return a,b,c,d,r2,mse
    if deg==4:
        e,d,c,b,a = model.coef_
        return a,b,c,d,e,r2,mse

def compute_results(df,deg=2):
    """
    Compute polynomial regression for each unique plant_ID.
    """
    results = []
    for plant_id in df['plant_ID'].unique():
        if deg==2:
            a, b, c, r2, mse = fit_polynomial_regression(df, plant_id,deg)
            results.append({'plant_ID': plant_id, 'a': a, 'b': b, 'c': c, 'R2': r2, 'MSE': mse})
        if deg==3:
            a, b, c, d, r2, mse = fit_polynomial_regression(df, plant_id,deg)
            results.append({'plant_ID': plant_id, 'a': a, 'b': b, 'c': c, 'd':d, 'R2': r2, 'MSE': mse})
        if deg==4:
            a, b, c, d, e, r2, mse = fit_polynomial_regression(df, plant_id,deg)
            results.append({'plant_ID': plant_id, 'a': a, 'b': b, 'c': c, 'd':d, 'e':e, 'R2': r2, 'MSE': mse})
    return pd.DataFrame(results)

def feature_eng_basic(df):
    #cast all colums to float 
    columns = ['Fo', 'Fm', 'Fv', 'QY_max', 'NDVI2', 'PRI', 'SIPI', 'NDVI', 'PSRI', 'MCARI1', 'OSAVI', 'Area_(mm2)', 'Slenderness']
    for x in columns:
        df[x] = df[x].astype('float')
       
    #add plant_ID column 
    df['plant_ID'] = df['transgenic_line'] + '_' + df['plant_ID_treatment']

    #convert RGB data 
    convert_percentage_column(df, '%RGB(72,84,58)')
    convert_percentage_column(df, '%RGB(73,86,36)')
    convert_percentage_column(df, '%RGB(57,71,46)')
    convert_percentage_column(df, '%RGB(59,71,20)')
    df = df.drop(['Size2?','Size1?','RGB(72,84,58)','RGB(73,86,36)','RGB(57,71,46)','RGB(59,71,20)'],axis=1)
    return df 

def plot_polynomial_regression(df, plant_id,d):
    """
    Plot the data points and the polynomial regression line for a given plant_ID.
    
    Parameters:
    - df: DataFrame containing the data.
    - plant_id: The plant_ID for which the plot needs to be generated.
    - d: degree of the regression 
    """
    # Filter data for the given plant_id
    #subset = df[df['plant_ID'] == plant_id]
    subset = df[df['plant_ID'] == plant_id].dropna(subset=['Day_after_sowing', 'Area_(mm2)'])
    
    # Extract X and y values
    X = subset['Day_after_sowing'].values.reshape(-1, 1)
    y = subset['Area_(mm2)'].values
    
    # Polynomial transformation
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    
    # Fit the polynomial regression
    model = LinearRegression().fit(X_poly, y)
    
    # Predict y values
    y_pred = model.predict(X_poly)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, y_pred, color='red', label='Regression Line')
    plt.title(f'Polynomial Regression for plant_ID: {plant_id}')
    plt.xlabel('Day_after_sowing')
    plt.ylabel('Area_(mm2)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'\n--- Regression {d} dregree with coefs : {model.coef_} ---\n')

    
plant_ids_out = [
    "irx9_2_SD16", "irx14_SD16", "irx9_2_SD15", "irx10_SD32", "irx9_2_MD32",
    "FC4_2_SD16", "irx9_2_SD22", "FC5_2_SD16", "irx14_C16", "irx9_2_SD19",
    "FC5_2_SD37", "Col0_SD14", "FC4_1_SD1", "FC4_2_SD3", "irx14_SD18",
    "irx9_2_SD10", "FC4_2_C11", "Col0_SD32", "FC4_2_SD23", "irx10_SD3",
    "irx14_MD26", "FC4_1_SD8", "FC5_1_C11", "FC4_1_SD11", "FC4_2_C16",
    "irx9_2_SD32", "FC5_2_C11", "irx10_C11", "Col0_C11", "FC5_1_SD16",
    "irx14_C11", "irx9_2_SD30", "irx14_C17", "irx10_C17", "FC4_1_SD28",
    "irx9_2_C16", "FC4_1_SD16", "irx9_2_SD20", "irx10_C16", "FC4_2_C19",
    "Col0_SD27", "FC5_2_C32", "FC4_1_C11", "FC5_1_C6", "FC5_1_SD3",
    "FC5_2_C17", "Col0_SD33", "FC4_1_MD28", "FC5_1_C29", "irx14_C6",
    "irx9_2_C17", "Col0_SD37", "FC4_2_C6", "FC5_2_SD8", "irx9_2_SD1",
    "irx14_SD33", "irx10_SD18", "FC4_2_C17", "irx9_2_SD8", "FC5_1_C17",
    "FC5_2_SD14", "irx9_2_SD36", "FC5_1_C32", "irx10_SD6", "FC4_2_SD8",
    "irx10_MD28", "FC5_2_MD21", "Col0_C17", "irx10_SD7", "irx14_MD31",
    "FC4_2_SD7", "irx10_SD8", "irx14_C32", "FC4_2_SD40", "irx9_2_SD17",
    "FC5_1_C36", "irx10_C34", "FC5_1_C34", "FC5_2_SD32", "FC5_2_C1",
    "irx14_SD32", "FC5_2_MD28", "FC4_1_C32", "FC4_2_SD18", "irx10_C19",
    "Col0_C9", "irx9_2_MD11", "irx10_SD31", "FC5_1_SD39", "FC5_2_SD6",
    "irx14_C12", "irx14_MD5", "FC5_2_MD26", "irx10_MD26", "irx9_2_SD9",
    "FC5_2_SD17", "FC4_1_C29", "FC5_2_C27", "irx14_SD11", "FC4_1_C17",
    "irx10_C20", "irx14_SD39", "irx9_2_C18", "FC4_2_SD32", "irx9_2_SD33",
    "FC4_2_C32", "FC4_1_SD18", "irx9_2_C35", "FC4_2_MD31", "irx9_2_SD5",
    "FC5_2_SD38", "FC5_1_SD24", "FC4_1_MD30", "irx14_C9", "FC5_1_SD31",
    "irx10_SD22", "FC4_2_SD11", "irx10_C38", "irx10_MD2", "irx9_2_C36",
    "irx9_2_SD35"
]


def compute_regression(data, days, feature, treatement, group_by, regression='linear'):
    """
    Compute regression (linear or polynomial) on a given feature for specified days.
    
    Parameters:
    - data: DataFrame containing the data.
    - days: List of days for which regression is to be computed.
    - feature: The feature on which regression is to be performed.
    - group_by: Column name to group the data by.
    - regression: Type of regression ('linear' or 'polynomial'). Default is 'linear'.
    
    Returns:
    - results_df: DataFrame containing the coefficients for each group.
    
    Usage:
        results_linear = compute_regression(df_late_1, [27, 28, 29], 'Area_(mm2)', 'control','transgenic_line', regression='linear')
        results_poly = compute_regression(df_late_1, [27, 28, 29], 'Area_(mm2)', 'control','transgenic_line', regression='polynomial')

        print("Linear Regression Results by transgenic_line for control group:")
        print(results_linear)

        print("\nPolynomial Regression Results by transgenic_line for control group:")
        print(results_poly)
    
    """
    
    # Filter data
    filtered_data = data[(data['Day_after_sowing'].isin(days)) & 
                         (data['treatement'] == treatement)]
    
    results = []
    
    for item in filtered_data[group_by].unique():
        subset = filtered_data[filtered_data[group_by] == item]
        
        X = subset['Day_after_sowing'].values.reshape(-1, 1)
        Y = subset[feature].values
        
        if regression == 'linear':
            model = LinearRegression().fit(X, Y)
            coefficient = model.coef_[0]
            
        elif regression == 'polynomial':
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, Y)
            coefficient = model.coef_  # Coefficient of the quadratic term
            
        results.append({group_by: item, 'coefficient': coefficient})
    
    results_df = pd.DataFrame(results)
    return results_df



def tsne_transgenic_vs_treatement(data, day=19):
    
    #'PRI', 'SIPI', 'NDVI', 'PSRI', 'MCARI1', 'OSAVI', 'Area_(mm2)', 'Slenderness', '%RGB(72,84,58)', '%RGB(73,86,36)', '%RGB(57,71,46)', '%RGB(59,71,20)'
    # List of parameters we are interested in
    parameters = ['PRI', 'SIPI', 'NDVI', 'PSRI', 'MCARI1', 'OSAVI', 'area_ratio', 'Slenderness', '%RGB(72,84,58)', '%RGB(73,86,36)', '%RGB(57,71,46)', '%RGB(59,71,20)']
    
    # Filter the data to get rows which have valid 'transgenic line' and 'treatement' values
    df_filtered = data.dropna(subset=['transgenic_line', 'treatement'])
    
    # Further filter the dataframe for the given day
    df_filtered = df_filtered[df_filtered['Day_after_sowing'] == day]
    
    # Encode the 'transgenic line' and 'treatement' columns for use in t-SNE
    le_transgenic = LabelEncoder()
    df_filtered['transgenic_encoded'] = le_transgenic.fit_transform(df_filtered['transgenic_line'])

    le_treatement = LabelEncoder()
    df_filtered['treatement_encoded'] = le_treatement.fit_transform(df_filtered['treatement'])

    # Prepare data for t-SNE
    tsne_data = df_filtered[['transgenic_encoded', 'treatement_encoded']]
    
    # Handling NaN values by replacing them with the median of their respective columns
    for param in parameters:
        df_filtered[param].fillna(df_filtered[param].median(), inplace=True)

    # Check for any infinite values and replace them
    df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[parameters])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(scaled_data)
    df_filtered['tsne_1'] = tsne_results[:, 0]
    df_filtered['tsne_2'] = tsne_results[:, 1]
    
    # Plot the results with a facet grid to see how 'treatement' clusters for each 'transgenic line'
    g = sns.FacetGrid(df_filtered, col="transgenic_line", col_wrap=3, height=5, hue='treatement')
    g = (g.map(plt.scatter, "tsne_1", "tsne_2", edgecolor="w", s=100).add_legend())

    plt.show()

    
    
def tsne_transgenic_vs_treatement_for_days(data, treat='control', days=[19,21,24,28,33,36]):
    """
    Applies t-SNE (t-distributed Stochastic Neighbor Embedding) on the given dataframe 
    for specified days and plots the results to visualize clustering based on 'transgenic_line'.
    
    This function filters the data for a specified treatment, applies t-SNE on selected parameters, 
    and then plots the results to visualize how different 'transgenic_line' values cluster 
    for each specified day.
    
    Parameters:
    - data (pd.DataFrame): The input dataframe containing the data to be processed and visualized.
    - treat (str, optional): The treatment type to filter the data on. Defaults to 'control'.
    - days (list of int, optional): List of days after sowing to consider for the visualization. 
      Defaults to [19,21,24,28,33,36].
    
    Returns:
    - None: The function displays the t-SNE plots but does not return any value.
    
    Dependencies:
    - Requires seaborn (as sns), matplotlib.pyplot (as plt), numpy (as np), sklearn.preprocessing (StandardScaler and LabelEncoder), 
      and sklearn.manifold.TSNE for execution.
    
    Notes:
    - The function considers a predefined list of parameters for t-SNE.
    - NaN values in the parameters are replaced with the median of their respective columns.
    - Infinite values are replaced with NaN.
    - The data is scaled using StandardScaler before applying t-SNE.
    - The results are plotted with 'transgenic_line' as the hue to visualize clustering.
    
    Example:
    >>> df = YOUR DATAFRAME HERE (after preprocessing)
    >>> tsne_transgenic_vs_treatement_for_days(df)
    """
    # List of parameters we are interested in
    parameters = ['PRI', 'SIPI', 'NDVI', 'PSRI', 'MCARI1', 'OSAVI', 'area_ratio', 'Slenderness', 
                  '%RGB(72,84,58)', '%RGB(73,86,36)', '%RGB(57,71,46)', '%RGB(59,71,20)']
    
    for day in days:
        print(f"Day: {day}")
        
        # Filter the data to get rows which have valid 'transgenic_line' and 'treatement' values
        df_filtered = data.dropna(subset=['transgenic_line', 'treatement'])
        
        # Filter for 'control' treatment
        df_filtered = df_filtered[df_filtered['treatement'] == treat]
        
        # Further filter the dataframe for the given days
        df_filtered = df_filtered[df_filtered['Day_after_sowing'].isin(days)]
        
        # Encode the 'transgenic line' for use in t-SNE
        le_transgenic = LabelEncoder()
        df_filtered['transgenic_encoded'] = le_transgenic.fit_transform(df_filtered['transgenic_line'])
       
        # Handling NaN values by replacing them with the median of their respective columns
        for param in parameters:
            df_filtered[param].fillna(df_filtered[param].median(), inplace=True)

        # Check for any infinite values and replace them
        df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_filtered[parameters])

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(scaled_data)
        df_filtered['tsne_1'] = tsne_results[:, 0]
        df_filtered['tsne_2'] = tsne_results[:, 1]
        
        # Plot the results with a facet grid to see how 'treatement' clusters for each 'transgenic line'
       # Plot the results to see how 'transgenic_line' clusters for each 'Day_after_sowing'
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x="tsne_1", y="tsne_2", hue='transgenic_line', data=df_filtered[df_filtered['Day_after_sowing'] == day], edgecolor="w", s=100)
        plt.title(f"t-SNE for Day {day}")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.show()

# Sample usage:
# df = YOUR DATAFRAME HERE (after preprocessing)
# tsne_transgenic_vs_treatement_for_days(df)

def merge_wli(df1, df2):
    """
    Renames columns in the second dataframe, merges it with the first dataframe on 'plant ID 2' key,
    and computes a new column 'WLI/NDVI' by dividing 'WLI' by 'NDVI'.
    
    Parameters:
    - df1 (pd.DataFrame): The main dataframe.
    - df2 (pd.DataFrame): The dataframe containing 'plant id 2' and '1440/960' columns to be renamed and merged.
    
    Returns:
    - pd.DataFrame: The merged dataframe with the new 'WLI/NDVI' column.
    """
    
    if df2['plant id 2'] is not None:

        # Rename columns in df2
        df2 = df2.rename(columns={'plant id 2': 'plant ID 2', '1440/960': 'WLI'})

        # Merge dataframes on 'plant ID 2'
        merged_df = pd.merge(df1, df2, on='plant ID 2', how='left')

        # Compute the new 'WLI/NDVI' column
        merged_df['NDVI'] = merged_df['NDVI'].astype(float)
        merged_df['WLI'] = merged_df['WLI'].astype(float)
        merged_df['WLI/NDVI'] = merged_df['WLI'] / merged_df['NDVI']
        merged_df = merged_df.drop(['WLI'], axis=1)

        return merged_df
    return "No common ID to use for merging"



def compute_ratios_and_merge(df_main, df_other_path, time_window, day, stat="mean"):
    """
    Compute ratios for specified columns based on a given time window, statistical measure, and merge with another dataframe.
    
    Parameters:
    - df_main (pd.DataFrame): The main input dataframe.
    - df_other_path (str): Path to the other dataframe to be merged with the main dataframe.
    - time_window (dict): A dictionary where keys are labels for the time window and values are lists 
                          indicating the start and end of the time window. 
                          e.g., {"23-31": [23, 31], "25-29": [25, 29]}
    - stat (str): The statistical measure to compute for the time window. Default is "mean".
                  Currently, only "mean" is supported.
    
    Returns:
    - pd.DataFrame: A merged dataframe with computed ratios.
    """
    
    # Load the other dataframe
    df_other = pd.read_csv(df_other_path, sep=';', decimal=',', na_values='#N/A')
    #print('Shape drr data before processing : ', df_other.shape)
    #df_other = df_other[df_other['outlier'] == 'no']
    df_other = df_other[(df_other['drr1'].str.strip() != '') & (df_other['drr2'].str.strip() != '')]
    #print('Shape drr data after processing : ', df_other.shape)
    df_other['drr1'] = df_other['drr1'].astype(float)
    df_other['drr2'] = df_other['drr2'].astype(float)
    
    # Define a function to assign a time window label
    
    def assign_time_window(day):
        if type(day)==int:
            return day
        for label, (start, end) in time_window.items():
            if start <= day <= end:
                return label
        return "other"

    # Apply the function to create a new column 'time_window'
    df_main['time_window'] = df_main['Day_after_sowing'].apply(assign_time_window)
    
    # Columns to consider for ratio computation
    columns_to_consider = ['Fo', 'Fm', 'Fv', 'QY_max', 'NDVI2', 'PRI', 'SIPI', 'NDVI', 'PSRI', 
                           'MCARI1', 'OSAVI','Slenderness','WLI/NDVI','%RGB(72,84,58)','%RGB(73,86,36)',
                          '%RGB(57,71,46)','%RGB(59,71,20)']

    # Filter dataframe for 'treatement' == 'control'
    control_df = df_main[df_main['treatement'] == 'control']

    # Check the statistical measure and compute accordingly
    if stat == "mean":
        # Compute mean for the specified columns grouped by 'transgenic_line' and 'time_window'
        mean_df = control_df.groupby(['transgenic_line', 'time_window'])[columns_to_consider].mean().reset_index()
    else:
        raise ValueError(f"Unsupported statistical measure: {stat}")

    # Rename columns in mean_df for merging
    mean_df = mean_df.rename(columns={col: f"mean_{col}" for col in columns_to_consider})

    # Merge the original dataframe with mean_df
    merged_df_main = df_main.merge(mean_df, on=['transgenic_line', 'time_window'], how='left')
    #print(merged_df_main.shape)
    # Compute the ratio for each specified column
    for col in columns_to_consider:
        merged_df_main[f"ratio_{col}"] = merged_df_main[col] / merged_df_main[f"mean_{col}"]
    
    # Create a new column in the main dataframe to match the format of plant_ID in the second dataframe
    merged_df_main['plant_ID'] = merged_df_main['transgenic_line'] + '_' + merged_df_main['plant_ID_treatment']
    #print('Shape drr data: ', df_other.shape)
    #print('head drr data before merge : ', df_other.head())
    # Merge the two dataframes on the plant_ID column
    final_merged_df = pd.merge(merged_df_main, df_other, on='plant_ID', how='inner')
    
    # Filter the merged dataframe based on the time window
    #final_merged_df = final_merged_df[final_merged_df['time_window'].isin(time_window.keys())]
    
    # Drop unnecessary columns
    #final_merged_df = pd.merge(df, df_other, on='plant_ID', how='inner')
    #final_merged_df = final_merged_df.drop(['Day_after_sowing', 'outlier'], axis=1)
    #print('Shape final merge data: ', final_merged_df.shape)
    #print('final merge data dataframe columns : ', final_merged_df.head())


    #for i,e in time_window:
    filtre = final_merged_df[final_merged_df['Day_after_sowing']==day]
    
    #filtre = final_merged_df.groupby(['plant_ID']).mean().reset_index()
    md = filtre[filtre['plant ID 2'].str.contains('MD')]
    sd = filtre[filtre['plant ID 2'].str.contains('SD')]
    #print('SD data for day:{day} :', sd.shape)
    #print('MD data for day:{day} :', md.shape)
    return sd, md

# Example usage:
# sd, md = compute_ratios_and_merge(df, '/Users/mac/Downloads/drr_22_to_37.csv', time_window={"27": [27]}, day=27)


def compute_ratios_and_merge_2days(df_main, df_other_path, time_window, day1, day2, stat="mean"):
    """
    Compute ratios for specified columns based on a given time window, statistical measure, and merge with another dataframe.
    
    Parameters:
    - df_main (pd.DataFrame): The main input dataframe.
    - df_other_path (str): Path to the other dataframe to be merged with the main dataframe.
    - time_window (dict): A dictionary where keys are labels for the time window and values are lists 
                          indicating the start and end of the time window. 
                          e.g., {"23-31": [23, 31], "25-29": [25, 29]}
    - stat (str): The statistical measure to compute for the time window. Default is "mean".
                  Currently, only "mean" is supported.
    
    Returns:
    - pd.DataFrame: A merged dataframe with computed ratios.
    """
    
    # Load the other dataframe
    df_other = pd.read_csv(df_other_path, sep=';', decimal=',', na_values='#N/A')
    #print('Shape drr data before processing : ', df_other.shape)
    
    #Filter outliers 
    #df_other = df_other[df_other['outlier'] == 'no']
    df_other = df_other[(df_other['drr1'].str.strip() != '') & (df_other['drr2'].str.strip() != '')]
    #print('Shape drr data after processing : ', df_other.shape)
    
    #define a function to assign a time window label
    def assign_time_window(day):
        if type(day)==int:
            return day
        for label, (start, end) in time_window.items():
            if start <= day <= end:
                return label
        return "other"

    # Apply the function to create a new column 'time_window'
    df_main['time_window'] = df_main['Day_after_sowing'].apply(assign_time_window)
    
    # Columns to consider for ratio computation
    columns_to_consider = ['Fo', 'Fm', 'Fv', 'QY_max', 'NDVI2', 'PRI', 'SIPI', 'NDVI', 'PSRI', 'MCARI1', 'OSAVI','Slenderness','WLI/NDVI']

    # Filter dataframe for 'treatement' == 'control'
    control_df = df_main[df_main['treatement'] == 'control']

    # Check the statistical measure and compute accordingly
    if stat == "mean":
        # Compute mean for the specified columns grouped by 'transgenic_line' and 'time_window'
        mean_df = control_df.groupby(['transgenic_line', 'time_window'])[columns_to_consider].mean().reset_index()
    else:
        raise ValueError(f"Unsupported statistical measure: {stat}")

    # Rename columns in mean_df for merging
    mean_df = mean_df.rename(columns={col: f"mean_{col}" for col in columns_to_consider})

    # Merge the original dataframe with mean_df
    merged_df_main = df_main.merge(mean_df, on=['transgenic_line', 'time_window'], how='left')
    
    # Compute the ratio for each specified column
    for col in columns_to_consider:
        merged_df_main[f"ratio_{col}"] = merged_df_main[col] / merged_df_main[f"mean_{col}"]
    
    # Create a new column in the main dataframe to match the format of plant_ID in the second dataframe
    merged_df_main['plant_ID'] = merged_df_main['transgenic_line'] + '_' + merged_df_main['plant_ID_treatment']
    #print('Shape drr data: ', df_other.shape)
    #print('head drr data before merge : ', df_other.head())
    
    # Merge the two dataframes on the plant_ID column
    final_merged_df = pd.merge(merged_df_main, df_other, on='plant_ID', how='inner')
    
    # Filter the merged dataframe based on the time window
    #final_merged_df = final_merged_df[final_merged_df['time_window'].isin(time_window.keys())]
    
    # Drop unnecessary columns
    #final_merged_df = pd.merge(df, df_other, on='plant_ID', how='inner')
    #final_merged_df = final_merged_df.drop(['Day_after_sowing', 'outlier'], axis=1)
    #print('Shape final merge data: ', final_merged_df.shape)
    #print('final merge data dataframe columns : ', final_merged_df.head())


    #for i,e in time_window:
    filtre1 = final_merged_df[final_merged_df['Day_after_sowing']==day1]
    filtre2 = final_merged_df[final_merged_df['Day_after_sowing']==day2]
    
    #filtre = final_merged_df.groupby(['plant_ID']).mean().reset_index()
    md1 = filtre1[filtre1['plant ID 2'].str.contains('MD')]
    sd1 = filtre1[filtre1['plant ID 2'].str.contains('SD')]
    print('SD1 shape :', sd1.shape)
    print('MD1 shape :', md1.shape)
    md2 = filtre2[filtre2['plant ID 2'].str.contains('MD')]
    sd2 = filtre2[filtre2['plant ID 2'].str.contains('SD')]
    print('SD2 shape :', sd2.shape)
    print('MD2 shape :', md2.shape)
    return sd1, md1, sd2, md2

# Example usage:
# sd1, md1, sd2, md2 = compute_ratios_and_merge_2days(df, '/Users/mac/Downloads/drr1_drr2_3rd.csv', time_window={"22": [22]}, day1=27,day2=37)

def merge_data_period(sd1, sd2): 
    # List of columns to keep from df2
    cols_to_keep = ['plant_ID', 'ratio_Slenderness', 'ratio_Fo', 'ratio_Fm', 'ratio_QY_max', 'ratio_Fv',
                    'ratio_NDVI2', 'ratio_PRI', 'ratio_NDVI', 'ratio_PSRI', 'ratio_OSAVI', 'ratio_WLI/NDVI','ratio_MCARI1','ratio_SIPI',
                    'ratio_%RGB(72,84,58)', 'ratio_%RGB(73,86,36)', 'ratio_%RGB(57,71,46)', 'ratio_%RGB(59,71,20)', 
                    'drr1', 'drr2']

    # Subset df2 to keep only the desired columns
    sd2_subset = sd2[cols_to_keep]

    # Rename the columns of df2_subset to add the '_2' suffix
    sd2_subset = sd2_subset.rename(columns={col: col + '_2' if col != 'plant_ID' else col for col in sd2_subset.columns})

    # Merge df1 with the subset of df2 on 'plant_ID'
    merged_df = pd.merge(sd1, sd2_subset, on='plant_ID', how='left')

    merged_df = merged_df.drop(['drr1_2','drr2_2','Tray_ID','plant ID 2','Position','Measuring_date', 'Round_Order','Fo', 'Fm', 'Fv', 'QY_max', 'NDVI2',
       'PRI', 'SIPI', 'NDVI', 'PSRI', 'MCARI1', 'OSAVI', 'Area_(mm2)',
       'Slenderness', 'n'], axis=1)

    return merged_df


#Example Usage : 
#merge_df_md = merge_data_period(md1,md2)


def preprocess_dataset_dendrogram(df):
    """
    Preprocesses the dataset for heatmap dendrogram plotting.
    
    This function performs several modifications on the dataset:
    1. Drops specified columns.
    2. Renames `%RGB(.,.,.)` columns.
    3. Removes the term `ratio` from specified columns.
    4. Modifies columns ending with `_2` to end with `_L` and others with `_E`.
    5. Changes column names `drr1` and `drr2` to `err_E` and `err_L` respectively.
    6. Modifies values in the `transgenic_line` column.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe to be preprocessed.
    
    Returns:
    - pd.DataFrame: The preprocessed dataframe.
    """
    
    # Drop specified columns
    columns_to_drop = ['Measuring_date', 'Round_Order', 'Tray_ID', 'Tray_Info', 'Position', 'plant_ID_treatment',
                       'Day_after_sowing', 'treatement', 'Fo', 'Fm', 'Fv', 'QY_max', 'NDVI2', 'PRI', 'SIPI', 'NDVI',
                       'PSRI', 'MCARI1', 'OSAVI', 'Area_(mm2)', 'Slenderness', 'mean_Fo', 'mean_Fm', 'mean_Fv',
                       'mean_QY_max', 'mean_NDVI2', 'mean_PRI', 'mean_SIPI', 'mean_NDVI', 'mean_PSRI', 'mean_MCARI1',
                       'mean_OSAVI', 'mean_Slenderness','n','outlier','WLI/NDVI']
    
    df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
    
    # Rename `%RGB(.,.,.)` columns
    df = df.rename(columns={
        'ratio_%RGB(59,71,20)': 'Hue 6_E',
        'ratio_%RGB(57,71,46)': 'Hue 5_E',
        'ratio_%RGB(73,86,36)': 'Hue 4_E',
        'ratio_%RGB(72,84,58)': 'Hue 3_E',
        'ratio_%RGB(59,71,20)_2': 'Hue 6_L',
        'ratio_%RGB(57,71,46)_2': 'Hue 5_L',
        'ratio_%RGB(73,86,36)_2': 'Hue 4_L',
        'ratio_%RGB(72,84,58)_2': 'Hue 3_L',
        
    })
    
    # Columns to modify
    columns_to_modify = ['ratio_Slenderness', 'ratio_Fo', 'ratio_Fm', 'ratio_QY_max', 'ratio_NDVI2', 'ratio_PRI',
                         'ratio_NDVI', 'ratio_PSRI', 'ratio_OSAVI','drr1', 'drr2','ratio_MCARI1','ratio_SIPI','ratio_Fv','ratio_Fv_2',
                         'ratio_Slenderness_2', 'ratio_Fo_2', 'ratio_Fm_2', 'ratio_QY_max_2', 'ratio_NDVI2_2', 'ratio_PRI_2','ratio_MCARI1_2','ratio_SIPI_2',
                         'ratio_NDVI_2', 'ratio_PSRI_2', 'ratio_OSAVI_2','ratio_WLI/NDVI_2','ratio_WLI/NDVI']
    
    # Remove the term `ratio` and modify column names
    rename_dict = {}
    for col in columns_to_modify:
        if col in ['drr1', 'drr2']:
            continue
        new_name = col.replace('ratio_', '')
        if new_name.endswith('_2'):
            new_name = new_name.replace('_2', '_L')
        else:
            new_name += '_E'
        rename_dict[col] = new_name
    
    # Rename `drr1` and `drr2`
    rename_dict['drr1'] = 'drr_E'
    rename_dict['drr2'] = 'drr_L'
    
    df = df.rename(columns=rename_dict)
    df['drr_E'] = df['drr_E'].astype(float)
    df['drr_L'] = df['drr_L'].astype(float)
    
    # Modify `transgenic_line` values
    df['transgenic_line'] = df['transgenic_line'].replace({
        'FC4_1': 'GH11.1',
        'FC4_2': 'GH11.2',
        'FC5_1': 'GH10.1',
        'FC5_2': 'GH10.2',
        'Irx9_2': 'Irx9'
    })
    
    return df

#example usage 
#merge_df_md = preprocess_dataset_dendrogram(merge_df_md)
#merge_df_sd = preprocess_dataset_dendrogram(merge_df_sd)
#


def plot_heatmap_with_dendrogram_2d_ratio(df,typo,day, features=['ratio_Slenderness','ratio_Fo', 'ratio_Fm', 'ratio_QY_max', 'ratio_NDVI2', 'ratio_PRI',
       'ratio_NDVI', 'ratio_PSRI', 'ratio_OSAVI','%RGB(72,84,58)', '%RGB(73,86,36)', '%RGB(57,71,46)', '%RGB(59,71,20)', 'drr1', 'drr2'], save=False):
    """
    Plots a heatmap with dendrogram for the given dataframe based on specified features.
    
    This function takes a dataframe, processes it by grouping on the 'transgenic_line' column, 
    and computes the mean for each feature. It then scales the features and plots a heatmap 
    with dendrograms for both rows and columns. The heatmap is saved as a PNG file.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be plotted.
    - typo (str): The type or condition of the data, used for the title and filename of the plot.
    - day (int): The day for which the data is considered, used for the title and filename of the plot.
    - features (list of str, optional): List of features/columns to consider from the dataframe. 
      Defaults to a predefined list of features.
    
    Returns:
    - None: The function saves the heatmap as a PNG file and displays it, but does not return any value.
    
    Warnings:
    - If non-finite values are detected in the normalized data, a warning message is printed.
    
    Dependencies:
    - Requires seaborn (as sns), matplotlib.pyplot (as plt), numpy (as np), and 
      sklearn.preprocessing.StandardScaler for execution.
    
    Example:
    >>> day = 22
    >>> sd, md = compute_ratios_and_merge(df, '/path/to/file.csv', time_window={"22": [day]}, day=day)
    >>> plot_heatmap_with_dendrogram_2d_ratio(md, 'medium drought', day, ['ratio_Slenderness', ...])
    """
    
    # Features to consider
    if not features:
        features = ['ratio_Slenderness','ratio_Fo', 'ratio_Fm', 'ratio_QY_max', 'ratio_NDVI2', 'ratio_PRI',
       'ratio_NDVI', 'ratio_PSRI', 'ratio_OSAVI','%RGB(72,84,58)', '%RGB(73,86,36)', '%RGB(57,71,46)', '%RGB(59,71,20)', 'drr1', 'drr2']
        to_scale = ['ratio_Slenderness','ratio_Fo', 'ratio_Fm', 'ratio_QY_max', 'ratio_NDVI2', 'ratio_PRI',  
                'ratio_NDVI', 'ratio_PSRI', 'ratio_OSAVI']
     
    
    # Convert drr1 and drr2 columns to float
    #df['drr1'] = df['drr1'].astype(float)
    #df['drr2'] = df['drr2'].astype(float)
    
    # Group by 'transgenic_line' and compute the mean for each feature
    grouped_data = df.groupby('transgenic_line')[features].mean()
    
    # Features to scale
    to_scale = features
    
    # Scale the specified features
    scaler = StandardScaler()
    grouped_data[to_scale] = scaler.fit_transform(grouped_data[to_scale])
    
    # Check for non-finite values
    if not np.all(np.isfinite(grouped_data)):
        print("Warning: Non-finite values detected in the normalized data.")
        return
    
    # Compute the linkage matrix for hierarchical clustering
    row_linkage = linkage(grouped_data, method='average')
    col_linkage = linkage(grouped_data.T, method='average')  # Transpose for column clustering
    
    # Plot the heatmap with dendrogram
    g = sns.clustermap(grouped_data, row_linkage=row_linkage, col_linkage=col_linkage, figsize=(18, 12), cmap='viridis',annot=True, fmt=".1f")
    plt.title(f"Heatmap Dendrogram for {typo} | day={day}")
    plt.show()
    if save:
        g.savefig(f"Heatmap Dendrogram for {typo} day={day}.png", format='png', dpi=720)
    
# Example usage :
#day=22
#sd, md = compute_ratios_and_merge(df, '/Users/mac/Downloads/drr_22_to_37.csv', time_window={"32": [day]}, day=day)
#plot_heatmap_with_dendrogram_2d_ratio(md,'medium drought',day,['ratio_Slenderness','ratio_Fo', 'ratio_Fm', 'ratio_QY_max', 'ratio_NDVI2', 'ratio_PRI','ratio_NDVI', 'ratio_PSRI', 'ratio_OSAVI','%RGB(72,84,58)', '%RGB(73,86,36)', '%RGB(57,71,46)', '%RGB(59,71,20)', 'drr1'])



def plot_pca(df, features=None):
    """
    Plots PCA (Principal Component Analysis) results for the given dataframe with density overlays.
    
    This function takes a dataframe, extracts specific features, and performs PCA to reduce 
    the dimensionality to 2 components. It then plots the PCA results with density overlays 
    for each unique 'transgenic_line' value. The PCA loadings (feature vectors) are also 
    plotted on the same graph.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be plotted.
    
    Returns:
    - None: The function displays the PCA plots with density overlays but does not return any value.
    
    Dependencies:
    - Requires seaborn (as sns), matplotlib.pyplot (as plt), pandas (as pd), and 
      sklearn.decomposition.PCA for execution.
    
    Notes:
    - The function considers a predefined list of features for PCA.
    - The density plots are created for each unique 'transgenic_line' value.
    - PCA loadings (feature vectors) are plotted as arrows on the PCA scatter plot.
    """
    
    if not features:
        #features to consider for PCA
        features = ['ratio_Slenderness', 'ratio_Fo', 'ratio_Fm', 'ratio_QY_max', 
                    'ratio_NDVI2', 'ratio_PRI', 'ratio_NDVI', 'ratio_PSRI', 'ratio_OSAVI',
                    '%RGB(72,84,58)', '%RGB(73,86,36)', '%RGB(57,71,46)', '%RGB(59,71,20)', 
                    'drr1', 'drr2']
    
    # Extract the features from the dataframe & drop nan
    df = df.dropna(subset=features)
    X = df[features]
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    # Create a dataframe for the PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['transgenic_line'] = df['transgenic_line'].values
    
    # Define the color palette
    palette = sns.color_palette("deep", len(pca_df['transgenic_line'].unique()))
    
    # Plot the PCA results
    plt.figure(figsize=(12, 10))
    
    # Plot density
    for idx, line in enumerate(pca_df['transgenic_line'].unique()):
        subset = pca_df[pca_df['transgenic_line'] == line]
        sns.kdeplot(subset['PC1'], subset['PC2'], cmap=sns.light_palette(palette[idx], as_cmap=True), 
                    shade=True, shade_lowest=False, alpha=0.5)
    
    # Scatter plot
    sns.scatterplot(x='PC1', y='PC2', hue='transgenic_line', data=pca_df, palette=palette)
    
    # Plot the feature vectors (loadings)
    for i, feature in enumerate(features):
        plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.5)
        plt.text(pca.components_[0, i]*1.2, pca.components_[1, i]*1.2, feature, color='g', ha='center', va='center')
    
    plt.title('PCA plot with density and feature vectors')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} explained variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} explained variance)")
    plt.grid(True)
    plt.show()


def plot_pca_pairs_with_density(df):
    """
    Plots PCA (Principal Component Analysis) results for the given dataframe with density overlays.
    
    This function takes a dataframe, extracts specific features, and performs PCA to reduce 
    the dimensionality to 2 components. It then plots the PCA results with density overlays 
    for each unique 'transgenic_line' value against the 'Col0' category. The PCA loadings 
    (feature vectors) are also plotted on the same graph.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be plotted.
    
    Returns:
    - None: The function displays the PCA plots with density overlays but does not return any value.
    
    Dependencies:
    - Requires seaborn (as sns), matplotlib.pyplot (as plt), pandas (as pd), and 
      sklearn.decomposition.PCA for execution.
    
    Notes:
    - The function considers a predefined list of features for PCA.
    - The density plots are created for each unique 'transgenic_line' value against the 'Col0' category.
    - PCA loadings (feature vectors) are plotted as arrows on the PCA scatter plot.
    """
    
    # Features to consider for PCA
    features = ['ratio_Slenderness', 'ratio_Fo', 'ratio_Fm', 'ratio_QY_max', 
                'ratio_NDVI2', 'ratio_PRI', 'ratio_NDVI', 'ratio_PSRI', 'ratio_OSAVI',
                '%RGB(72,84,58)', '%RGB(73,86,36)', '%RGB(57,71,46)', '%RGB(59,71,20)', 
                'drr1', 'drr2']
    
    # Extract the features from the dataframe
    df = df.dropna()
    X = df[features]
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    # Create a dataframe for the PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['transgenic_line'] = df['transgenic_line'].values
    
    # Unique transgenic_line values excluding 'Col0'
    unique_values = [val for val in pca_df['transgenic_line'].unique() if val != 'Col0']
    
    # Define the color palette
    palette = sns.color_palette("deep", 2)  # Two colors: one for 'Col0' and one for the other category
    
    # Create subplots
    fig, axes = plt.subplots(nrows=len(unique_values), figsize=(10, 5*len(unique_values)))
    
    for idx, (line, ax) in enumerate(zip(unique_values, axes)):
        subset = pca_df[pca_df['transgenic_line'].isin(['Col0', line])]
        
        # Density plot for 'Col0'
        sns.kdeplot(subset[subset['transgenic_line'] == 'Col0']['PC1'], 
                    subset[subset['transgenic_line'] == 'Col0']['PC2'], 
                    cmap=sns.light_palette(palette[0], as_cmap=True), 
                    shade=True, shade_lowest=False, alpha=0.5, ax=ax)
        
        # Density plot for the other category
        sns.kdeplot(subset[subset['transgenic_line'] == line]['PC1'], 
                    subset[subset['transgenic_line'] == line]['PC2'], 
                    cmap=sns.light_palette(palette[1], as_cmap=True), 
                    shade=True, shade_lowest=False, alpha=0.5, ax=ax)
        
        # Scatter plot
        sns.scatterplot(x='PC1', y='PC2', hue='transgenic_line', data=subset, palette=palette, ax=ax)
        
        # Plot the feature vectors (loadings)
        for i, feature in enumerate(features):
            ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.5)
            ax.text(pca.components_[0, i]*1.2, pca.components_[1, i]*1.2, feature, color='g', ha='center', va='center')
        
        ax.set_title(f"PCA plot for Col0 and {line}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} explained variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} explained variance)")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
#Example Usage :     
#plot_pca_pairs_with_density(sd1)    



def PLS_DA_pair(data, pair_line:list):
    """
    Performs Partial Least Squares Discriminant Analysis (PLS-DA) for a pair of transgenic lines.
    
    This function processes the data by selecting features corresponding to the specified pair of transgenic lines,
    encodes the transgenic line labels, scales the features, and then splits the dataset into training and testing sets.
    It initializes and trains a PLSRegression model, predicts on both training and testing sets, computes accuracy, 
    and visualizes the results in a 2D score plot.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing features and labels.
    
    pair_line : list
        A list containing the two transgenic lines to be compared using PLS-DA.

    Returns
    -------
    None
        This function does not return any value. It prints the training and testing accuracy and
        displays a 2D score plot for the training data.

    Notes
    -----
    - The function assumes that 'transgenic_line' and 'treatement' are columns within the dataframe.
    - The label encoding, train-test split, and PLSRegression are hardcoded within the function.
    - Visualization is produced using matplotlib and seaborn, with the assumption that they are imported 
      as `plt` and `sns` respectively.

    Examples
    --------
    >>> data = pd.read_csv('path_to_data.csv')
    >>> pair_line = ['line_1', 'line_2']
    >>> PLS_DA_pair(data, pair_line)
    
    This will output the shape of the training data, training and testing accuracy, and display a scatter plot.
    """
    from sklearn.cross_decomposition import PLSRegression
    
    #test on sd data
    #data = sd_27_37.drop(cold,axis=1).dropna()

    # Separate features and target + drop nan 
    X = data[(data['transgenic_line']==pair_line[0]) | (data['transgenic_line']==pair_line[1])]
    y = X["transgenic_line"]
    X = X.drop(["transgenic_line","treatement"], axis=1)

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale the features
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
    
    # Show the processed data
    print(f'\n--- species = {pair_line},\n\t X_train shape : {X_train.shape}, y_train shape{y_train.shape} ---\n')
    # Convert the integer labels to one-hot encoding
    one_hot_encoder = OneHotEncoder(sparse=False)
    y_train_one_hot = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = one_hot_encoder.transform(y_test.reshape(-1, 1))
    #y_train_one_hot
    
    # Initialize the PLS-DA model
    pls_da = PLSRegression(n_components=2)

    # Train the model
    pls_da.fit(X_train, y_train)

    # Transform the training data to get scores (projections onto PLS components)
    scores_train = pls_da.transform(X_train)

    # Transform the testing data
    scores_test = pls_da.transform(X_test)

    # Predict the labels for the training and testing sets
    y_train_pred = np.argmax(pls_da.predict(X_train), axis=1)
    y_test_pred = np.argmax(pls_da.predict(X_test), axis=1)

    # Calculate the accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=scores_train[:, 0], y=scores_train[:, 1], hue=label_encoder.inverse_transform(y_train), palette='Set1', style=label_encoder.inverse_transform(y_train), s=100)
    plt.title('PLS-DA 2D Score Plot (Training Data)')
    plt.xlabel('PLS Component 1')
    plt.ylabel('PLS Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()
    
    print(f'\n--- train accuracy : {train_accuracy}, test accuracy : {test_accuracy} ---\n')


def example_sd_pipeline_dendrogram(df, drr, wli):
    """
    Processes and analyzes datasets to compute and plot a heatmap with a dendrogram 
    depicting the relationships between different features under medium and severe drought 
    conditions for specified days.

    This function performs the following steps:
    1. Basic feature engineering on the input dataframe.
    2. Merging main dataset with wli data.
    3. Computing ratios and merging drr datasets for medium/severe drought conditions for specified days.
    4. Merging datasets on two specified periods.
    5. Preprocessing the merged dataset for dendrogram computation.
    6. Computing and plotting a heatmap with a dendrogram based on specified features.

    Parameters
    ----------
    df : pandas.DataFrame
        The main dataset. 
        Example:
        pd.read_csv("/Users/mac/Downloads/Tdata_Amiens.csv", sep=';', decimal=',', na_values='#N/A')
    
    drr : pandas.DataFrame
        DRR dataset (not directly used in this function but might be used in the underlying functions).
        Example:
        pd.read_csv('/Users/mac/Downloads/drr1_drr2_3rd.csv', sep=';', decimal=',', na_values='#N/A')
    
    wli : pandas.DataFrame
        WLI dataset used for merging with the main dataset.
        Example:
        pd.read_csv('/Users/mac/Downloads/wli.csv', sep=';', decimal=',', na_values='#N/A')

    Returns
    -------
    matplotlib.figure.Figure
        A figure object displaying the heatmap with a dendrogram depicting the relationships between 
        specified features under medium and severe drought conditions for specified days.

    Examples
    --------
    >>> df = pd.read_csv("/Users/mac/Downloads/Tdata_Amiens.csv" , sep=';', decimal=',', na_values='#N/A')
    >>> drr = pd.read_csv('/Users/mac/Downloads/drr1_drr2_3rd.csv', sep=';', decimal=',', na_values='#N/A')
    >>> wli = pd.read_csv('/Users/mac/Downloads/wli.csv', sep=';', decimal=',', na_values='#N/A')
    >>> h = example_sd_pipeline_dendrogram(df, drr, wli)
    """
    
    data = "/Users/mac/Downloads/Tdata_Amiens.csv" # Put your data between the triple quotes
    df = pd.read_csv(data, sep=';', decimal=',', na_values='#N/A')
    #basic feature engineering
    df = feature_eng_basic(df)
    
    drr = pd.read_csv('/Users/mac/Downloads/drr1_drr2_3rd.csv', sep=';', decimal=',', na_values='#N/A')
    wli = pd.read_csv('/Users/mac/Downloads/wli.csv', sep=';', decimal=',', na_values='#N/A')
    wli['plant id 2'] = wli['plant id 2'].rename('plant ID 2')
    #merge main dataset with wli data
    df = merge_wli(df,wli)

    #compute ratio/WW for all features and split dataset into medium/severe drought for a given day 
    sd_27, _ = compute_ratios_and_merge(df, '/Users/mac/Downloads/drr_22_to_37.csv', time_window={"27": [27]}, day=27)
    sd_37, _ = compute_ratios_and_merge(df, '/Users/mac/Downloads/drr_22_to_37.csv', time_window={"37": [37]}, day=37)
    #merge the two datasets on the two periods 
    merge_df_sd = merge_data_period(sd_27,sd_37)
    #change columns names before compute the heatmap 
    merge_df_sd = preprocess_dataset_dendrogram(merge_df_sd)
    #compute and plot the heatmap 
    h = plot_heatmap_with_dendrogram_2d_ratio(merge_df_sd,'severe drought 37',37,['Hue 3_E', 'Hue 4_E', 'Hue 5_E',
           'Hue 6_E', 'Fo_E', 'Fm_E', 'WLI/NDVI_E','WLI/NDVI_L','MCARI1_L','MCARI1_E',
           'QY_max_E', 'NDVI2_E', 'PRI_E', 'NDVI_E', 'PSRI_E','SIPI_L','SIPI_E','Fv_E','Fv_L',
           'OSAVI_E', 'Slenderness_E', 'drr_E', 'drr_L',
           'Slenderness_L', 'Fo_L', 'Fm_L', 'QY_max_L', 'NDVI2_L', 'PRI_L',
           'NDVI_L', 'PSRI_L', 'OSAVI_L', 'Hue 3_L', 'Hue 4_L', 'Hue 5_L',
           'Hue 6_L'])
    return h 
    

def calculate_mean_se(dataframe):
    """
    Calculate the mean and standard error of all "ratio_" parameters for each transgenic line.
    
    This function takes a pandas DataFrame as input, filters columns starting with "ratio_" and 
    includes the "transgenic_line" column. It then groups the DataFrame by "transgenic_line", and 
    calculates the mean and standard error for all "ratio_" parameters within each group.
    
    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing "ratio_" parameters and a "transgenic_line" column.
    
    Returns:
    pd.DataFrame: A multi-index DataFrame with mean and standard error values for each "ratio_" parameter 
                  and transgenic line. Rows represent transgenic lines and columns represent "ratio_" parameters 
                  with two sub-columns: "mean" and "se" (standard error).
                  
     Examples
    --------
    >>> day=27
    >>> sd_27, md = compute_ratios_and_merge(df, '/Users/mac/Downloads/drr_22_to_37.csv', time_window={"27": [day]}, day=day)
    >>> res = calculate_mean_se(sd_27)

    """
    # Extract columns that start with "ratio_" and "transgenic_line"
    ratio_columns = [col for col in dataframe.columns if col.startswith("ratio_")]+['drr1','drr2']
    df_ratio = dataframe[ratio_columns + ["transgenic_line"]]
    
    # Group by "transgenic_line"
    grouped = df_ratio.groupby("transgenic_line")
    
    # Calculate mean and standard error for each group
    result_data = {}
    for name, group in grouped:
        mean_values = group.mean()
        std_values = group.std()
        se = std_values / np.sqrt(len(group))
        result_data[name] = pd.DataFrame({"mean": mean_values, "se": se})
    
    # Concatenate the results into a multi-index DataFrame
    result_df = pd.concat(result_data, axis=1)
    
    return result_df






