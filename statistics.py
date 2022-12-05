import pandas as pd
from scipy.stats import f_oneway
# import tukey hsd
from statsmodels.stats.multicomp import MultiComparison


def anova_on_df(df: pd.DataFrame, group_names: list[str], measurements: list[str],
                measurement_type: str = 'mean') -> list[str]:
    """ANOVA of given variables between given groups in a dataframe. 
    If significant, Tukey's HSD is performed.
    
    Args
        df (pd.DataFrame): dataframe containing the data
        group_names (list[str]): list of group names to compare between
        measurements (list[str]): list of measurements to compare between groups
        measurement_type (str): What type of measurement to take. Options are 'mean', 'median', 'std'
    
    Returns
        list[str]: list of significant measurements
    """
    significant_measurements = []
    for measurement in measurements:
        f, p = f_oneway(*[df[df['group'] == group][f'{measurement_type} {measurement}'] for group in group_names])
        
        if p < 0.05:
            print(f"ANOVA for {measurement_type} {measurement}")
            print("Significant difference between groups")
            print(f'f: {f}, p: {p}')
            significant_measurements.append(measurement)
    
    return significant_measurements

def tukey_hsd_on_df(df: pd.DataFrame, group_names: list[str], measurements: list[str],
                    measurement_type: str = 'mean') -> None:
    """Tukey's HSD of given variables between given groups in a dataframe.
    
    Args
        df (pd.DataFrame): dataframe containing the data
        group_names (list[str]): list of group names to compare between
        measurements (list[str]): list of measurements to compare between groups
        measurement_type (str): What type of measurement to take. Options are 'mean', 'median', 'std'
    """
    for measurement in measurements:
        print(f"Tukey's HSD for {measurement_type} {measurement}")
        mc = MultiComparison(df[f'{measurement_type} {measurement}'], df['group'])
        print(mc.tukeyhsd().summary())
