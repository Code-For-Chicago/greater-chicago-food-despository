from src.config import CENSUS_KEY
import jenkspy
import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Collection
from src import data


def download_census_data(geo_ls=["zip", "county"]) -> None:
    '''
    Top level function to run the queries
    '''

    # Census tables
    detailed_table = 'https://api.census.gov/data/2018/acs/acs5?'
    subject_table = 'https://api.census.gov/data/2018/acs/acs5/subject?'

    # define race instance
    # Values name format: topic_property_subproperty...
    # B03002_003E: Does not include people of hispanic/latino origin
    race_variables = {'B03002_001E': 'race_total',
                      'B03002_005E': 'race_native',
                      'B03002_004E': 'race_black', 'B03002_003E': 'race_white',
                      'B03002_009E': 'race_twoplus_total',
                      'B03002_007E': 'race_pacific',
                      'B03002_008E': 'race_other', 'B03002_006E': 'race_asian',
                      'B03002_012E': 'race_hispaniclatino_total'}
    # race_functions = [processRaceData]
    # variable does not need to be defined, but it is for readability
    race = CensusRequest("race", detailed_table, race_variables)

    # define poverty instance
    poverty_variables = {'S1701_C01_001E': 'poverty_population_total',
                         'S1701_C02_001E': 'poverty_population_poverty',
                         'S1701_C02_002E': 'poverty_population_poverty_child'}
    # If additional subdivision are needed
    # 'S1701_C02_003E' = AGE!!Under 18 years!! Under 5 years!!
    # 'S1701_C02_004E' = AGE!!Under 18 years!! 5 to 17 years!!
    # poverty_functions = [processPovertyData]
    poverty = CensusRequest("poverty", subject_table, poverty_variables)

    get_and_save_census_data([race, poverty],
                             dump_output_path='final_jsons/df_dump.json',
                             merged_output_path='final_jsons/df_merged_json.json',  # noqa: E501
                             geo_ls=geo_ls)


def get_census_response(table_url: str,
                        get_ls: Collection[str],
                        geo: str) -> List[List[str]]:
    '''
    Concatenates url string and returns response from census api query
    input:
        table_url (str): census api table url
        get_ls (ls): list of tables to get data from
        geo (str): geographic area and filter
    output:
        list of rows. first row is header:
            [NAME, <elements of get_ls>, state, county/zip header]
    '''
    if geo == 'zip':
        geo_parameter = 'zip code tabulation area:*'
    elif geo == 'county':
        geo_parameter = 'county:*&in=state:17'
    else:
        raise ValueError('Unsupported geography type: ' + geo)

    get = 'NAME,' + ",".join(get_ls)
    url = f'{table_url}get={get}&for={geo_parameter}&key={CENSUS_KEY}'
    response = requests.get(url)
    try:
        data_table = response.json()
    except json.JSONDecodeError:
        print("Error reading json from census response. Make sure you have a valid census key. Census Response: " + response.text)  # noqa: E501
        data_table = []
    return data_table


def create_percentages(df: pd.DataFrame, total_col_str: str) -> pd.DataFrame:
    '''
    Calculates percentages and removes NaN for dict conversion
    Returns calculated percent_df
    '''
    # divides df by total column to calculate percentages
    divide_by_total = lambda x: x / df[total_col_str]  # noqa: E731, E501
    # Casts to type float64 for numpy interoperability
    percent_df: pd.DataFrame = df.apply(divide_by_total) \
                                 .drop(total_col_str, axis=1) \
                                 .astype('float64')

    # Rounds to save space
    percent_df = np.round(percent_df, 6)

    return percent_df


class CensusRequest:
    def __init__(self,
                 metric: str,
                 table_url: str,
                 variables: Dict[str, str]) -> None:
        self.metric = metric
        self.table_url = table_url
        self.variables = variables


def county_fips(reverse=False) -> dict:
    '''
    Requests county fips from census API and returns list of IL county FIPS
    input: reverse (bool) reverses keys and values in output
    output: il_json (dict) {'county name': 'fip'}
    '''

    url = 'https://api.census.gov/data/2010/dec/sf1?get=NAME&for=county:*'

    response = requests.get(url)

    # filter for IL
    def il_county_filter(x):
        return x[1] == '17'

    response_json = response.json()

    if reverse:
        il_json = {county[1] + county[2]: county[0]
                   for county in response_json if il_county_filter(county)}
    else:
        il_json = {county[0]: county[1] + county[2]
                   for county in response_json if il_county_filter(county)}

    return il_json


def calculate_natural_breaks_bins(df: pd.DataFrame, bin_count: int,
                                  column_names: List[str]) -> pd.DataFrame:  # noqa: 501
    """
    :param df: Pandas dataframe.
    :param bin_count: Number of bins used to classify data.
    :param column_names: dataframe column names used to calculate breaks
    :return: Dictionary of column name and list of bin cutoff limits.
    """
    bin_dict = {}
    for cn in column_names:
        column_data = df[cn].dropna().to_list()
        bin_dict[cn] = jenkspy.jenks_breaks(column_data, nb_class=bin_count)
    return pd.DataFrame(bin_dict)


def majority(series: pd.Series) -> Optional[str]:
    '''
    Returns majority race demographic
    for each geo_area
    If no majority, returns 'majority_minority'
    '''
    # indexes max value, returns NA for NA rows
    idx = series.idxmax()
    try:
        value = series[idx]
    except KeyError:
        # if NA row, idx = NA
        return None
    if value >= 0.5:
        return idx
    else:
        return 'majority_minority'


def dataframe_and_bins_from_census_rows(all_rows: List[List[str]], geography_type: str, request: CensusRequest) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[float]]]]:  # noqa: E501

    # Translate the census variables into our descriptive names
    columns = [request.variables.get(column_name, column_name)
               for column_name in all_rows[0]]

    dataframe = pd.DataFrame(columns=columns, data=all_rows[1:])

    # Without forcing the types, the numbers end up as strings
    dataframe = dataframe.astype('string')  # without this, you get "TypeError: object cannot be converted to an IntegerDtype" during when we convert to integers  # noqa: E501
    conversion_dict = {v: "Int64" for v in request.variables.values()}
    dataframe = dataframe.astype(conversion_dict)

    bins = {}

    if geography_type == "county":
        fip_series = dataframe.loc[:, 'state'] + dataframe.loc[:, 'county']
        fip_series.rename('FIPS', inplace=True)
        dataframe = pd.concat([dataframe, fip_series], axis=1)
        dataframe = dataframe.set_index('FIPS').drop(['state', 'county'], axis=1)  # noqa: E501
    elif geography_type == "zip":
        dataframe = dataframe.set_index('zip code tabulation area') \
                 .drop(['NAME', 'state'], axis=1) \
                 .filter(regex='^(6[0-2])[0-9]+', axis=0)  # noqa: E501
    else:
        raise ValueError("Unsupported geography type: " + geography_type)

    if request.metric == "race":
        race_df = dataframe.loc[:, request.variables.values()]  # we only need the numeric columns  # noqa: E501
        pct_df = create_percentages(race_df, 'race_total')
        # creates series of majority race demographics
        majority_series = pct_df.apply(majority, axis=1)
        majority_series.name = 'race_majority'

        # converts NAN to None, for proper JSON encoding
        working_df = pct_df.where(pd.notnull(pct_df), None)
        # creates series of race percentages as a dictionary
        # this allows us to add percentages to the main table,
        # without adding many more columns
        pct_dict_series = working_df.apply(pd.Series.to_dict, axis=1)
        pct_dict_series.name = 'race_percentages'

        # creates df from the series for merging
        pct_and_majority_df = pd.concat([pct_dict_series,
                                         majority_series], axis=1)

        dataframe = dataframe.merge(pct_and_majority_df,
                                    left_index=True, right_index=True,
                                    suffixes=(False, False))

    elif request.metric == "poverty":
        poverty_df = dataframe.loc[:, request.variables.values()]  # we only need the numeric columns   # noqa: E501
        pct_df = create_percentages(poverty_df, 'poverty_population_total')

        # converts NAN to None, for proper JSON encoding
        working_df = pct_df.where(pd.notnull(pct_df), None)
        # creates series of poverty percentages as a dictionary
        # this allows us to add percentages to the main table,
        # without adding many more columns
        pct_dict_series = working_df.apply(pd.Series.to_dict, axis=1)
        pct_dict_series.name = 'poverty_percentages'

        dataframe = dataframe.merge(pct_dict_series,
                                    left_index=True, right_index=True,
                                    suffixes=(False, False))

        quantile_df = pct_df.apply(np.quantile, q=(0, 0.25, 0.5, 0.75, 1))  # noqa: E501
        # round for space and avoid floating point imprecision
        quantile_df = np.round(quantile_df, 6)
        quantile_dict = quantile_df.to_dict(orient='list')

        # create quantile bins using natural breaks algorithm.
        # bin_count could be increased to > 4 if needed.
        natural_breaks_df = calculate_natural_breaks_bins(pct_df, bin_count=4,
                                                          column_names=["poverty_population_poverty",  # noqa: E501
                                                                        "poverty_population_poverty_child"])  # noqa: E501
        natural_breaks_df = np.round(natural_breaks_df, 6)
        natural_breaks_dict = natural_breaks_df.to_dict(orient='list')

        bins = {
            'quantiles': quantile_dict,
            'natural_breaks': natural_breaks_dict
        }
    else:
        raise ValueError("Unsupported metric type: " + geography_type)

    return dataframe, bins


def get_census_data(request: CensusRequest, geography_type: str) -> data.GCFDData:  # noqa: E501
    census_rows = get_census_response(request.table_url,
                                      request.variables.keys(),
                                      geography_type)
    dataframe, bins = dataframe_and_bins_from_census_rows(census_rows,
                                                          geography_type,
                                                          request)

    if geography_type == "county":
        wrapper = data.from_county_dataframe(dataframe)
    elif geography_type == "zip":
        wrapper = data.from_zip_dataframe(dataframe)
    else:
        raise ValueError("Unsupported geography type: " + geography_type)

    wrapper.meta.data_metrics[request.metric] = request.variables
    wrapper.meta.data_bins = bins

    return wrapper


def get_and_save_census_data(data_requests: List[CensusRequest],
                             dump_output_path: str = "",
                             merged_output_path: str = "",
                             geo_ls: List[str] = ["zip", "county"],
                             pretty_print: bool = False) -> None:

    combined_data = data.GCFDData()

    for request in data_requests:
        for geo in geo_ls:
            new_data = get_census_data(request, geo)
            combined_data = data.combine(combined_data, new_data)

    if dump_output_path != "":
        with open(dump_output_path, "w") as f:
            f.write(data.to_json(combined_data, pretty_print=pretty_print))

    if merged_output_path != "":
        merged_data = data.merge(combined_data)
        with open(merged_output_path, "w") as f:
            f.write(data.to_json(merged_data, pretty_print=pretty_print))
