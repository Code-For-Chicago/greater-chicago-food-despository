import pandas as pd
from src.census_response import calculate_natural_breaks_bins, CensusRequest, get_and_save_census_data  # noqa: E501


def assert_file_contents_equal(path_a: str, path_b: str):
    with open(path_a) as file_a:
        with open(path_b) as file_b:
            assert file_a.read() == file_b.read()


def test_race_dump():
    geo_ls = ["zip", "county"]

    detailed_table = 'https://api.census.gov/data/2018/acs/acs5?'

    race_metrics = {'B03002_001E': 'race_total',
                    'B03002_005E': 'race_native'}
    race = CensusRequest('race', detailed_table, race_metrics)

    actual_output_path = "final_jsons/census_race_dump_actual_output.json"
    get_and_save_census_data([race], dump_output_path=actual_output_path, geo_ls=geo_ls)  # noqa: E501

    assert_file_contents_equal(actual_output_path, "tests/resources/census_race_dump_expected_output.json")  # noqa: E501


def test_race_merged():
    geo_ls = ["zip", "county"]

    detailed_table = 'https://api.census.gov/data/2018/acs/acs5?'

    race_metrics = {'B03002_001E': 'race_total',
                    'B03002_005E': 'race_native'}
    race = CensusRequest('race', detailed_table, race_metrics)

    actual_output_path = "final_jsons/census_race_merged_actual_output.json"
    get_and_save_census_data([race], merged_output_path=actual_output_path, geo_ls=geo_ls)  # noqa: E501

    assert_file_contents_equal(actual_output_path, "tests/resources/census_race_merged_expected_output.json")  # noqa: E501


def test_poverty_dump():
    geo_ls = ["zip", "county"]

    subject_table = 'https://api.census.gov/data/2018/acs/acs5/subject?'

    poverty_metrics = {'S1701_C01_001E': 'poverty_population_total',
                       'S1701_C02_001E': 'poverty_population_poverty',
                       'S1701_C02_002E': 'poverty_population_poverty_child'}
    poverty = CensusRequest('poverty', subject_table, poverty_metrics)

    actual_output_path = "final_jsons/census_poverty_dump_actual_output.json"
    get_and_save_census_data([poverty], dump_output_path=actual_output_path, geo_ls=geo_ls)  # noqa: E501

    assert_file_contents_equal(actual_output_path, "tests/resources/census_poverty_dump_expected_output.json")  # noqa: E501


def test_poverty_merged():
    geo_ls = ["zip", "county"]

    subject_table = 'https://api.census.gov/data/2018/acs/acs5/subject?'

    poverty_metrics = {'S1701_C01_001E': 'poverty_population_total',  # noqa: E501
                       'S1701_C02_001E': 'poverty_population_poverty',  # noqa: E501
                       'S1701_C02_002E': 'poverty_population_poverty_child'}  # noqa: E501
    poverty = CensusRequest('poverty', subject_table, poverty_metrics)

    actual_output_path = "final_jsons/census_poverty_merged_actual_output.json"
    get_and_save_census_data([poverty], merged_output_path=actual_output_path, geo_ls=geo_ls)  # noqa: E501

    assert_file_contents_equal(actual_output_path, "tests/resources/census_poverty_merged_expected_output.json")  # noqa: E501


def test_calculate_natural_breaks_bins_correctly_categorizes_valid_data():
    test_zip_pct_df = {
        "poverty_population_poverty": {
            "60002": 0.075331, "60004": 0.047824, "60005": 0.079041,
            "60007": 0.038282, "60008": 0.062858, "60010": 0.042144,
            "60012": 0.024543, "60013": 0.064643, "60014": 0.066453,
            "60015": 0.044636, "60016": 0.091621, "60018": 0.138853,
            "60020": 0.087967, "60021": 0.036229, "60022": 0.024301,
            "60025": 0.061288,
        },
        "poverty_population_poverty_child": {
            "60002": 0.01978, "60004": 0.013142, "60005": 0.016204,
            "60007": 0.007048, "60008": 0.014999, "60010": 0.010662,
            "60012": 0.007078, "60013": 0.022712, "60014": 0.021133,
            "60015": 0.009923, "60016": 0.022829, "60018": 0.054427,
            "60020": 0.014337, "60021": 0.0, "60022": 0.001416,
            "60025": 0.018359,
        }
    }
    df = pd.DataFrame.from_dict(test_zip_pct_df)
    column_names = ["poverty_population_poverty",
                    "poverty_population_poverty_child"]
    actual = calculate_natural_breaks_bins(df, 4, column_names).to_dict(orient="list")  # noqa: E501

    expected = {
        'poverty_population_poverty': [
            0.024301, 0.047824, 0.066453, 0.091621, 0.138853],
        'poverty_population_poverty_child': [
            0.0, 0.007078, 0.016204, 0.022829, 0.054427]
    }

    assert actual == expected


def test_calculate_natural_breaks_removes_not_a_number_values_when_calculating_bins():  # noqa: E501
    test_zip_pct_with_nan_df = {
        "poverty_population_poverty": {
            "60002": 0.075331, "60004": None, "60005": 0.079041,
            "60007": 0.038282, "60008": 0.062858, "60010": 0.042144,
            "60012": None, "60013": 0.064643, "60014": None, "60015": 0.044636,
            "60016": 0.091621, "60018": 0.138853, "60020": 0.087967,
            "60021": 0.036229, "60022": 0.024301, "60025": None,
        },
        "poverty_population_poverty_child": {
            "60002": None, "60004": None, "60005": None, "60007": None,
            "60008": 0.014999, "60010": 0.010662, "60012": 0.007078,
            "60013": 0.022712, "60014": 0.021133, "60015": 0.009923,
            "60016": 0.022829, "60018": 0.054427, "60020": 0.014337,
            "60021": 0.0, "60022": 0.001416, "60025": 0.018359,
        }
    }
    df = pd.DataFrame.from_dict(test_zip_pct_with_nan_df)

    column_names = ["poverty_population_poverty",
                    "poverty_population_poverty_child"]
    actual = calculate_natural_breaks_bins(df, 2, column_names)

    expected = {
        'poverty_population_poverty': [0.024301, 0.064643, 0.138853],
        'poverty_population_poverty_child': [0.0, 0.022829, 0.054427]
    }

    assert actual.to_dict(orient="list") == expected
