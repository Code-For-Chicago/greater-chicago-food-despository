'''
Test/Demo for data implementation

Read in DF, create DataObjects, tests export
'''
import os
import sys

# Module Not Found Error without this code
sys.path.append(os.path.abspath(''))

import pandas as pd  # noqa: E402
from src import data  # noqa: E402


# def read_data():
#     assert GCFDData.get_data() == {}
#     base = pd.read_pickle('./tests/resources/data_60002_base.xz')
#     pct = pd.read_pickle('./tests/resources/data_60002_pct.xz')
#     GCFDData('zip', base, fp="./tests/resources/base_obj.pkl")
#     GCFDData('poverty_percentages', pct,
#              parent='zip', fp='./tests/resources/pct_obj.pkl')
#     return GCFDData


# def test_pickle_output():
#     read_data()
#     data_copy_zip = GCFDData.get_data()['zip']
#     GCFDData.clear_data()
#     assert GCFDData.get_data() == {}
#     GCFDData.load_data("./tests/resources/base_obj.pkl")
#     data_new_zip = GCFDData.get_data()['zip']
#     assert type(data_copy_zip) == type(data_new_zip)
#     assert data_copy_zip.to_dict() == data_new_zip.to_dict()
#     GCFDData.clear_data()


# def test_export_output():
#     read_data()
#     fp = './tests/resources/test_data_class_new.json'
#     GCFDData.export_data(fp)
#     with open('./tests/resources/test_data_class_main.json') as main:
#         with open(fp) as f:
#             assert main.read() == f.read()
#     GCFDData.clear_data()

def test_combine():
    race_data = data.Wrapper()
    race_data.zip = { "race_total": { "60002": 24066 } }
    race_data.county = { "race_total": { "17001": 66427 } }
    race_data.meta.data_metrics = {"race": { "B03002_001E": "race_total" } }
    race_data.meta.data_bins = {
        "quantiles": { "race_total": [0.1, 0.2, 0.3, 0.4, 0.5] },
        "natural_breaks": { "race_total": [0.0, 0.2, 0.4, 0.6, 0.8] } }

    poverty_data = data.Wrapper()
    poverty_data.zip = { "poverty_population_total": { "60002": 24014 } }
    poverty_data.county = { "poverty_population_total": { "17001": 64844 } }
    poverty_data.meta.data_metrics = {"poverty": { "S1701_C01_001E": "poverty_population_total" } }
    poverty_data.meta.data_bins = {
        "quantiles": { "poverty_population_total": [0.1, 0.3, 0.5, 0.7, 0.9] },
        "natural_breaks": { "poverty_population_total": [0.6, 0.7, 0.8, 0.9, 0.11] } }

    combined_data = data.combine(race_data, poverty_data)
    assert combined_data.zip["race_total"]["60002"] == 24066
    assert combined_data.zip["poverty_population_total"]["60002"] == 24014
    assert combined_data.county["race_total"]["17001"] == 66427
    assert combined_data.county["poverty_population_total"]["17001"] == 64844
    assert combined_data.meta.data_metrics["race"]["B03002_001E"] == "race_total"
    assert combined_data.meta.data_metrics["poverty"]["S1701_C01_001E"] == "poverty_population_total"
    assert combined_data.meta.data_bins["quantiles"]["race_total"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert combined_data.meta.data_bins["natural_breaks"]["race_total"] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert combined_data.meta.data_bins["quantiles"]["poverty_population_total"] == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert combined_data.meta.data_bins["natural_breaks"]["poverty_population_total"] == [0.6, 0.7, 0.8, 0.9, 0.11]


def test_dump_json():

    some_data = data.Wrapper()
    some_data.zip = { "race_total": { "60002": 1234 } }
    some_data.county = { "race_total": { "17001": 5678 } }
    some_data.meta.data_metrics = {"race": { "B03002_001E": "race_total" } }
    some_data.meta.data_bins = {
        "quantiles": { "race_total": [0.1, 0.2, 0.3, 0.4, 0.5] },
        "natural_breaks": { "race_total": [0.0, 0.2, 0.4, 0.6, 0.8] } }

    json_str = data.json_dump(some_data, pretty_print=True)
    with open('./tests/resources/data_dump.json') as expected:
        assert expected.read() == json_str


def test_from_dataframe():
    df = pd.DataFrame({"fips": ["17001"], "race_total": [1234]})
    df.set_index("fips", inplace=True)
    wrapper = data.from_county_dataframe(df)
    assert wrapper.county["race_total"]["17001"] == 1234


def test_merge():
    some_data = data.Wrapper()
    some_data.meta.data_metrics = {"race": { "B03002_001E": "race_total" } }
    some_data.meta.data_bins = { "quantiles": { "race_total": [0.1, 0.2, 0.3, 0.4, 0.5] } }
    some_data.county = { "race_native": { "17001": 1234 }, "race_total": { "17001": 5678 } }
    some_data.zip = { "race_native": { "60002": 1357 }, "race_total": { "60002": 2468 }  }

    merged_data = data.merge(some_data)
    assert merged_data.meta.data_metrics == some_data.meta.data_metrics
    assert merged_data.meta.data_bins == some_data.meta.data_bins
    assert merged_data.county_data["17001"]["race_data"]["race_native"] == 1234
    assert merged_data.county_data["17001"]["race_data"]["race_total"] == 5678
    assert merged_data.zip_data["60002"]["race_data"]["race_native"] == 1357
    assert merged_data.zip_data["60002"]["race_data"]["race_total"] == 2468
