# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% The purpose of this script is to conduct Data Integration & Transformation, Data Cleaning, &
# Data Preparation before developing Hurricane Idalia's Discrete Event Simulation (DES) Model 

#%%
# Importing Pacakges
import pandas as pd
import numpy as np
pd.options.display.max_rows = 9999

#%%
###############################################Data Intergration & Transformation######################################
# Importing Critcal Facilities (mobile logistic nodes (mln_df), planned network path locations (pnpl_df)
# logistic nodes (ln_df), evacuation nodes (en_df), and evacuation mean (em_df) ), Highwater data (hw)
# and Sensor data (sn)) csv files
hw_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Hurricane_Idalia_High-water_Mark_Data.csv')
sn_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Hurrican_Idalia_Sensor_Data.csv')
mln_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Mobile_Logistic_Nodes/Bus Terminals.csv')
em_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Evacuation_Nodes/Evacuation_Mean_(mobile_speed_capacity).csv')
en_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Evacuation_Nodes/Evacuation_Nodes(evacuation_zones_by_county).csv')

#%%
# CSV's for final table pnpl_df = coast guard (cg_df), fire stations (fs_df), food banks (fb_df)
# and national guard (ng_df) datasets
cg_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Planned_Network_Path_Loctions/Coast Guard.csv')
fs_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Planned_Network_Path_Loctions/Fire Stations.csv')
fb_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Planned_Network_Path_Loctions/Food Banks.csv')
ng_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Planned_Network_Path_Loctions/National Guard.csv')

#%%
# CSV's for final table ln_df = Airports (private) (airpri_df), airports public (airpub_df)
# disaster recovery centers mobile (drcm_df), disaster recovery centers (drc_df), 
# private schools (prisch_df), public schools (pubsch_df), emergency medical facilities (emf_df)
# emergency operational centers (eoc_df), healthcare facilities (hf_df), hospitals (hos_df), 
# relief agencies (ra_df), risk shelter inventory general (rsig_df), risk shelter inventory pet friendly (rsipf_df)
# and and risk shelter inventory special needs (rsisn_df) 
airpri_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Airports - Private.csv')
airpub_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Airports - Public.csv')
drcm_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Disaster Recovery Centers - Mobile.csv')
drc_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Disaster Recovery Centers.csv')
prisch_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Private Schools.csv')
pubsch_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Public Schools.csv')
emf_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Emergency Medical Services.csv')
eoc_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Emergency Operations Center.csv')
hf_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Healthcare Facilities.csv')
hos_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Hospitals.csv')
ra_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Relief Agencies.csv')
rsig_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Risk Shelter Inventory - General.csv')
rsipf_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Risk Shelter Inventory - Pet Friendly.csv')
rsisn_df = pd.read_csv('/Users/dequeishadiggs/Documents/DIS-9902A/Chapter3_Week4/Notes/Data/CSV/Critical Facilities/Logistic_Nodes/Risk Shelter Inventory - Special Needs.csv')

#%%
############################################Data Cleaning & Data Prepration############################################
# Conducting Data Cleaning Processes removing unnessary columns, filtering only nessary columns
# renaming columns, and reformating lat long for hw_df, sn_df, mln_df, and CSV files for ppl_df
hw_df = hw_df.drop(columns =['latitude_dd', 'longitude_dd', 'site', 'vertical_datums', 'flag_member', 'Links'
                            , 'approval', 'marker', 'survey_member', 'peak_summary', 'verticalDatumName', 'verticalMethodName'
                            , 'approvalMember', 'markerName', 'flagMemberName', 'site_no', 'sitePriorityName', 'networkNames'
                            , 'survey_date', 'approval_id', 'marker_id', 'peak_summary_id', 'survey_member_id', 'last_updated'
                            , 'last_updated_by', 'horizontal_datums', 'vertical_collect_methods', 'horizontal_collect_methods'
                            , 'hwm_types', 'hwm_qualities', 'event', 'siteZone','surveyMemberName', 'files', 'bank', 'flag_member_id'])
sn_df = sn_df.drop(columns = ['housingType', 'Links', 'site', 'event', 'files', 'data_files', 'instrument_status', 'deployment_type'
                              , 'siteHDatum', 'sitePriorityName', 'siteZone', 'deployment_type_id', 'location_description', 'serial_number'
                              , 'housing_serial_number', 'interval', 'inst_collection_id', 'housing_type_id', 'sensor_brand_id', 'vented'
                              , 'last_updated', 'last_updated_by', 'instr_collection_conditions', 'sensor_brand', 'sensor_type', 'housing_type', 'event_id'
                              ,'site_id', 'instrument_id'])
mln_df = mln_df.drop(columns = ['Address', 'Zip', 'USNG', 'OWNERSHIP', 'PARCELID', 'x', 'y', 'LONGLAT'])
cg_df = cg_df.drop(columns = ['Address', 'Zip', 'USNG', 'OWNERSHIP','PARCELID', 'x', 'y', 'LONGLAT'])

# Renaming X and Y columns to Longitude (X) and Latitude (Y)
mln_df = mln_df.rename(columns = {'X':'mln_Longitude', 'Y': 'mln_Latitude'})
cg_df = cg_df.rename(columns = {'X':'cg_Longitude', 'Y': 'cg_Latitude', 'City': 'cg_City', 'Name':'cg_Name'
                                , 'COUNTY': 'cg_COUNTY', 'FACILITY_TYPE' : 'cg_FACILITY_TYPE'})

#%%
# Filtering fs_df to keep only the nessary columns
fs_df = fs_df.loc[:, ['CITY', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming coumns
fs_df = fs_df.rename(columns = {'CITY': 'fs_CITY', 'COUNTY': 'fs_COUNTY'
                                 , 'FACILITY_TYPE' : 'fs_FACILITY_TYPE'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
fs_Longitude = []
fs_Latitude = []

# For each row in a varible,
for row in fs_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to fs_Longitude
        fs_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to fs_Latitude
        fs_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to fs_Longitide
        fs_Longitude.append(np.NaN)
        # append a missing value to fs_Latitude
        fs_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
fs_df['fs_Longitude'] = fs_Longitude
fs_df['fs_Latitude'] = fs_Latitude

# Dropping unnessary columns
fs_df = fs_df.drop(columns = ['LONGLAT'])

#%%
# Filtering fb_df to keep only the nessary columns
fb_df = fb_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
fb_Longitude = []
fb_Latitude = []

# For each row in a varible,
for row in fb_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to fb_Longitude
        fb_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to fb_Latitude
        fb_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to fb_Longitide
        fb_Longitude.append(np.NaN)
        # append a missing value to fb_Latitude
        fb_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
fb_df['fb_Longitude'] = fb_Longitude
fb_df['fb_Latitude'] = fb_Latitude

# Dropping unnessary columns
fb_df = fb_df.drop(columns = ['LONGLAT'])

# Renaming coumns
fb_df = fb_df.rename(columns = {'City': 'fb_CITY', 'COUNTY': 'fb_COUNTY'
                                 , 'FACILITY_TYPE' : 'fb_FACILITY_TYPE'})

#%%
# Filtering to keep only the nessary columns
ng_df = ng_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
ng_Longitude = []
ng_Latitude = []

# For each row in a varible,
for row in ng_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to ng_Longitude
        ng_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to ng_Latitude
        ng_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to ng_Longitide
        ng_Longitude.append(np.NaN)
        # append a missing value to ng_Latitude
        ng_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
ng_df['ng_Longitude'] = ng_Longitude
ng_df['ng_Latitude'] = ng_Latitude

# Dropping unnessary columns
ng_df = ng_df.drop(columns = ['LONGLAT'])

# Renaming coumns
ng_df = ng_df.rename(columns = {'City': 'ng_CITY', 'COUNTY': 'ng_COUNTY'
                                 , 'FACILITY_TYPE' : 'ng_FACILITY_TYPE'})

#%%
# Filtering CSV files for the ln_df datasets to keep only the nessary columns and Renaming columns
airpri_df = airpri_df.loc[:, ['City', 'COUNTY', 'LAT','LONG','NAME_OTHER', 'FACILITY_TYPE'
                      , 'DESCRIPTION', 'FAC_STATUS']]

airpri_df = airpri_df.rename(columns = {'LAT': 'airpri_LAT', 'LONG': 'airpri_LONG', 'City': 'airpri_City'
                                        , 'COUNTY' : 'airpri_COUNTY', 'NAME_OTHER' : 'airpri_NAME_OTHER'
                                        , 'FACILITY_TYPE' : 'airpri_FACILITY_TYPE', 'DESCRIPTION' : 'airpri_DESCRIPTION'
                                        , 'FAC_STATUS' : 'airpri_FAC_STATUS'})

airpub_df = airpub_df.loc[:, ['City', 'COUNTY', 'LAT','LONG','NAME_OTHER', 'FACILITY_TYPE'
                      , 'DESCRIPTION', 'FAC_STATUS']]

airpub_df = airpub_df.rename(columns = {'LAT': 'airpub_LAT', 'LONG': 'airpub_LONG', 'City': 'airpub_City'
                                        , 'COUNTY' : 'airpub_COUNTY', 'NAME_OTHER' : 'airpub_NAME_OTHER'
                                        , 'FACILITY_TYPE' : 'airpub_FACILITY_TYPE', 'DESCRIPTION' : 'airpub_DESCRIPTION'
                                        , 'FAC_STATUS' : 'airpub_FAC_STATUS'})

#%%
# Filtering to keep only the nessary columns
drcm_df = drcm_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
drcm_Longitude = []
drcm_Latitude = []

# For each row in a varible,
for row in drcm_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to drcm_Longitude
        drcm_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to drcm_Latitude
        drcm_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to drcm_Longitide
        drcm_Longitude.append(np.NaN)
        # append a missing value to drcm_Latitude
        drcm_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
drcm_df['drcm_Longitude'] = drcm_Longitude
drcm_df['drcm_Latitude'] = drcm_Latitude

# Dropping unnessary columns
drcm_df = drcm_df.drop(columns = ['LONGLAT'])

# Renaming Columns
drcm_df = drcm_df.rename(columns={'City': 'drcm_City', 'COUNTY' : 'drcm_COUNTY', 'FACILITY_TYPE' : 'drmc_FACILITY_TYPE'})

#%%
# Filtering to keep only the nessary columns
drc_df = drc_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
drc_Longitude = []
drc_Latitude = []

# For each row in a varible,
for row in drc_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to drc_Longitude
        drc_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to drc_Latitude
        drc_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to drc_Longitide
        drc_Longitude.append(np.NaN)
        # append a missing value to drc_Latitude
        drc_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
drc_df['drc_Longitude'] = drc_Longitude
drc_df['drc_Latitude'] = drc_Latitude

# Dropping unnessary columns
drc_df = drc_df.drop(columns = ['LONGLAT'])

# Renaming Columns
drc_df = drc_df.rename(columns={'City': 'drc_City', 'COUNTY' : 'drc_COUNTY', 'FACILITY_TYPE' : 'drc_FACILITY_TYPE'})

#%%
# Filtering to keep only the nessary columns
prisch_df = prisch_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
prisch_Longitude = []
prisch_Latitude = []

# For each row in a varible,
for row in prisch_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to prisch_Longitude
        prisch_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to prisch_Latitude
        prisch_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to prisch_Longitide
        prisch_Longitude.append(np.NaN)
        # append a missing value to prisch_Latitude
        prisch_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
prisch_df['prisch_Longitude'] = prisch_Longitude
prisch_df['prisch_Latitude'] = prisch_Latitude

# Dropping unnessary columns
prisch_df = prisch_df.drop(columns = ['LONGLAT'])

# Renaming Columns
prisch_df = prisch_df.rename(columns={'City': 'prisch_City', 'COUNTY' : 'prisch_COUNTY', 'FACILITY_TYPE' : 'prisch_FACILITY_TYPE'})

#%%
# Filtering to keep only the nessary columns
pubsch_df = pubsch_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
pubsch_Longitude = []
pubsch_Latitude = []

# For each row in a varible,
for row in pubsch_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to pubsch_Longitude
        pubsch_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to pubsch_Latitude
        pubsch_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to pubsch_Longitide
        pubsch_Longitude.append(np.NaN)
        # append a missing value to pubsch_Latitude
        pubsch_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
pubsch_df['pubsch_Longitude'] = pubsch_Longitude
pubsch_df['pubsch_Latitude'] = pubsch_Latitude

# Dropping unnessary columns
pubsch_df = pubsch_df.drop(columns = ['LONGLAT'])

# Renaming Columns
pubsch_df = pubsch_df.rename(columns={'City': 'pubsch_City', 'COUNTY' : 'pubsch_COUNTY', 'FACILITY_TYPE' : 'pubsch_FACILITY_TYPE'})

#%%
# Filtering to keep only the nessary columns
emf_df = emf_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
emf_Longitude = []
emf_Latitude = []

# For each row in a varible,
for row in emf_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to emf_Longitude
        emf_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to emf_Latitude
        emf_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to emf_Longitide
        emf_Longitude.append(np.NaN)
        # append a missing value to emf_Latitude
        emf_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
emf_df['emf_Longitude'] = emf_Longitude
emf_df['emf_Latitude'] = emf_Latitude

# Dropping unnessary columns
emf_df = emf_df.drop(columns = ['LONGLAT'])

# Renaming Columns
emf_df = emf_df.rename(columns={'City': 'emf_City', 'COUNTY' : 'emf_COUNTY', 'FACILITY_TYPE' : 'emf_FACILITY_TYPE'})

#%%
# Filtering to keep only the nessary columns
eoc_df = eoc_df.loc[:, ['City', 'County', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
eoc_Longitude = []
eoc_Latitude = []

# For each row in a varible,
for row in eoc_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to eoc_Longitude
        eoc_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to eoc_Latitude
        eoc_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to eoc_Longitide
        eoc_Longitude.append(np.NaN)
        # append a missing value to eoc_Latitude
        eoc_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
eoc_df['eoc_Longitude'] = eoc_Longitude
eoc_df['eoc_Latitude'] = eoc_Latitude

# Dropping unnessary columns
eoc_df = eoc_df.drop(columns = ['LONGLAT'])

# Renaming Columns
eoc_df = eoc_df.rename(columns={'City': 'eoc_City', 'County' : 'eoc_County', 'FACILITY_TYPE' : 'eoc_FACILITY_TYPE'})

#%%
# Filtering to keep only the nessary columns
hf_df = hf_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'TTL_BEDS']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
hf_Longitude = []
hf_Latitude = []

# For each row in a varible,
for row in hf_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to fb_Longitude
        hf_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to hf_Latitude
        hf_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to hf_Longitide
        hf_Longitude.append(np.NaN)
        # append a missing value to hf_Latitude
        hf_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
hf_df['hf_Longitude'] = hf_Longitude
hf_df['hf_Latitude'] = hf_Latitude

# Dropping unnessary columns
hf_df = hf_df.drop(columns = ['LONGLAT'])

# Renaming coumns
hf_df = hf_df.rename(columns = {'City': 'hf_CITY', 'COUNTY': 'hf_COUNTY'
                                 , 'FACILITY_TYPE' : 'hf_FACILITY_TYPE', 'TTL_BEDS' : 'hf_TTL_BEDS'})

#%%
# Filtering to keep only the nessary columns
hos_df = hos_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'TTL_BEDS']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
hos_Longitude = []
hos_Latitude = []

# For each row in a varible,
for row in hos_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to hos_Longitude
        hos_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to hos_Latitude
        hos_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to hos_Longitide
        hos_Longitude.append(np.NaN)
        # append a missing value to hos_Latitude
        hos_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
hos_df['hos_Longitude'] = hos_Longitude
hos_df['hos_Latitude'] = hos_Latitude

# Dropping unnessary columns
hos_df = hos_df.drop(columns = ['LONGLAT'])

# Renaming coumns
hos_df = hos_df.rename(columns = {'City': 'hos_CITY', 'COUNTY': 'hos_COUNTY'
                                 , 'FACILITY_TYPE' : 'hos_FACILITY_TYPE', 'TTL_BEDS' : 'hos_TTL_BEDS'})

#%%
# Filtering to keep only the nessary columns
ra_df = ra_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
ra_Longitude = []
ra_Latitude = []

# For each row in a varible,
for row in ra_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to ra_Longitude
        ra_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to ra_Latitude
        ra_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to ra_Longitide
        ra_Longitude.append(np.NaN)
        # append a missing value to ra_Latitude
        ra_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
ra_df['ra_Longitude'] = ra_Longitude
ra_df['ra_Latitude'] = ra_Latitude

# Dropping unnessary columns
ra_df = ra_df.drop(columns = ['LONGLAT'])

# Renaming coumns
ra_df = ra_df.rename(columns = {'City': 'ra_CITY', 'COUNTY': 'ra_COUNTY'
                                 , 'FACILITY_TYPE' : 'ra_FACILITY_TYPE'})

#%%
# Filtering to keep only the nessary columns
rsig_df = rsig_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'SHELTER_TYPE', 'Risk_Capacity_Spaces',
                          'Evacuation_Zone', 'General_Pop', 'SPECIAL_NEEDS', 'Pet_Friendly']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
rsig_Longitude = []
rsig_Latitude = []

# For each row in a varible,
for row in rsig_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to rsig_Longitude
        rsig_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to rsig_Latitude
        rsig_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to rsig_Longitide
        rsig_Longitude.append(np.NaN)
        # append a missing value to rsig_Latitude
        rsig_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
rsig_df['rsig_Longitude'] = rsig_Longitude
rsig_df['rsig_Latitude'] = rsig_Latitude

# Dropping unnessary columns
rsig_df = rsig_df.drop(columns = ['LONGLAT'])

# Renaming coumns
rsig_df = rsig_df.rename(columns = {'City': 'rsig_CITY', 'COUNTY': 'rsig_COUNTY', 'SHELTER_TYPE' : 'rsig_SHELTER_TYPE'
                                 , 'FACILITY_TYPE' : 'rsig_FACILITY_TYPE', 'Risk_Capacity_Spaces' : 'rsig_Risk_Capacity_Spaces'
                                 , 'Evacuation_Zone' : 'rsig_Evacuation_Zone', 'General_Pop' : 'rsig_General_Pop'
                                 , 'SPECIAL_NEEDS' : 'rsig_SPECIAL_NEEDS', 'Pet_Friendly' : 'rsig_Pet_Friendly'})

#%%
# Filtering to keep only the nessary columns
rsipf_df = rsipf_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'SHELTER_TYPE', 'Risk_Capacity_Spaces',
                          'Evacuation_Zone', 'General_Pop', 'SPECIAL_NEEDS', 'Pet_Friendly']]

# Reformating LONGLAT so that Longitude and Latitude columns are separate
# Create two lists for the loop results to be placed
rsipf_Longitude = []
rsipf_Latitude = []

# For each row in a varible,
for row in rsipf_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to rsipf_Longitude
        rsipf_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to rsipf_Latitude
        rsipf_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to rsipf_Longitide
        rsipf_Longitude.append(np.NaN)
        # append a missing value to rsipf_Latitude
        rsipf_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
rsipf_df['rsipf_Longitude'] = rsipf_Longitude
rsipf_df['rsipf_Latitude'] = rsipf_Latitude

# Dropping unnessary columns
rsipf_df = rsipf_df.drop(columns = ['LONGLAT'])

# Renaming coumns
rsipf_df = rsipf_df.rename(columns = {'City': 'rsipf_CITY', 'COUNTY': 'rsipf_COUNTY', 'SHELTER_TYPE' : 'rsipf_SHELTER_TYPE'
                                 , 'FACILITY_TYPE' : 'rsipf_FACILITY_TYPE', 'Risk_Capacity_Spaces' : 'rsipf_Risk_Capacity_Spaces'
                                 , 'Evacuation_Zone' : 'rsipf_Evacuation_Zone', 'General_Pop' : 'rsipf_General_Pop'
                                 , 'SPECIAL_NEEDS' : 'rsipf_SPECIAL_NEEDS', 'Pet_Friendly' : 'rsipf_Pet_Friendly'})

#%%
# Filtering to keep only the nessary columns
rsisn_df = rsisn_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'SHELTER_TYPE', 'Risk_Capacity_Spaces',
                          'Evacuation_Zone', 'General_Pop', 'SPECIAL_NEEDS', 'Pet_Friendly']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
rsisn_Longitude = []
rsisn_Latitude = []

# For each row in a variable,
for row in rsisn_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to rsisn_Longitude
        rsisn_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to rsisn_Latitude
        rsisn_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to rsisn_Longitide
        rsisn_Longitude.append(np.NaN)
        # append a missing value to rsisn_Latitude
        rsisn_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
rsisn_df['rsisn_Longitude'] = rsisn_Longitude
rsisn_df['rsisn_Latitude'] = rsisn_Latitude

# Dropping unnessary columns
rsisn_df = rsisn_df.drop(columns = ['LONGLAT'])

# Renaming coumns
rsisn_df = rsisn_df.rename(columns = {'City': 'rsisn_CITY', 'COUNTY': 'rsisn_COUNTY', 'SHELTER_TYPE' : 'rsisn_SHELTER_TYPE'
                                 , 'FACILITY_TYPE' : 'rsisn_FACILITY_TYPE', 'Risk_Capacity_Spaces' : 'rsisn_Risk_Capacity_Spaces'
                                 , 'Evacuation_Zone' : 'rsisn_Evacuation_Zone', 'General_Pop' : 'rsisn_General_Pop'
                                 , 'SPECIAL_NEEDS' : 'rsisn_SPECIAL_NEEDS', 'Pet_Friendly' : 'rsisn_Pet_Friendly'})

#%%
# Appending coast guard (cg_df), fire stations (fs_df), food banks (fb_df)
# and national guard (ng_df) data frames to pnpl_df
pn_df = [cg_df, fs_df, fb_df, ng_df]
# Dataframe has NaN values due to tables not being the same size, 
# however all the data is present for each dataframe
pnpl_df = pd.concat(pn_df)

# Appending Airports(private) (airpri_df), airports public (airpub_df)
# disaster recovery centers mobile (drcm_df), disaster recovery centers (drc_df), 
# private schools (prisch_df), public schools (pubsch_df), emergency medical facilities (emf_df)
# emergency operational centers (eoc_df), healthcare facilities (hf_df), hospitals (hos_df), 
# relief agencies (ra_df), risk shelter inventory general (rsig_df), risk shelter inventory pet friendly (rsipf_df)
# and and risk shelter inventory special needs (rsisn_df) data frames to ln_df
l_df = [airpri_df, airpub_df, drcm_df, drc_df, prisch_df, pubsch_df, emf_df, eoc_df, hf_df, hos_df
        , ra_df, rsig_df, rsipf_df, rsisn_df]
# Dataframe has NaN values due to tables not being the same size, 
# however all the data is present for each dataframe
ln_df = pd.concat(l_df)

#%%
# Exporting (mobile logistic nodes (mln_df), planned path locations (pnpl_df)
# logistic nodes (ln_df), Highwater data (hw), and Sensor data (sn)) to csv files

hw_df.to_csv('HI_Highwater.csv', encoding = 'utf-8')
sn_df.to_csv("HI_Sensor.csv", encoding = 'utf-8')
mln_df.to_csv("Mobile_Logistic_Nodes.csv", encoding = 'utf-8')
pnpl_df.to_csv("Planned Network Path Locations.csv", encoding = 'utf-8')
ln_df.to_csv("Logistic Nodes.csv", encoding = 'utf-8')
