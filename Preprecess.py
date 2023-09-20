import numpy as np
from collections import Counter
import pandas as pd
import os, sys
import torch
from torch_geometric.data import HeteroData

###### Adjust the inputs ######

# input path of input datasets and path to save HeteroData 
PATH = './20191031_v3.1/'
PATH_save = './'

# input version name.
VERSION_NAME = 'data_lotus_20230828_USA'

# input csv file names to use.
selected_filenames = ['degrees', 
                    'funding_rounds', 
                    'investments', 
                    'investors', 
                    'jobs',
                    'organizations', 
                    'people'] 

# input lists of column names to use.
selected_columns = {selected_filenames[0]: ['person_uuid', 'degree_type', 'subject'],
                    selected_filenames[1]: ['funding_round_uuid', 'company_uuid', 'country_code', 'investor_uuids', 'investment_type', 'announced_on', 'raised_amount_usd', 'investor_count'],
                    selected_filenames[2]: ['funding_round_uuid', 'investor_uuid'],
                    selected_filenames[3]: ['uuid', 'investor_type', 'investment_count','total_funding_usd'],
                    selected_filenames[4]: ['person_uuid', 'org_uuid', 'is_current', 'job_type'],
                    selected_filenames[5]: ['uuid', 'company_name', 'country_code', 'founded_on', 'closed_on', 'employee_count', 'funding_rounds', 'funding_total_usd', 'category_group_list', 'status'],
                    selected_filenames[6]: ['uuid', 'gender', 'primary_organization_uuid'],}

# input the conditions for target organization.
target_orgs_founded_on = ['2009-10-31', '2017-10-31'] # 2 yrs ~ 10 yrs is recommended. 
target_orgs_country_code = ['USA'] # can input several country codes. 

# input the conditions for sucessful organization.
# if any company whose fund history has at leat one element of the list below, and that is not 'closed' in status,
# the company is labeled as 'success' 
success_investment_type_list = ['series_a',
                                'series_b', 
                                'series_c', 
                                'series_d', 
                                'series_e', 
                                'series_f',
                                'series_g',
                                'series_h',
                                'series_i']

# 'category_group_list' mapping dictionary
broad_categories = {
                        'Technology': ['Artificial Intelligence', 'Consumer Electronics', 'Data and Analytics', 
                       'Design', 'Hardware', 'Information Technology', 'Internet Services', 
                       'Mobile', 'Platforms', 'Privacy and Security', 'Science and Engineering', 'Software'],
                        'Health Care': ['Health Care'],
                        'Commerce and Shopping': ['Commerce and Shopping'],
                        'Media and Entertainment': ['Content and Publishing', 'Media and Entertainment', 'Video'],
                        'Real Estate': ['Real Estate'],
                        'Energy': ['Energy'],
                        'Food and Beverage': ['Consumer Goods', 'Food and Beverage'],
                        'Transportation': ['Transportation'],
                        'Manufacturing': ['Manufacturing']
                            }

# degree mapping dictionary
degree_mapping = {
        'MBA': [
            'MBA', 'M.B.A.', 'Master of Business Administration', 'Masters in Business Administration', 
            "Executive MBA", "Master of Business Administration (MBA)", 
        ],
        'BS': [
            'BS', 'B.S.', 'Bachelor of Science', 'BSc', 'B.Sc', 'B.Sc.', 'B.S', 
            "Bachelor's degree in science", "BSc."
        ],
        'BA': ['BA', 'B.A.', 'B.A', 'Bachelor of Arts', "Bachelor of Arts (B.A.)", "A.B." ],
        "Bachelor_gen" : [
            'Bachelor', 'Degree', "Bachelor's Degree", 'Bachelors', "Bachelor's", 
            "Bachelor's degree", "Bachelor Degree", ],
        'Master_gen': [
            'MS', 'M.S.', 'Master', 'Graduate', 'Masters', "Master's degree", 
            "Master's", "Master's Degree", "Master Degree"],
        'MA': ['MA', 'M.A.', 'Master of Arts', "M.A", ],
        'MSc': ['MSc', 'M.Sc.', "Msc",'Master of Science'],
        'PhD': ['PhD', 'Ph.D.', 'Doctor of Philosophy', 'Ph.D', "Phd", ],
        'JD': ['JD', 'J.D.', 'Juris Doctor', "J.D"],
        'BBA': ['BBA', 'B.B.A.', 'Bachelor of Business Administration'],
        'LLB': ['LLB', 'L.L.B.', 'Bachelor of Laws'],
        'LLM': ['LLM', 'L.L.M.', 'Master of Laws'],
        'MD': ['MD', 'M.D.', 'Doctor of Medicine'],
        'BEd': ['BEd', 'B.Ed.', 'Bachelor of Education'],
        'MEd': ['MEd', 'M.Ed.', 'Master of Education'],
        'BTech': ['BTech', 'B.Tech.', 'Bachelor of Technology'],
        'MTech': ['MTech', 'M.Tech.', 'Master of Technology'],
        'BCom': ['BCom', 'B.Com', 'B.Com.', 'Bachelor of Commerce'],
        'BFA' : ['BFA']
        # Additional common degrees can be added here
    }
    
# subject mapping dictionary
subject_mapping = {
        'Computer and Engineering': [
            'Computer Science', 'Electrical Engineering', 'Mechanical Engineering', 
            'Computer Engineering', 'Engineering', 'Chemical Engineering', 
            'Information Technology', 'Software Engineering', 'Information Systems', 
            'Industrial Engineering', 'Civil Engineering'
        ],
        'Business and Finance': [
            'Economics', 'Finance', 'Business Administration', 
            'Marketing', 'Accounting', 'Business', 'Management', 
            'International Business', 'Business Management', 'Entrepreneurship', 
            'Business Administration and Management', 'General Management', 'MBA'
        ],
        'Sciences': [
            'Physics', 'Psychology', 'Mathematics', 'Chemistry'
        ],
        'Law': [
            'Law'
        ],
        'Social Sciences': [
            'Political Science'
        ],
        'Humanities': [
            'History', 'English', 'Philosophy'
        ],
        'Life Sciences': ['Biology', 'Biochemistry', 'Medicine'],
        'Communications': [
            'Journalism', 'Communications'
        ],
        'unknown': ['unknown']
    }

###### End of adjustable inputs ######

def _create_df_list(_PATH):
    df_dict = {}
    for filename in selected_filenames:
        temp = pd.read_csv(f'{_PATH}{filename}.csv', usecols=selected_columns[filename])
        df_dict[filename] = temp[selected_columns[filename]]
    
    return df_dict


def _create_target_org_list(df_dict):
    target_orgs = list(df_dict['organizations']['uuid'].loc[
                (df_dict['organizations']['founded_on'] > target_orgs_founded_on[0]) & 
                (target_orgs_founded_on[1] > df_dict['organizations']['founded_on']) &
                (df_dict['organizations']['country_code'].isin(target_orgs_country_code))])
    
    return target_orgs


def _create_invests_edges(df_dict, target_uuids):
    funding_rounds_csv = df_dict['funding_rounds']
    investments_csv = df_dict['investments']

    funding_rounds_csv = funding_rounds_csv[funding_rounds_csv['company_uuid'].isin(target_uuids)]

    # 'investor_uuids' in 'funding_rounds' has many NaNs.
    # we need to replace the 'investor_uuid"s"' in 'funding_rounds' with 'investor_uuid' in 'investments'.
    invests_edge = pd.merge(funding_rounds_csv, investments_csv, how='left', left_on='funding_round_uuid', right_on='funding_round_uuid')
    invests_edge = invests_edge.drop(['investor_uuids'], axis=1)

    # There maybe some NaNs in 'investor_uuid' in 'investments'.
    invests_edge = invests_edge.dropna(subset=['investor_uuid'])

    return invests_edge


def _create_works_edges(df_dict, target_uuids):
    people_csv = df_dict['people']
    # jobs_csv = df_dict['jobs']

    people_csv = people_csv[people_csv['primary_organization_uuid'].isin(target_uuids)]

    # works_edge = pd.merge(people_csv, jobs_csv, how='left', left_on='uuid', right_on='person_uuid')
    works_edge = people_csv[['uuid', 'primary_organization_uuid']]

    return works_edge


def _create_people_node(df_dict, works_edge):
    people_csv = df_dict['people']
    jobs_csv = df_dict['jobs']
    degrees_csv = df_dict['degrees']

    def map_degree(degree):
        if pd.isnull(degree):
            return 'Unknown'
        for standard_degree, variations in degree_mapping.items():
            if degree in variations:
                return standard_degree
        return 'Other'
    
    def map_subject(subject):
        if pd.isnull(subject):
            return 'Unknown'
        for standard_subject, subjects in subject_mapping.items():
            if subject in standard_subject:
                return standard_subject
        return 'Other'

    # we only need investors that have 'works' edges.
    target_org_uuids = list(works_edge['primary_organization_uuid'].unique())
    people_csv = people_csv[people_csv['primary_organization_uuid'].isin(target_org_uuids)]

    people_node = pd.merge(people_csv, jobs_csv, how='left', left_on='uuid', right_on='person_uuid')
    people_node = pd.merge(people_node, degrees_csv, how='left', left_on='uuid', right_on='person_uuid')

    people_node['degree_type_mapped'] = people_node['degree_type'].apply(map_degree)
    people_node['subject_mapped'] = people_node['subject'].apply(map_subject)

    people_node = pd.get_dummies(people_node, columns=['gender', 'is_current', 'job_type', 'degree_type_mapped', 'subject_mapped'], dtype=int)
    people_node = people_node.drop(['primary_organization_uuid', 'person_uuid_x', 'person_uuid_y', 'degree_type', 'subject'], axis=1)
    
    return people_node


def _create_investors_node(df_dict, invests_edge): 
    investor_csv = df_dict['investors']

    # we only need investors that have 'invests' edges.
    target_investor_uuids = list(invests_edge['investor_uuid'].unique())
    investor_csv = investor_csv[investor_csv['uuid'].isin(target_investor_uuids)]
    
    investors_node = pd.get_dummies(investor_csv, columns=['investor_type'], dtype=int)

    return investors_node


def _create_organizations_node(df_dict, invests_edge, works_edge):
    org_from_invests_edge = list(invests_edge['company_uuid'].unique())
    org_from_works_edge = list(works_edge['primary_organization_uuid'].unique())
    target_org_list = list(set(org_from_invests_edge) | set(org_from_works_edge)) # orgs with edges only
    org_node = df_dict['organizations'][df_dict['organizations']['uuid'].isin(target_org_list)]

    # group funding records by company uuid, and make a successful orgs list.
    success_orgs_list = []
    temp = invests_edge[['company_uuid', 'investment_type', 'announced_on']]
    temp = temp.sort_values(by=['announced_on'])
    sid = list(temp['company_uuid'].unique())
    for id in sid:
        fund_history = temp.groupby(['company_uuid']).get_group(id)['investment_type']
        _match = any(item in success_investment_type_list for item in fund_history)
        if _match : 
            success_orgs_list.append(id)
    
    # now label the success / not success orgs. 
    
    org_node['y'] = org_node.apply(lambda row: 1 if row['uuid'] in success_orgs_list
                                and row['status'] != 'closed'
                                else 0,
                                axis =1)
    
    def map_to_single_broad_category(sectors):
        if pd.isnull(sectors):
            return 'Unknown'
    
        # Counter to keep track of the number of votes for each broad category
        votes = Counter()
        sectors = sectors.split(',')
        for sector in sectors:
            sector = sector.strip()  # Remove any extra whitespace
            for broad_category, sub_categories in broad_categories.items():
                if sector in sub_categories:
                    votes[broad_category] += 1
        
        # If no mapping found, put it under 'Other'
        if not votes:
            return 'Other'
        
        # Find the category with the most votes (hard voting)
        most_common_category, _ = votes.most_common(1)[0]
        
        return most_common_category
    
    def extract_min_numeric_value(range_str):
        if '-' in range_str:
            return int(range_str.split('-')[0])
        elif range_str.endswith('+'):
            return int(range_str[:-1])
        else:
            return None
            
    org_node['category'] = org_node['category_group_list'].apply(map_to_single_broad_category)
    print(org_node['category'].value_counts())

    org_node['employee_counts'] = org_node['employee_count'].apply(extract_min_numeric_value)

    org_node = org_node.drop(['company_name', 'country_code', 'founded_on', 'closed_on', 'employee_count', 'status', 'category_group_list'], axis=1)
    org_node = pd.get_dummies(org_node, columns=['category'], dtype=int)

    return org_node



print('Collecting the input datasets...')

df_dict = _create_df_list(PATH)

target_uuids = _create_target_org_list(df_dict)

invests_edge = _create_invests_edges(df_dict, target_uuids)
works_edge = _create_works_edges(df_dict, target_uuids)

people_node = _create_people_node(df_dict, works_edge)
investors_node = _create_investors_node(df_dict, invests_edge)
org_node = _create_organizations_node(df_dict, invests_edge, works_edge)

print('All types of nodes and edges have been formed!')
print('Creating HeteroData...')

org_y = torch.tensor(org_node['y'].values, dtype=torch.int64)
org_x = torch.tensor(org_node.drop(['uuid'], axis=1).values, dtype=torch.float)
investor_x = torch.tensor(investors_node.drop(['uuid'], axis=1).values, dtype=torch.float)
ppl_x = torch.tensor(people_node.drop(['uuid', 'org_uuid'], axis=1).values, dtype=torch.float)

org_mapping = {uuid: i for i, uuid in enumerate(org_node['uuid'].unique())}
inv_mapping = {uuid: i for i, uuid in enumerate(investors_node['uuid'].unique())}
ppl_mapping = {uuid: i for i, uuid in enumerate(people_node['uuid'].unique())}

# temporary solution for src below.
# funding_rounds.csv has several funding rounds whose invester_uuid is not in investors.csv. 
invests_edge = invests_edge[invests_edge['investor_uuid'].isin(list(inv_mapping.keys()))]

src = [inv_mapping[uuid] for uuid in invests_edge['investor_uuid'] if uuid in inv_mapping] 
dst = [org_mapping[uuid] for uuid in invests_edge['company_uuid'] if uuid in org_mapping]
invests_edge_index = torch.tensor([src, dst], dtype=torch.int64)

src = [ppl_mapping[uuid] for uuid in works_edge['uuid'] if uuid in ppl_mapping]
dst = [org_mapping[uuid] for uuid in works_edge['primary_organization_uuid'] if uuid in org_mapping]
works_edge_index = torch.tensor([src, dst], dtype=torch.int64)

data = HeteroData()
data['org'].x = org_x
data['org'].y = org_y 
data['investor'].x = investor_x
data['people'].x = ppl_x
data['investor', 'invests', 'org'].edge_index = invests_edge_index
data['people', 'works', 'org'].edge_index = works_edge_index

print('HeteroData has been made!')
print(data)

torch.save(data, f'{PATH_save}{VERSION_NAME}.pt')
print(f"Hetero data {PATH_save}{VERSION_NAME}.pt has been saved!")