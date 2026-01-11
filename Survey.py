#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import datetime
import random


# ### The following are parameters that you can change: file name, numerical values corresponding to agree/disagree etc

# In[ ]:


# here: definition of the file from where to take the data

xlsx_filename='Q4_new.xlsx'
sheet_name_quarter='Corporate '
row_where_table_starts=1


# In[ ]:


# these are parameters for the output file
name_of_file_to_save_results='tech_automate.xlsx'


# In[ ]:


# here we do some mapping of the agree, strongly agree, disagree etc with numbers, to handle them later (the mapping is arbitrary, as long as there is one it's ok)

# DON'T WORRY ABOUT THE CASE, "Strongly agree" or "Strongly Agree" or "STROngly agrEE"... are all fine
strongly_agree_string="Strongly agree"
agree_String="Agree"
not_used_String="Not used it"
disagree_string="Disagree"
strongly_disagree_string="Strongly disagree"

strongly_agree_value=4
agree_value=3
disagree_value=2
strongly_disagree_value=1
not_used_value=0


# In[ ]:


# put here all the mappings that you want to add
# you can add also spelling mistakes: for example in Q1 2022 there is 'Disgree' without the 'a'

mapping_dictionary={
    strongly_agree_string.lower():strongly_agree_value,
    agree_String.lower():agree_value,
    not_used_String.lower():not_used_value,
    # you can add also spelling mistakes: following is "Disgree" without the 'a'
    "Disgree".lower():disagree_value,
    disagree_string.lower():disagree_value,
    strongly_disagree_string.lower():strongly_disagree_value
}


# In[ ]:


# RETAIL HAS NO TRAINING????
#HS HAS NO TRAINING EITHER????
# you can put here all the columns of which you want to renme the header in the chart sheet
# (MAKE SURE THE CASE AND THE WORDING IS CORRECT AND THAT THERE IS AN 'Overall' COLUMN)
# USE COMMA SEPARATED ROWS, EXCEPT THE LAST ROW WHICH GOES WITHOUT COMMA
# PLEASE ADD HERE ANYTHING THAT RELATES TO THE:
# TRAINING, OVERALL, INFORMED, DEVICE TYPE, LOCATION
rename_columns={
    # informed section: column need be renamed "Informed"
    "Group Workplace Solutions keeps me informed with the latest workplace technology information and developments:":"Informed",
    
    "How likely is it that you would recommend Group Workplace Solutions learning/training to a colleague?2":"Training",
    
    "Overall how well do you feel the tools and support provided by Group Workplace Solutions help you to do your job?":"Overall",
    
    "Please select your region:":"Location",
   
    "What is your current job role?2":"Role",
    
    "What type of device have you used the most in the last 30 days?2":"Device Type"
}

# put here the list of all numerical valued columns (except the agree/disagree columns)
additional_numerical_columns=["Informed","Training","Overall"]


# In[ ]:


# this is the column where comments are
techpulse_comment_columns=[]

def add_comment_column(col_name):
    global techpulse_comment_columns
    
    techpulse_comment_columns.append(col_name)

#add_comment_column('If Group Workplace Solutions could change one thing for you what would it be? (optional)')
add_comment_column('If Group Workplace Solutions (GWS) could change one thing for you what would it be? (Optional)')
# other examples are
# add_comment_column('Comments')


# In[ ]:


# this is a threshold value for correlations: higher than this we set variables are correlated
# it can be betwee 0 and 1, for example .65 means correlation at 65%
correlation_threshold=.7


# In[ ]:


# THIS IS FOR NPS
start_detractors=1
end_detractors=3
start_neutrals=4
end_neutrals=5
start_promoters=6
end_promoters=7


# ### From here on just sit and relax, nothing to be changed :)

# In[ ]:


# define some constants to be used in pivot generation
row_where_heatmap_starts=5
column_where_heatmap_starts=1


# In[ ]:


# here load the data from the excel sheet
xl = pd.ExcelFile(xlsx_filename)
df_techpulse=xl.parse(sheet_name=sheet_name_quarter,header=(row_where_table_starts-1))


# In[ ]:


if('Directorate' in df_techpulse.columns):
    final_columns=['Department', 'Location', 'Org Unit Code', 'Directorate', 'Directorate Code', 'Device Type']
else:
    final_columns=["Location","Role"]
#final_columns=["Which Contact Centre are you based in when not working remotely?", "What is your current job role"]
#final_columns=["Location","Role"]


# In[ ]:


nr_decimals=1


# In[ ]:


for col_nr in np.arange(len(final_columns)):
    if(final_columns[col_nr] in rename_columns):
        final_columns[col_nr]=rename_columns[final_columns[col_nr]]


# In[ ]:


# add all agree/disagree to a set so it's easy to check which column is a agree/disagree column
set_values=set()
for key in mapping_dictionary:
    set_values.add(key)


# In[ ]:


# here get the list of which columns contain the agree/disagree values
# leave as is

agree_disagree_columns=[]
values_to_check=10
# check what are the agree/disagree columns
for col in df_techpulse.columns:
    # check the first 5 values

    if( set(df_techpulse[col].head(values_to_check).apply(lambda x: str(x).lower()).values).issubset(set_values) ):
        agree_disagree_columns.append(col)
        #agree_disagree_columns.add(col)


# In[ ]:


# these are the existing columns
print("The following are the existing columns:")
print(list(df_techpulse.columns))


# In[ ]:


# df_techpulse_chart will contain the transformed columns
# in this box we will create df_techpulse_chart and:
# 1: add 'ID','Email' columns
# 2: add the agree/disagree columns
# 3: add columns to rename (like to shorten "Overall how well do..." into "Overall")
# 4: add the final columns

# rename the following columns
df_techpulse_chart=copy.deepcopy(df_techpulse[['ID','Email']])
df_techpulse_chart[agree_disagree_columns]=df_techpulse[agree_disagree_columns].apply(lambda x: x.astype(str).str.lower())

# add also the columns to rename
for old_col_name in rename_columns:
    if(old_col_name in df_techpulse.columns):
        df_techpulse_chart[rename_columns[old_col_name]]=df_techpulse[old_col_name]

# add the final columns to df_techpulse_chart
df_techpulse_chart[final_columns]=df_techpulse.rename(columns=rename_columns)[final_columns]


# In[ ]:


# now transform the agree/disagree into numbers and put into the df_techpulse_chart
for column in agree_disagree_columns:
    df_techpulse_chart[column]=df_techpulse_chart[column].apply(lambda x: mapping_dictionary[x] if x in mapping_dictionary else x)


# In[ ]:


# here indicate that numerical columns are the agree/disagree plus the 
numerical_columns=agree_disagree_columns+additional_numerical_columns


# In[ ]:


print("And these are the columns which we'll have in the chart sheet:")
print(list(df_techpulse_chart.columns))


# ### Here we do aggregation by the directorate, device type etc

# In[ ]:


# it splits the full location string in spark into its component, based on the delimiter '/'
def get_country_location_local(loc):

    if(type(loc)==type(np.nan)):
        return ['','','']
    
    if(loc==''):
        return ['','','']

    lst=loc.split('/')
    country=lst[0]
    if(len(lst)>1):
        location=lst[1]
    else:
        location=''

    if(len(lst)>2):
        building=lst[2]
    else:
        building=''
    
    return [country,location,building]


# In[ ]:


def get_country_location(loc):
    [country,location,building]=get_country_location_local(loc)
    
    if(location != ''):
        return country+'_'+location
    else:
        return country

def get_country_location_building(loc, filter_list):
    [country,location,building]=get_country_location_local(loc)
    
    if( (building.lower() not in filter_list) and (location.lower() not in filter_list) ):
        return ''
    else:
        str_ret=country
        if(location!=''):
            str_ret=str_ret+'_'+location

        if(building!=''):
            str_ret=str_ret+'_'+building
        
        return str_ret


# In[ ]:


#df_techpulse_chart['Country_location']=''
#df_techpulse_chart['Country_location_building']=''

buildings_breakdown_requested_list_lowercase=[s.lower() for s in buildings_breakdown_requested_list]

if('Location' not in df_techpulse_chart.columns):
    df_techpulse_chart['Location']=''
    

df_techpulse_chart['Country_location']=df_techpulse_chart['Location'].apply(get_country_location)
df_techpulse_chart['Country_location_building']=df_techpulse_chart['Location'].apply(lambda x: get_country_location_building(x, buildings_breakdown_requested_list_lowercase))
df_techpulse_chart=df_techpulse_chart.replace('nan',np.nan)


# In[ ]:


def calculate_pivot(input_dataset, index_column, aggregate_func,columns_order,values_to_pivot, not_used_value=None, not_used_column_list=None):
    df_pivot_dataset=copy.deepcopy(input_dataset)
    
    # we do so that unknown is at the bottom order, if it starts with z
    df_pivot_dataset[index_column]=df_pivot_dataset[index_column].replace(np.nan,'zzzzzUnknown')
    
    if(not_used_value is not None):
        # in order to calculate mean values we substitute the not_used with NULL, so these values do not participate to the mean calculation
        if(not_used_column_list is not None):
            # only replace the not_used in some specific columns
            df_pivot_dataset[not_used_column_list]=df_pivot_dataset[not_used_column_list].replace(not_used_value,np.nan)
        else:
            # replace the not used in all
            df_pivot_dataset=df_pivot_dataset.replace(not_used_value,np.nan)

    # here create the heatmap
    pv_tb=df_pivot_dataset.pivot_table(values=values_to_pivot,index=index_column,aggfunc=aggregate_func,fill_value=np.nan)
        
    # here we round the values
    pv_tb=pv_tb.applymap(lambda x: np.round(x,decimals=2))
    # rename unknown correctly
    pv_tb=pv_tb.rename(index={'zzzzzUnknown':'Unknown'}, errors='ignore')
    # columns order tend to change so we return with the same orser as before
    
    if(pv_tb is not None and len(pv_tb)>1):
        return pv_tb[columns_order]
    else:
        return pv_tb


# In[ ]:


def get_result(d,func):
    # -1 means not possible/not existing or wrong
    result=-1
    
    if((d is not None) and (len(d)>0)):
        tentative_d=d[pd.notnull(d)]
        if(len(tentative_d)>0):
            tentative_result=func(d[pd.notnull(d)])
            if(not pd.isnull(tentative_result)):
                result=np.round(tentative_result,nr_decimals)
        
    return result

def my_min(d):
    return get_result(d,np.min)


# In[ ]:


if('Device Type' in final_columns):
    # Operating system, get mean (only for those who have used it)
    pv_tba=calculate_pivot(df_techpulse_chart, 'Device Type', np.mean,numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)
    # then get nr responses (only those who have used it)
    pv_tbb=calculate_pivot(df_techpulse_chart, 'Device Type', 'count',numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)
else:
    pv_tba=None
    pv_tbb=None

if('Directorate' in final_columns):    
    # directorate, get mean (only for those who have used it)
    pv_tbc=calculate_pivot(df_techpulse_chart, 'Directorate', np.mean,numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)
    # then get nr responses (only those who have used it)
    pv_tbd=calculate_pivot(df_techpulse_chart, 'Directorate', 'count',numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)
else:
    # assume it is 'Role'
    # directorate, get mean (only for those who have used it)
    pv_tbc=calculate_pivot(df_techpulse_chart, 'Role', np.mean,numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)
    # then get nr responses (only those who have used it)
    pv_tbd=calculate_pivot(df_techpulse_chart, 'Role', 'count',numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)
    
# country location, get mean (only for those who have used it)
pv_tbe=calculate_pivot(df_techpulse_chart, 'Country_location', np.mean,numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)
# then get nr responses (only those who have used it)
pv_tbf=calculate_pivot(df_techpulse_chart, 'Country_location', 'count',numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)

# country location building, get mean (only for those who have used it)
df_country_build=df_techpulse_chart[df_techpulse_chart['Country_location_building']!='']
df_country_build=df_country_build[df_country_build['Country_location_building']!=df_country_build['Country_location']]
df_country_build=df_country_build.drop('Country_location', axis=1)
df_country_build=df_country_build.rename(columns={'Country_location_building':'Country_location'})

pv_tbe1=calculate_pivot(df_country_build, 'Country_location', my_min,numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)
# then get nr responses (only those who have used it)
pv_tbf1=calculate_pivot(df_country_build, 'Country_location', 'count',numerical_columns,numerical_columns,not_used_value=not_used_value,not_used_column_list=agree_disagree_columns)

pv_tbe=pd.concat([pv_tbe,pv_tbe1]).sort_index()
pv_tbf=pd.concat([pv_tbf,pv_tbf1]).sort_index()


# ### The step below will save the output file

# In[ ]:


sheet_name=sheet_name_quarter+'_Charts'

with pd.ExcelWriter(name_of_file_to_save_results) as writer:
   
    # firstly put the percentages
    cur_col=column_where_heatmap_starts
    cur_row=row_where_heatmap_starts

    df=pd.DataFrame()
    all_responses=np.array(df_techpulse_chart[agree_disagree_columns].apply(lambda x: len(x),axis=0))
    all_not_used=np.array(df_techpulse_chart[agree_disagree_columns].apply(lambda x: len(x[x==not_used_value]),axis=0))
    all_used=all_responses-all_not_used
    all_agree=np.array(df_techpulse_chart[agree_disagree_columns].apply(lambda x: len(x[x==agree_value]),axis=0))
    all_strongly_agree=np.array(df_techpulse_chart[agree_disagree_columns].apply(lambda x: len(x[x==strongly_agree_value]),axis=0))
    all_disagree=np.array(df_techpulse_chart[agree_disagree_columns].apply(lambda x: len(x[x==disagree_value]),axis=0))
    all_strongly_disagree=np.array(df_techpulse_chart[agree_disagree_columns].apply(lambda x: len(x[x==strongly_disagree_value]),axis=0))
    
    decimals=0
    df['Not used %']=np.round(100*np.divide(all_not_used,all_responses),decimals)
    df['Used %']=np.round(100*np.divide(all_used,all_responses),decimals)
    df['Strongly Agree %']=np.round(100*np.divide(all_strongly_agree,all_used),decimals)
    df['Agree %']=np.round(np.divide(100*all_agree,all_used),decimals)
    df['Strongly Disgree %']=np.round(100*np.divide(all_strongly_disagree,all_used),decimals)
    df['Disagree %']=np.round(np.divide(100*all_disagree,all_used),decimals)

    len_columns=len(df.T)
    df.set_index(df_techpulse_chart[agree_disagree_columns].columns).T.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col)
    
    row_start=row_where_heatmap_starts+len_columns+2
    
    # we put the overall mean score first across all the agree/disagree columns    
    
    df_tmp=df_techpulse_chart[agree_disagree_columns].apply(lambda x: np.round(np.mean(x[x!=not_used_value]),2),axis=0).reset_index()
    cur_col=column_where_heatmap_starts+1
    cur_row=row_start+1
    
    len_columns=len(df_tmp)
    df_tmp.set_index('index').T.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col,index=False)
    
    # now put the final three heatmap columns (learning informed overall)
    df_tmp=df_techpulse_chart[list(set(numerical_columns)-set(agree_disagree_columns))].apply(lambda x: np.round(np.mean(x),2),axis=0).reset_index()
    cur_col=cur_col+len_columns
    cur_row=row_start+1
    df_tmp.set_index('index').T.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col,index=False)
    
    # here write nr of responses for the agree/disagree columns
    cur_col=column_where_heatmap_starts+1
    cur_row=row_start+3
    df_tmp=df_techpulse_chart[agree_disagree_columns].apply(lambda x: np.count_nonzero(np.where(x!=not_used_value)),axis=0).reset_index()
    df_tmp.set_index(0).T[1:].to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col,index=False)
    
    # now put the final three heatmap columns (learning informed overall)
    df_tmp=df_techpulse_chart[list(set(numerical_columns)-set(agree_disagree_columns))].apply(lambda x: np.count_nonzero(~np.isnan(x)),axis=0).reset_index()
    cur_col=cur_col+len_columns
    cur_row=row_start+3
    df_tmp.set_index(0).T[1:].to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col,index=False)
    
    worksheet = writer.sheets[sheet_name]
    # firstly write dataset info
    worksheet.write(0, 0, "Data for file: "+xlsx_filename)
    worksheet.write(1, 0, "Sheet: "+sheet_name_quarter)
    
    # then we start the dance
    worksheet.write(row_start+2, column_where_heatmap_starts, "Overall score");cur_row=cur_row+1;
    worksheet.write(row_start+3, column_where_heatmap_starts, "Number of responses")
    
    
    if( (pv_tba is not None) and (pv_tbb is not None) ):
        # here add device type breakdown
        cur_row=cur_row+1
        cur_col=column_where_heatmap_starts

        pv_tba.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col)
        cur_row=cur_row+len(pv_tba)+1

        # here add nr of responses for device type breakdown
        #worksheet.write(cur_row, cur_col, 'Device Type: nr of responses');cur_row=cur_row+1;
        worksheet.write(cur_row, cur_col, 'nr of responses:');cur_row=cur_row+1;

        pv_tbb.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col)

        cur_row=cur_row+len(pv_tbb)+1
    

    
    # here add Directorate breakdown
    cur_row=cur_row+1
    cur_col=column_where_heatmap_starts
    
    pv_tbc.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col)
    cur_row=cur_row+len(pv_tbc)+1
    
    
    # here add nr responses for directorate
    #worksheet.write(cur_row, cur_col, 'Directorate: nr of responses');cur_row=cur_row+1;
    worksheet.write(cur_row, cur_col, 'nr of responses:');cur_row=cur_row+1;
    cur_col=column_where_heatmap_starts
    
    pv_tbd.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col)
    cur_row=cur_row+len(pv_tbd)+1
    
    
    
    # here add Site breakdown
    cur_row=cur_row+1
    cur_col=column_where_heatmap_starts
    
    pv_tbe.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col)
    cur_row=cur_row+len(pv_tbe)+1
    
    
    # here add nr responses for directorate
    #worksheet.write(cur_row, cur_col, 'Site breakdown: nr of responses');cur_row=cur_row+1;
    worksheet.write(cur_row, cur_col, 'nr of responses:');cur_row=cur_row+1;
    cur_col=column_where_heatmap_starts
    
    pv_tbf.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col)
    cur_row=cur_row+len(pv_tbf)+1

    
    # now add the full table
    cur_row=cur_row+2
    cur_col=0
    df_techpulse_chart.to_excel(writer,sheet_name=sheet_name, startrow=cur_row, startcol=cur_col, index=False)
    


# ## Here produce the wordcloud
# 
# 

# In[ ]:


def produce_wordcloud(df,text_columns):
    text=''
    for txt_col in text_columns:
        text =text+ " ".join(comment for comment in df[txt_col].replace(np.nan,'').astype(str))
    if(text==''):
        text='NO TEXT WAS FIND IN THE COMMENTS: CHECK THAT COLUMN NAMES OF COMMENT COLUMNS IS CORRECT!!!'
    return text


# In[ ]:


with open('all_wordcloud.txt', 'w') as f:
    f.write(produce_wordcloud(df_techpulse, techpulse_comment_columns))

with open('promoters_wordcloud.txt', 'w') as f:
    df_tmp=copy.deepcopy(df_techpulse_chart[df_techpulse_chart['Overall']>=start_promoters][['Email']])
    f.write(produce_wordcloud(df_techpulse.merge(df_tmp,how='inner',on=['Email']), techpulse_comment_columns))

with open('neutrals_wordcloud.txt', 'w') as f:
    df_tmp=copy.deepcopy(df_techpulse_chart[(df_techpulse_chart['Overall']>=start_neutrals) | (df_techpulse_chart['Overall']<=end_neutrals)][['Email']])
    f.write(produce_wordcloud(df_techpulse.merge(df_tmp,how='inner',on=['Email']), techpulse_comment_columns))

with open('detractors_wordcloud.txt', 'w') as f:
    df_tmp=copy.deepcopy(df_techpulse_chart[df_techpulse_chart['Overall']<=end_detractors][['Email']])
    f.write(produce_wordcloud(df_techpulse.merge(df_tmp,how='inner',on=['Email']), techpulse_comment_columns))


# ## Here calculate NPS

# In[ ]:


def calc_NPS(df,nps_column):
    all_ppl=len(df)
    
    promoters=len(df[df[nps_column]>=start_promoters])
    detractors=len(df[df[nps_column]<=end_detractors])
    
    return ((100*promoters)/all_ppl) - ((100*detractors)/all_ppl)


# In[ ]:


def calc_NPS_upg(df,nps_column, local_start_promoters=None, local_end_detractors=None):
    global start_promoters,end_detractors
    
    if(local_start_promoters is None):
        local_start_promoters=start_promoters
    
    if(local_end_detractors is None):
        local_end_detractors=end_detractors
    
    all_ppl=len(df)
    
    promoters=len(df[df[nps_column]>=int(local_start_promoters)])
    if(int(local_start_promoters)!=float(local_start_promoters)):
        fraction=float(local_start_promoters)-int(local_start_promoters)
        tmp=int(len(df[df[nps_column]==int(local_start_promoters)])*fraction)
        promoters=promoters-tmp

    detractors=len(df[df[nps_column]<=int(local_end_detractors)])
    if(int(local_end_detractors)!=float(local_end_detractors)):
        fraction=float(local_end_detractors)-int(local_end_detractors)
        tmp=int(len(df[df[nps_column]==int(local_end_detractors)])*fraction)
        detractors=detractors+tmp
    
    return ((100*promoters)/all_ppl) - ((100*detractors)/all_ppl)


# In[ ]:


print('THIS IS THE NPS:')
calc_NPS(df_techpulse_chart,'Overall')


# ## From here on we do the correlation analysis

# In[ ]:


df_techpulse_chart_normalised=copy.deepcopy(df_techpulse_chart)


# In[ ]:


import sklearn as sklearn
from sklearn.preprocessing import MinMaxScaler
mx = MinMaxScaler()


df_techpulse_chart_normalised[numerical_columns]=mx.fit_transform(df_techpulse_chart[numerical_columns])


# In[ ]:


# this is the correlation matrix
corr_mat_norm_data=df_techpulse_chart_normalised[numerical_columns].corr()
#corr_mat_norm_data


# In[ ]:


# here plot the heatmap
plt.figure(figsize=(20,20))
#upp_mat = np.triu(corr_mat_norm_data)

sns.heatmap(corr_mat_norm_data, annot=True, cmap=plt.cm.CMRmap_r)

plt.show()


# In[ ]:


correlations={}

# make a working copy of the correlation matrix
corr_mat_np=corr_mat_norm_data.to_numpy()
nr_rows=corr_mat_norm_data.shape[0]
columns=corr_mat_norm_data.columns



# here scan rows and columns to get the correlation values above the threshold
for row in np.arange(0,nr_rows):
    for col in np.arange(0,nr_rows):
        if(col==row):
            continue
        
        if((corr_mat_np[row,col]>correlation_threshold) or (corr_mat_np[row,col]<(-correlation_threshold))):
            row_name=columns[row]
            col_name=columns[col]
            if(row_name in correlations):
                correlations[columns[row]].append(col_name)
            else:
                correlations[row_name]=[col_name]


# In[ ]:


# this is a test, contact Leo if the result is not 'Test ok!!!'
if((corr_mat_np<(-correlation_threshold)).any()):
    print('WARNING: THERE ARE NEGATIVE CORRELATIONS: PLEASE ASK FOR FURTHER ANALYSIS')
else:
    print('Test ok!!!')


# In[ ]:


# here we eliminate duplicates correlation sets

set_lists=[]

for con in correlations:
    lst_el=set(list(correlations[con]+[con]))
    set_lists.append(lst_el)

eliminate=set()

check_ok=False
while(not check_ok):
    check_ok=True
    
    for pos in np.arange(len(set_lists)):
        for pos1 in np.arange(len(set_lists)):

            if(pos1==pos):
                continue

            if((set_lists[pos]==set_lists[pos1]) or (set_lists[pos].issubset(set_lists[pos1]))):
                set_lists.pop(pos)
                check_ok=False
                break
        
        if(check_ok==False):
            break


# In[ ]:


# these are correlations: i.e. things that tend to vary accordingly
# people tend to express the same choices
print('Correlations:')
print('\n')
for cr in set_lists:
    print(cr)
    print('\n')


# ## Here you can do analysis of correlated categories

# In[ ]:


# in order to have like colors we generate a dictionary indexed by values
cl_cnt=0
color_dictionary={}
for cl in mcolors.TABLEAU_COLORS:
    color_dictionary[cl_cnt]=mcolors.TABLEAU_COLORS[cl]
    cl_cnt=cl_cnt+1
    if(cl_cnt>7):
        break


# In[ ]:


# here you can do word analysis for the correlated categories


for lst in set_lists:
    cols=list(lst)    
    print('++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++')
    print('Correlation group: ')
    print(cols)
    print('++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++')
    print('')
    
    
    fig, axs = plt.subplots(1, len(lst), figsize=(12, 4))
    
    for i in np.arange(len(cols)):
        #print(df_techpulse_chart[cols[i]].value_counts().sort_index())
        dfc=df_techpulse_chart[cols[i]].dropna().astype('int64')
        
        # get colors to use
        local_color_dict=copy.deepcopy(color_dictionary)
        for cnt_clr in np.arange(8):
            if(cnt_clr not in dfc.values):
                local_color_dict.pop(cnt_clr)
        
        dfc.value_counts().sort_index().plot(kind='pie', autopct='%1.0f%%', 
                                                        startangle=0,ax=axs[i],colors=list(local_color_dict.values()))
        
    
    plt.show()
    


# In[ ]:


# here you can do word analysis for the correlated categories
set_lengths=[]
for lst in set_lists:
    cols=list(lst)
    
    #fig, axs = plt.subplots(1, len(lst), figsize=(12, 4))
    
    values=set()
    for i in np.arange(len(cols)):
        for val in df_techpulse_chart[cols[i]].value_counts().index:
            values.add(val)
    
    values=sorted(values)

    lengths={}
    # check which values match
    for val in values:
        df=df_techpulse_chart[df_techpulse_chart[cols[0]]==val]
        for i in np.arange(1,len(cols)):
            df=df[df[cols[i]]==val]
        
        lengths[val]=len(df)
    
    set_lengths.append(lengths)
    #plt.show()


# In[ ]:


for i in np.arange(len(set_lists)):
    nr_total_answers=len(df_techpulse_chart[list(set_lists[i])].dropna())

    cols=list(set_lists[i])
    print('++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++')
    print('Correlation group: ')
    print(cols)
    print('++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++')
    print('')
    print("Total valid answers across correlation group (without 'N/A'): ",nr_total_answers)
    print('')
    D=set_lengths[i]
    plt.figure()
    percentages_array=np.round(np.array([float(x) for x in D.values()])*100/nr_total_answers,1)
    plt.title(str(np.round(np.sum(percentages_array),1))+'% of people gave the same answer')
    graph=plt.bar(range(len(D)),percentages_array , align='center')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xticks(range(len(D)), list(D.keys()))
    
    for j in np.arange(len(graph)):
        p=graph[j]
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2,
                 y+height*1.01,
                 str(list(D.values())[j])+'='+str(percentages_array[j])+'%',
                 ha='center',
                 weight='bold')
    
    plt.show()


# ## Here you can see how each column influences the NPS score

# In[ ]:


values_mapping={}
for kk in mapping_dictionary:
    values_mapping[mapping_dictionary[kk]]=kk


# In[ ]:


df_techpulse_chart.columns


# In[ ]:


# for any column, you can see how it influences the nps
column_to_check='Excel'

indx=np.array(df_techpulse_chart[column_to_check].dropna().astype('int64').value_counts().sort_index().index)

print('For column ',column_to_check,':')
tot=len(df_techpulse_chart[(df_techpulse_chart[column_to_check]!='') & (df_techpulse_chart[column_to_check]!= np.nan) ])
print('Total: ', tot)
print('')
for ind in indx:
    print('++++++++++++++++++++++++++++++++++++++')
    print('List for ',ind, ' (',values_mapping[ind],')')
    df1=df_techpulse_chart[df_techpulse_chart[column_to_check]==ind]
    nr_ppl=len(df1)
    print('Nr of people who gave ',ind,':',nr_ppl,'(',np.round(nr_ppl*100/tot,1),'% of the total in column ',column_to_check,')')
    if(nr_ppl>0):
        nps=calc_NPS(df1,'Overall')
        supp=len(df1[(df1['Overall']>=start_promoters) & (df1['Overall']<=end_promoters)])
        neu=len(df1[(df1['Overall']>=start_neutrals) & (df1['Overall']<=end_neutrals)])
        det=len(df1[(df1['Overall']>=start_detractors) & (df1['Overall']<=end_detractors)])

        print('Supporters: ',supp,'(',np.round(supp*100/nr_ppl,1),'% of the total giving ',ind,')')
        print('Neutrals: ',neu,'(',np.round(neu*100/nr_ppl,1),'% of the total giving ',ind,')')
        print('Detractors: ',det,'(',np.round(det*100/nr_ppl,1),'% of the total giving ',ind,')')
        print('NPS for people who gave ',ind,':',nps)
    else:
        print('No nps calculated')

    print('')
    print('++++++++++++++++++++++++++++++++++++++')


# ## Analysis of probability of being supporter/neutral/detractor based on agree/disagree choices

# In[ ]:


supporters_count={}
neutrals_count={}
detractors_count={}

total_sum=0

for col in agree_disagree_columns:
    indx=np.array(df_techpulse_chart[col].dropna().astype('int64').value_counts().sort_index().index)
    # check each value in this column
    for ind in indx:
        df1=df_techpulse_chart[df_techpulse_chart[col]==ind]
        nr_ppl=len(df1)
        total_sum=total_sum+nr_ppl
        if(nr_ppl>0):
            df_supporters=df1[(df1['Overall']>=start_promoters) & (df1['Overall']<=end_promoters)]
            df_neutrals=df1[(df1['Overall']>=start_neutrals) & (df1['Overall']<=end_neutrals)]
            df_detractors=df1[(df1['Overall']>=start_detractors) & (df1['Overall']<=end_detractors)]
            # increment supporters
            if(ind in supporters_count):
                supporters_count[ind]=supporters_count[ind]+len(df_supporters)
            else:
                supporters_count[ind]=len(df_supporters)
            
            # increment neutrals
            if(ind in neutrals_count):
                neutrals_count[ind]=neutrals_count[ind]+len(df_neutrals)
            else:
                neutrals_count[ind]=len(df_neutrals)

            # increment detractors
            if(ind in detractors_count):
                detractors_count[ind]=detractors_count[ind]+len(df_detractors)
            else:
                detractors_count[ind]=len(df_detractors)


# normalize values
len_agr_disagr=len(agree_disagree_columns)
total_sum=total_sum/len_agr_disagr
for val in supporters_count:
    supporters_count[val]=supporters_count[val]/len_agr_disagr
for val in neutrals_count:
    neutrals_count[val]=neutrals_count[val]/len_agr_disagr
for val in detractors_count:
    detractors_count[val]=detractors_count[val]/len_agr_disagr


# In[ ]:


labels=['% prob of being supporter','% prob of being neutral','% prob of being detractor']

for val in supporters_count:
    total=supporters_count[val]+neutrals_count[val]+detractors_count[val]
    percentages=np.round(np.array([supporters_count[val],neutrals_count[val],detractors_count[val]])*100/total,1)
    plt.figure()
    plt.title('Probabilities for people who choose "'+values_mapping[val]+'"')
    plt.pie(percentages, labels=percentages)
    plt.legend(labels, loc='upper right', bbox_to_anchor=(1.7, .7))
    plt.tight_layout()
    plt.show()


# # Here add cluster analysis
# 
# ## step 1: Clustering pain points using K-Means to identify themes
# 

# In[ ]:


from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import re

col_name = 'If Workplace could change one thing for you what would it be? (Optional)'
# Drop rows where comments are empty
documents = df_techpulse[col_name].dropna().astype(str).tolist()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    # Filter out stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

processed_docs = [preprocess_text(doc) for doc in documents]

# Converting text into a matrix of importance scores
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(processed_docs)


# Reduces the "noise" and helps the K-Means algorithm focus on core patterns
pca = PCA(n_components=0.95, random_state=42) # Keep 95% of variance
X_reduced = pca.fit_transform(X.toarray())

# K-Means Clustering
# We define 'k' based on how number of "themes" 
k = 5
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10, random_state=42)
clusters = model.fit_predict(X_reduced)

# Mapping results back to original data
# Create a results dataframe to see which comment belongs to which theme
results_df = pd.DataFrame({
    'Original_Comment': documents,
    'Cluster_ID': clusters
})

# Identify the top terms per cluster
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(k):
    print(f"Cluster {i}: ", end='')
    for ind in order_centroids[i, :5]: # Top 5 words per cluster
        print(f'{terms[ind]} ', end='')
    print('\n')
    
# ## step 2: simplify the variables, shrink all correlated variables into 1 into each correlation group

# In[ ]:


def get_NPS_category(x,local_start_neutrals=None,local_end_neutrals=None,local_start_promoters=None):
    global start_neutrals,end_neutrals,start_promoters
    
    if(local_start_neutrals is None):
        local_start_neutrals=start_neutrals
    
    if(local_end_neutrals is None):
        local_end_neutrals=end_neutrals
    
    if(local_start_promoters is None):
        local_start_promoters=start_promoters
    
    if((x>=local_start_neutrals) and (x<=local_end_neutrals)):
        result='neutrals'
    elif(x>=local_start_promoters):
        result='promoters'
    else:
        result='detractors'
    
    return result


# In[ ]:


# pick randomly one of the variables for each of the correlation groups and assign it
df_techpulse_chart_simplified=copy.deepcopy(df_techpulse_chart)
numerical_columns_simplified=copy.deepcopy(numerical_columns)


# In[ ]:


df_techpulse_chart_simplified.columns


# In[ ]:


group_nr=1
numerical_columns_simplified_local=copy.deepcopy(numerical_columns_simplified)
correlation_group_lists={}
columns_to_drop=set()
for lst in set_lists:
    lst=list(lst)
    if((lst[0] in numerical_columns_simplified) and ('Overall' not in lst)):
        # assume they are all in
        col_nr=random.randint(0,len(lst)-1)
        new_col_name='Correlation_group_'+str(group_nr)
        df_techpulse_chart_simplified[new_col_name]=df_techpulse_chart_simplified[lst[col_nr]]
        #df_techpulse_chart_simplified=df_techpulse_chart_simplified.drop(columns=lst)
        columns_to_drop.add(lst[col_nr])
        numerical_columns_simplified_local=list(set(numerical_columns_simplified_local)-set(lst))
        numerical_columns_simplified_local.append('Correlation_group_'+str(group_nr))
        correlation_group_lists[group_nr]=lst
    group_nr=group_nr+1
df_techpulse_chart_simplified=df_techpulse_chart_simplified.drop(columns=list(columns_to_drop))
numerical_columns_simplified=numerical_columns_simplified_local


# In[ ]:


df_techpulse_chart_simplified_anonymised=copy.deepcopy(df_techpulse_chart_simplified[numerical_columns_simplified])


# In[ ]:


df_techpulse_chart_simplified_anonymised['NPS_category']=df_techpulse_chart_simplified_anonymised['Overall'].apply(lambda x: get_NPS_category(x))


# In[ ]:


numerical_columns_simplified_no_overall=copy.deepcopy(numerical_columns_simplified)


# In[ ]:


remove_cols=['Overall','Training','Informed']
for col in remove_cols[:1]:
    if(col in numerical_columns_simplified_no_overall):
        numerical_columns_simplified_no_overall.remove(col)


# ## Now here analyse personas

# In[ ]:


names=['promoters','neutrals','detractors']
cnt_name=0
nm=names[cnt_name]
df_promoters=df_techpulse_chart_simplified_anonymised[df_techpulse_chart_simplified_anonymised['NPS_category']==nm]
tot_promoters=len(df_promoters)
print('nr ',nm,': ',str(tot_promoters))

cnt_name=1
nm=names[cnt_name]
df_neutrals=df_techpulse_chart_simplified_anonymised[df_techpulse_chart_simplified_anonymised['NPS_category']==nm]
tot_neutrals=len(df_neutrals)
print('nr ',nm,': ',str(tot_neutrals))

cnt_name=2
nm=names[cnt_name]
df_detractors=df_techpulse_chart_simplified_anonymised[df_techpulse_chart_simplified_anonymised['NPS_category']==nm]
tot_detractors=len(df_detractors)
print('nr ',nm,': ',str(tot_detractors))


# In[ ]:


nr_cols=4
nr_rows=int(len(numerical_columns_simplified_no_overall)/nr_cols)
leftover_cols=int(len(numerical_columns_simplified_no_overall)%nr_cols)

if(leftover_cols>0):
    nr_rows=nr_rows+1
print("PERCENTAGES ARE CALCULATED OUT OF THE TOTAL: PROMOTERS/TOTAL POPULATION, NEUTRALS/TOTAL POP. DETRACTORS/TOTAL POP.")

figure, axis = plt.subplots(nr_rows, nr_cols)

col_counter=0
nr_effective_cols=nr_cols
for rw in np.arange(nr_rows):
    if((rw==nr_rows-1) and (leftover_cols>0)):
        nr_effective_cols=leftover_cols
    for cl in np.arange(nr_effective_cols):
        col_nm=numerical_columns_simplified_no_overall[col_counter]
        col_counter=col_counter+1
        # get the values from the full column
        values=set()
        for val in df_techpulse_chart_simplified_anonymised[col_nm].value_counts().index:
            values.add(val)
        values=sorted(values)

        df_col_promoters=df_promoters[col_nm]
        df_col_neutrals=df_neutrals[col_nm]
        df_col_detractors=df_detractors[col_nm]
        
        df_total_col=df_techpulse_chart_simplified_anonymised[col_nm]
        tot_promoters=len(df_total_col)
        tot_neutrals=len(df_total_col)
        tot_detractors=len(df_total_col)

        lengths_promoters={}
        lengths_neutrals={}
        lengths_detractors={}
        # check which values match
        for val in values:
            lengths_promoters[val]=len(df_col_promoters[df_col_promoters==val])
            lengths_neutrals[val]=len(df_col_neutrals[df_col_neutrals==val])
            lengths_detractors[val]=len(df_col_detractors[df_col_detractors==val])
        


        width = 0.2
        delta_x=width
        percentages_array_promoters=np.round(np.array([float(x) for x in lengths_promoters.values()])*100/tot_promoters,1)
        percentages_array_neutrals=np.round(np.array([float(x) for x in lengths_neutrals.values()])*100/tot_neutrals,1)
        percentages_array_detractors=np.round(np.array([float(x) for x in lengths_detractors.values()])*100/tot_detractors,1)
        axis[rw, cl].set_title(col_nm, fontdict={'fontsize':10})
        axis[rw, cl].bar(np.array(values)-delta_x,percentages_array_promoters , align='center', width=width,color=['green']*len(values))
        axis[rw, cl].bar(np.array(values),percentages_array_neutrals , align='center', width=width,color=['#ffbf00']*len(values))
        axis[rw, cl].bar(np.array(values)+delta_x,percentages_array_detractors , align='center', width=width,color=['red']*len(values))
        axis[rw, cl].set_ylim([0, 50])
        if(cl==0):
            axis[rw, cl].yaxis.set_major_formatter(mtick.PercentFormatter())
            if(rw>0):
                if(len(values)<7):
                    axis[rw, cl].set_xticks(values,'')
                else:
                    axis[rw, cl].set_xticks(values)
            else:
                axis[rw, cl].set_xticks(values)
                
            axis[rw, cl].set_yticks([50])
        else:
            #axis[rw, cl].xaxis.set_ticks_position('none') 
            axis[rw, cl].yaxis.set_ticks_position('none')
            if(len(values)<7):
                axis[rw, cl].set_xticks(values,'')
            else:
                axis[rw, cl].set_xticks(values)
            axis[rw, cl].set_yticks([50,100],'')
        #axis[rw, cl].gca().xticks(values)        

        
for cl in np.arange(leftover_cols,nr_cols):
    rw=nr_rows-1
    axis[rw, cl].set_visible(False)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1.9,
                    top=1.9,
                    wspace=0.05,
                    hspace=0.6)
#fig.tight_layout()    
plt.show()
for grp in correlation_group_lists:
    print('Correlation group ', grp,': ',correlation_group_lists[grp])


# In[ ]:


nr_cols=4
nr_rows=int(len(numerical_columns_simplified_no_overall)/nr_cols)
leftover_cols=int(len(numerical_columns_simplified_no_overall)%nr_cols)

if(leftover_cols>0):
    nr_rows=nr_rows+1
print("PERSONAS FOR PROMOTERS, DETRACTORS AND NEUTRALS")
print("PERCENTAGES ARE CALCULATED OUT OF THE TOTAL PROMOTERS, TOTAL DETRACTORS AND TOTAL NEUTRALS")
print("SO FOR EACH CHOICE AGREE/DISAGREE WE HAVE IN EACH BAR: PROMOTERS/TOTAL_PROMOTERS, NEUTRALS/TOTAL NEUTRALS AND DETRACTORS/TOTAL DETRACTORS")
figure, axis = plt.subplots(nr_rows, nr_cols)

col_counter=0
nr_effective_cols=nr_cols
for rw in np.arange(nr_rows):
    if((rw==nr_rows-1) and (leftover_cols>0)):
        nr_effective_cols=leftover_cols
    for cl in np.arange(nr_effective_cols):
        col_nm=numerical_columns_simplified_no_overall[col_counter]
        col_counter=col_counter+1
        # get the values from the full column
        values=set()
        for val in df_techpulse_chart_simplified_anonymised[col_nm].value_counts().index:
            values.add(val)
        values=sorted(values)

        df_col_promoters=df_promoters[col_nm]
        df_col_neutrals=df_neutrals[col_nm]
        df_col_detractors=df_detractors[col_nm]
        
        tot_promoters=len(df_promoters)
        tot_neutrals=len(df_neutrals)
        tot_detractors=len(df_detractors)


        lengths_promoters={}
        lengths_neutrals={}
        lengths_detractors={}
        # check which values match
        for val in values:
            lengths_promoters[val]=len(df_col_promoters[df_col_promoters==val])
            lengths_neutrals[val]=len(df_col_neutrals[df_col_neutrals==val])
            lengths_detractors[val]=len(df_col_detractors[df_col_detractors==val])


        width = 0.2
        delta_x=width
        percentages_array_promoters=np.round(np.array([float(x) for x in lengths_promoters.values()])*100/tot_promoters,1)
        percentages_array_neutrals=np.round(np.array([float(x) for x in lengths_neutrals.values()])*100/tot_neutrals,1)
        percentages_array_detractors=np.round(np.array([float(x) for x in lengths_detractors.values()])*100/tot_detractors,1)
        axis[rw, cl].set_title(col_nm, fontdict={'fontsize':10})
        axis[rw, cl].bar(np.array(values)-delta_x,percentages_array_promoters , align='center', width=width,color=['green']*len(values))
        axis[rw, cl].bar(np.array(values),percentages_array_neutrals , align='center', width=width,color=['#ffbf00']*len(values))
        axis[rw, cl].bar(np.array(values)+delta_x,percentages_array_detractors , align='center', width=width,color=['red']*len(values))
        axis[rw, cl].set_ylim([0, 100])
        if(cl==0):
            axis[rw, cl].yaxis.set_major_formatter(mtick.PercentFormatter())
            if(rw>0):
                if(len(values)<7):
                    axis[rw, cl].set_xticks(values,'')
                else:
                    axis[rw, cl].set_xticks(values)
            else:
                axis[rw, cl].set_xticks(values)
            axis[rw, cl].set_yticks([50,100])
        else:
            #axis[rw, cl].xaxis.set_ticks_position('none') 
            axis[rw, cl].yaxis.set_ticks_position('none')
            if(len(values)<7):
                axis[rw, cl].set_xticks(values,'')
            else:
                axis[rw, cl].set_xticks(values)
            axis[rw, cl].set_yticks([50,100],'')
        #axis[rw, cl].gca().xticks(values)        

        
for cl in np.arange(leftover_cols,nr_cols):
    rw=nr_rows-1
    axis[rw, cl].set_visible(False)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1.9,
                    top=1.9,
                    wspace=0.05,
                    hspace=0.6)
#fig.tight_layout()    
plt.show()
for grp in correlation_group_lists:
    print('Correlation group ', grp,': ',correlation_group_lists[grp])


# # Number of people per unit score in the 'Overall' column

# In[ ]:


values=set()
df_col=df_techpulse_chart_simplified_anonymised['Overall']
total_ppl=len(df_col)
for val in df_col.value_counts().index:
    values.add(val)
values=sorted(values)
lengths={}
# check which values match
for val in values:
    lengths[val]=len(df_col[df_col==val])


# In[3]:


#sample
values=set()
df_col=df['Overall']
total_ppl=len(df_col)
for val in df_col.value_counts().index:
    values.add(val)
values=sorted(values)
lengths={}
# check which values match
for val in values:
    lengths[val]=len(df_col[df_col==val])


# In[ ]:


color_array=['red','red','red','#ffbf00','#ffbf00','green','green']
color_array_true=[]
for idcol in np.arange(1,len(color_array)+1):
    if(idcol in lengths):
        color_array_true.append(color_array[idcol-1])

plt.figure()

percentages_array=np.round(np.array([float(x) for x in lengths.values()])*100/total_ppl,1)
people_array=np.array([float(x) for x in lengths.values()])
plt.title("People who voted for each unit of the 'Overall' score")
graph=plt.bar(np.array(list(lengths.keys())),percentages_array , align='center',color=color_array_true)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
#plt.yticks([0,50,100])
plt.xticks(values)

for j in np.arange(len(graph)):
    p=graph[j]
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x+width/2,
             y+height*1.01,
             str(people_array[j])+'='+str(percentages_array[j])+'%',
             ha='center',
             weight='bold')

plt.show()

if(False):
    plt.figure()

    cumulative_perc=np.cumsum(percentages_array)
    people_array=np.array([float(x) for x in lengths.values()])
    plt.title("Cumulative percentage (adding up) of each unit 'Overall' score")
    graph=plt.bar(np.array(list(lengths.keys())),cumulative_perc , align='center',color=color_array_true)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xticks(values)

    for j in np.arange(len(graph)):
        p=graph[j]
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2,
                 y+height*1.01,
                 '+'+str(percentages_array[j])+'%='+str(cumulative_perc[j])+'%',
                 ha='center',
                 weight='bold')

    plt.show()


# In[ ]:


# EXAMPLES
print('NPS as is (with 6 and 7 being supporters)')
print(calc_NPS_upg(df_techpulse_chart,'Overall', local_start_promoters=6, local_end_detractors=3))
print('NPS widening the range of supporters to 5,6,7')
print(calc_NPS_upg(df_techpulse_chart,'Overall', local_start_promoters=5, local_end_detractors=3))
print('NPS if 20% of people who voted 5 join the supporters club')
print(calc_NPS_upg(df_techpulse_chart,'Overall', local_start_promoters=5.8, local_end_detractors=3))


# In[ ]:


print('Last successful run at: ')
print(datetime.datetime.now())

