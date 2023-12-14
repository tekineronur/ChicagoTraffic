#!/usr/bin/env python
# coding: utf-8

# Onur Tekiner
# Data visualization 
# Project
# 

# Data Cleaning

# In[2]:


import pandas as pd


# In[3]:


raw_data = pd.read_csv('Traffic_Crashes_-_Crashes.csv')


# In[4]:


raw_data.head()


# In[5]:


raw_data.shape


# In[6]:


raw_data.isnull().sum()


# In[7]:


raw_data[raw_data['NUM_UNITS'].isnull()]


# In[8]:


indices_to_remove=raw_data[raw_data['NUM_UNITS'].isnull()].index


# In[9]:


raw_data.drop(indices_to_remove, inplace=True)


# In[10]:


raw_data.isnull().sum()


# In[11]:


columns_to_drop = ['CRASH_DATE_EST_I',
                   'LANE_CNT',
                   'INTERSECTION_RELATED_I',
                   'NOT_RIGHT_OF_WAY_I',
                   'PHOTOS_TAKEN_I',
                   'STATEMENTS_TAKEN_I',
                   'DOORING_I',
                   'WORK_ZONE_I',
                   'WORK_ZONE_TYPE',
                   'WORKERS_PRESENT_I',
                  'CRASH_RECORD_ID',
                   'LOCATION',
                   'RD_NO']


# In[12]:


raw_data.drop(columns=columns_to_drop, inplace=True)


# In[13]:


raw_data.head()


# In[14]:


raw_data.isnull().sum()


# In[15]:


raw_data['REPORT_TYPE'].value_counts()


# In[16]:


raw_data[raw_data['INJURIES_TOTAL'].isnull()]


# In[17]:


indices_to_remove2=raw_data[raw_data['INJURIES_TOTAL'].isnull()].index


# In[18]:


raw_data.drop(indices_to_remove2, inplace=True)


# In[19]:


raw_data.isnull().sum()


# In[20]:


raw_data.drop(columns='REPORT_TYPE', inplace=True)


# In[21]:


raw_data.isnull().sum()


# In[22]:


raw_data['STREET_DIRECTION'].value_counts()


# In[23]:


raw_data['MOST_SEVERE_INJURY'].value_counts()


# In[24]:


raw_data['BEAT_OF_OCCURRENCE'].value_counts()


# In[25]:


#'BEAT_OF_OCCURRENCE' column represent territory code for the police office. I will remove that column too
raw_data.drop(columns='BEAT_OF_OCCURRENCE', inplace=True)


# In[26]:


raw_data.isnull().sum()


# In[27]:


#filling null values on 'STREET_DIRECTION' and 'MOST_SEVERE_INJURY' with most frequent values.
mf_street_direction = raw_data['STREET_DIRECTION'].mode()[0]
mf_most_severe_injury = raw_data['MOST_SEVERE_INJURY'].mode()[0]

raw_data['STREET_DIRECTION'].fillna(mf_street_direction, inplace=True)
raw_data['MOST_SEVERE_INJURY'].fillna(mf_most_severe_injury, inplace=True)


# In[28]:


raw_data.isnull().sum()


# I want to see are there any relationship between hit and run values including null values and injuries columns

# In[29]:


raw_data['HIT_AND_RUN_I'].value_counts()


# In[30]:


raw_data.columns


# In[31]:


raw_data[raw_data['HIT_AND_RUN_I'].isnull()].iloc[:,22:29].sum()


# In[32]:


raw_data[raw_data['HIT_AND_RUN_I'].notnull()].iloc[:,22:29].sum()


# In[33]:


#all values on INJURIES_UNKNOWN are 0, I think I remove that column too.


# In[34]:


raw_data.drop(columns='INJURIES_UNKNOWN', inplace=True)


# In[35]:


#Checking total injuries where "Hit and Run" column are only null and not null.
print("Total(when Hit and Run are null):",raw_data[raw_data['HIT_AND_RUN_I'].isnull()].iloc[:,22:28].sum().sum())
raw_data[raw_data['HIT_AND_RUN_I'].isnull()].iloc[:,22:28].sum()


# In[36]:


print("Total(when Hit and Run is 'Yes'):",raw_data[raw_data['HIT_AND_RUN_I']=="Y"].iloc[:,22:28].sum().sum())
raw_data[raw_data['HIT_AND_RUN_I']=="Y"].iloc[:,22:28].sum()


# In[37]:


print("Total(when Hit and Run is 'No'):",raw_data[raw_data['HIT_AND_RUN_I']=="N"].iloc[:,22:28].sum().sum())
raw_data[raw_data['HIT_AND_RUN_I']=="N"].iloc[:,22:28].sum()


# In[38]:


raw_data['HIT_AND_RUN_I'].value_counts(dropna=False)


# In[39]:


#There are too many null values. I couldn't find any signigicant pattern to fill null values with "Y" or "N")
#So I have decided to remove Hit And Column too.


# In[40]:


raw_data.drop(columns='HIT_AND_RUN_I', inplace=True)


# In[41]:


raw_data.isnull().sum()


# In[42]:


# Fill missing latitude and longitude values with mean
raw_data['LATITUDE'].fillna(raw_data['LATITUDE'].mean(), inplace=True)
raw_data['LONGITUDE'].fillna(raw_data['LONGITUDE'].mean(), inplace=True)


# In[43]:


raw_data.isnull().sum().sum()


# In[44]:


print(raw_data.dtypes)


# In[45]:


raw_data['POSTED_SPEED_LIMIT'].value_counts()


# In[46]:


#This column is more messy than I though. I will create bins and group with each other.
bins = [0, 15, 30, 45, float('inf')]
labels = ['0-15', '15-30', '30-45', 'Above 45']


# In[47]:


import pandas as pd


# In[48]:


raw_data['SPEED_LIMIT_CATEGORY'] = pd.cut(raw_data['POSTED_SPEED_LIMIT'], bins=bins, labels=labels, include_lowest=True)


# In[49]:


raw_data['SPEED_LIMIT_CATEGORY'].value_counts()


# In[50]:


raw_data.drop(columns='POSTED_SPEED_LIMIT', inplace=True)


# In[51]:


#we also doesn't need to STREET_NO column
raw_data.drop(columns='STREET_NO', inplace=True)


# In[52]:


raw_data['NUM_UNITS'].value_counts()


# In[53]:


import numpy as np


# In[54]:


# I will group this column too
bins = [-np.inf, 1, 2, 3, 4, np.inf]  # Adjust bins as needed
labels = ['1', '2', '3', '4', '5+']

raw_data['NUM_UNITS_CATEGORY'] = pd.cut(raw_data['NUM_UNITS'], bins=bins, labels=labels, include_lowest=True)
raw_data.drop(columns='NUM_UNITS', inplace=True)


# In[55]:


raw_data['NUM_UNITS_CATEGORY'].value_counts()


# In[56]:


#since they are representing categorical values, I will change data types as well
raw_data['CRASH_HOUR'] = raw_data['CRASH_HOUR'].astype('category')
raw_data['CRASH_DAY_OF_WEEK'] = raw_data['CRASH_DAY_OF_WEEK'].astype('category')
raw_data['CRASH_MONTH'] = raw_data['CRASH_MONTH'].astype('category')


# In[57]:


print(raw_data.dtypes)


# Handling Categorical Variables

# In[58]:


#checking which columns are categorical in the rest of dataset
columns=raw_data.iloc[:,1:18].columns


# In[59]:


for i in columns:
    print(5*'-',i,5*'-')
    print(raw_data[i].value_counts())
    print(30*'-','\n')


# In[60]:


#I will remove street column and change date columns to datetime data types.
datetime_columns=['CRASH_DATE', 'DATE_POLICE_NOTIFIED']
raw_data['CRASH_DATE'] = pd.to_datetime(raw_data['CRASH_DATE'])
raw_data['DATE_POLICE_NOTIFIED'] = pd.to_datetime(raw_data['DATE_POLICE_NOTIFIED'])
raw_data.drop(columns='STREET_NAME', inplace=True)


# In[61]:


print(raw_data.dtypes)


# In[62]:


data=raw_data


# EDA

# In[63]:


data['YEAR'] = data['CRASH_DATE'].dt.year


# In[64]:


import matplotlib.pyplot as plt


# In[65]:


data['YEAR'].value_counts()


# In[66]:


# Lets check data distribution by year
year=data['YEAR'].value_counts().index
year_values=data['YEAR'].value_counts()


# In[67]:


plt.figure(figsize=(10, 6)) 
plt.bar(year, year_values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Crash Counts by Year')
plt.xticks(year,rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()

plt.show()


# In[68]:


# I think there is missing in the data before 2016, so I will remove before 2017
data = data[data['YEAR'] >= 2017]


# In[69]:


data['YEAR'].value_counts()


# In[70]:


#Let check fatal accidents data
data[data['MOST_SEVERE_INJURY']=='FATAL'].describe()


# In[71]:


data[data['MOST_SEVERE_INJURY']!='FATAL'].describe()


# In[72]:


# Lets check day distribution by year where accidents are fatal
fatal_hour=data[data['MOST_SEVERE_INJURY']=='FATAL']['CRASH_HOUR'].value_counts()
fatal_hour


# In[73]:


fatal_hour=fatal_hour.sort_index()
fatal_hour


# In[74]:


average=fatal_hour.mean()


# In[75]:


#Fatal Accidents Distribution by Hour
plt.figure(figsize=(10, 6)) 
bars=plt.bar(fatal_hour.index,fatal_hour, color='blue', label='Higher than average')

#If it is higher than average fatal accidents become red bar otherwise is blue.
for i in range(len(fatal_hour)):
    if fatal_hour[i] > average:
        bars[i].set_color('red')
    else: 
        bars[i].set_color('blue') 

#For the line for average fatal accidents
plt.axhline(y=average, color='gray', linestyle='--', label=f'Average: {average:.2f}')
plt.legend()

plt.xlabel('Hour')
plt.ylabel('Count')
plt.title('Crash Counts by Hour(Military Clock)')
plt.xticks(range(24))
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# In[76]:


#Since we don't know how to distrubute of crash hours data, lets check to data distribution by hour
hour=data['CRASH_HOUR'].value_counts()


# In[77]:


hour=hour.sort_index()


# In[78]:


hour


# In[79]:


plt.figure(figsize=(10, 6)) 
plt.bar(hour.index, hour, color='purple')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.title('Crash Counts by Hour')
plt.xticks(range(24),rotation=45)
plt.tight_layout()

plt.show()


# In[80]:


#Lets check to ratio for fatal rates out of all accidents
fatal_ratio=fatal_hour/hour+fatal_hour


# In[81]:


#Lets check fatal accidents ratio(Fatal Accidents/Total Accidents) distribution by hour
plt.figure(figsize=(16, 6)) 
plt.bar(fatal_ratio.index, fatal_ratio, color='green')
plt.xlabel('Hour')
plt.ylabel('Rate')
plt.title('Fatal Crash Rate by Hour')
plt.xticks(range(24),rotation=45)
plt.tight_layout()

plt.show()


# In[82]:


#Let's check data distrubution by day
day_dist=data['CRASH_DAY_OF_WEEK'].value_counts()
day_dist=day_dist.sort_index()


# In[83]:


days=['','Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Friday', 'Sat']


# In[84]:


tick_positions = list(range(len(days)))
plt.figure(figsize=(10, 6)) 
plt.bar(day_dist.index, day_dist, color='orange')
plt.xlabel('Day')
plt.ylabel('Count')
plt.title('Crash Counts by Day of Week')
plt.xticks(tick_positions, days, rotation=45) 

plt.tight_layout()

plt.show()


# In[85]:


fatal_day=data[data['MOST_SEVERE_INJURY']=='FATAL']['CRASH_DAY_OF_WEEK'].value_counts()
fatal_day=fatal_day.sort_index()


# In[86]:


tick_positions = list(range(len(days)))
plt.figure(figsize=(10, 6)) 
plt.bar(fatal_day.index, fatal_day, color='m')
plt.xlabel('Day')
plt.ylabel('Count')
plt.title('Fatal Crash Counts by Day of Week')
plt.xticks(tick_positions, days, rotation=45) 

plt.tight_layout()

plt.show()


# In[87]:


fatal_day_rate=fatal_day/day_dist+fatal_day


# In[88]:


tick_positions = list(range(len(days)))
plt.figure(figsize=(10, 6)) 
plt.bar(fatal_day_rate.index, fatal_day_rate, color='0.8')
plt.xlabel('Day')
plt.ylabel('Count')
plt.title('Fatal Crash Rate by Day of Week')
plt.xticks(tick_positions, days, rotation=45) 

plt.tight_layout()

plt.show()


# In[89]:


categorical_columns = [
    'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
    'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE', 'ALIGNMENT', 'ROADWAY_SURFACE_COND',
    'ROAD_DEFECT', 'CRASH_TYPE', 'DAMAGE', 'PRIM_CONTRIBUTORY_CAUSE',
    'SEC_CONTRIBUTORY_CAUSE', 'STREET_DIRECTION', 'MOST_SEVERE_INJURY', 'SPEED_LIMIT_CATEGORY',
    'NUM_UNITS_CATEGORY','CRASH_HOUR','CRASH_DAY_OF_WEEK','CRASH_MONTH']


# In[90]:


#Let's check reasons beyond these fatal crashes
for i in categorical_columns: 
    values=data[data['MOST_SEVERE_INJURY']=='FATAL'][i].value_counts()
    print('\n',values,'\n')


# In[91]:


#Since there are some misleading only looking fatal crashes reasons,
#I want to check distribution of reasons together fatal and non fatal crashes
def double_bar(column):
    for i in range(2):
        if i==0: 
            values = data[data['MOST_SEVERE_INJURY'] != 'FATAL'][column].value_counts()  
            plt.figure(figsize=(8, 6))   
            values.plot(kind='bar')  
            plt.title(f'Bar Chart for {column} with NO FATAL injuries')  
            plt.xlabel(i)  
            plt.ylabel('Count')  
            plt.show()
        if i!=0: 
            values = data[data['MOST_SEVERE_INJURY'] == 'FATAL'][column].value_counts()  
            plt.figure(figsize=(8, 6))   
            values.plot(kind='bar',color='red')  
            plt.title(f'Bar Chart for {column} with FATAL injuries')  
            plt.xlabel(i)  
            plt.ylabel('Count')  
            plt.show()
    


# In[92]:


double_bar('WEATHER_CONDITION')


# In[93]:


double_bar('LIGHTING_CONDITION')


# In[94]:


double_bar('FIRST_CRASH_TYPE')


# In[95]:


double_bar('TRAFFICWAY_TYPE')


# In[96]:


double_bar('ROADWAY_SURFACE_COND')


# In[97]:


double_bar( 'CRASH_TYPE')


# In[98]:


double_bar('DAMAGE')


# In[99]:


double_bar('PRIM_CONTRIBUTORY_CAUSE')


# In[100]:


double_bar('SEC_CONTRIBUTORY_CAUSE')


# In[101]:


double_bar('SPEED_LIMIT_CATEGORY')


# In[102]:


#Mapping


# In[103]:


#I I will create map for non fatal and fatal accidents
import altair as alt

# Separate fatal and non-fatal accidents
fatal_accidents = data[data['MOST_SEVERE_INJURY'] == 'FATAL']
# Because Altair doesn't accept more than 5000 data in once, I sampled 5000 rows from original data
non_fatal_accidents = data[data['MOST_SEVERE_INJURY'] != 'FATAL'].sample(n=5000, random_state=42)

fatal_layer = alt.Chart(fatal_accidents).mark_circle(size=5, color='red').encode(
    latitude='LATITUDE:Q',
    longitude='LONGITUDE:Q',
    tooltip=['LATITUDE', 'LONGITUDE']
).properties(
    width=600,
    height=400
).project(
    type='mercator'
)

non_fatal_layer = alt.Chart(non_fatal_accidents).mark_circle(size=5, color='blue').encode(
    latitude='LATITUDE:Q',
    longitude='LONGITUDE:Q',
    tooltip=['LATITUDE', 'LONGITUDE']
).properties(
    width=600,
    height=400
).project(
    type='mercator'
)

# Overlay the layers to create a combined map
combined_map = (non_fatal_layer + fatal_layer)

combined_map.interactive()


# Correlation Test

# In[104]:


data.dtypes


# In[105]:


encoded_data = pd.get_dummies(data, columns=categorical_columns)


# In[106]:


print('Total columns:',encoded_data.shape[1])
encoded_data.head()


# In[107]:


print(encoded_data.dtypes)


# In[108]:


encoded_data['MOST_SEVERE_INJURY_FATAL']


# In[109]:


correlation_with_target = encoded_data.corr()['MOST_SEVERE_INJURY_FATAL'].abs().sort_values(ascending=False)
correlation_with_target


# In[113]:


correlation_with_target.iloc[1:].head(10)


# In[110]:


correlation_with_target.head(10)


# In[114]:


top_features = correlation_with_target.iloc[1:].head(10).index.tolist()
top_features


# In[112]:


selected_data = encoded_data[top_features]
selected_data

