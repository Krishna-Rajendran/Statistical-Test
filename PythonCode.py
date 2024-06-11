import pandas as pd
import numpy as np

data = pd.read_csv("laughter-corpus.csv")
#print(data)

#-------------------------------------------------------------------------------------------------------
print("Q1.  Is the number of laughter events higher for women than for men?")

# Splitting the data for female and male dataframe
total_events = len(data)
female_df = data[data["Gender"] == "Female"]
male_df = data[data["Gender"] == "Male"]
print(total_events, female_df, male_df)

# Given details
total_speakers = 120
female_count = 63
male_count = 57

# Finding observed and expected laughter counts for female and male
observed_female_count = len(female_df)
observed_male_count = len(male_df)
observed_data = [observed_male_count, observed_female_count]
print("O: ", observed_data)

expected_female_count = round((female_count/total_speakers)*total_events)
expected_male_count = round((male_count/total_speakers)*total_events)
expected_data = [expected_male_count, expected_female_count]
print("E: ", expected_data)

# Chi-square test
chi_square_value = 0
for i in range (0,2):
    chi_square_value += pow((observed_data[i] - expected_data[i]),2)/expected_data[i]
print("Chi-square value: ", chi_square_value)

# Finding degree of freedom and critical value
confidence_level = 0.95
significant_level = 1-confidence_level
degree_of_freedom = data["Gender"].nunique() - 1
print("Confidence level: ", confidence_level*100,"%")
print("Degree of freedom: ", degree_of_freedom)

#-------------------------------------------------------------------------------------------------------
print("\nQ2.Is the number of laughter events higher for callers than for receivers?") 

# Splitting the data for caller and receiver dataframe
caller_df = data[data["Role"] == "Caller"]
receiver_df = data[data["Role"] == "Receiver"]

# Given details
caller_count = 60
receiver_count = 60

# Finding observed and expected counts for caller and receiver
observed_data = [len(caller_df), len(receiver_df)]
print("O: ", observed_data)

expected_caller_count = round((caller_count/total_speakers)*total_events)
expected_receiver_count = round((receiver_count/total_speakers)*total_events)
expected_data = [expected_caller_count, expected_receiver_count]
print("E: ", expected_data)

# Chi-square test
chi_square_value = 0
for i in range (0,2):
    chi_square_value += pow((observed_data[i] - expected_data[i]),2)/expected_data[i]
print("Chi-square value: ", chi_square_value)

# Finding degree of freedom and critical value
confidence_level = 0.95
significant_level = 1-confidence_level
degree_of_freedom = data["Gender"].nunique() - 1
print("Confidence level: ", confidence_level*100,"%")
print("Degree of freedom: ", degree_of_freedom)

#-------------------------------------------------------------------------------------------------------
print("\nQ3. Are laughter events longer for women?")

# z test
mean_population = np.sum(data["Duration"])/total_events
mean_sample_women = np.sum(female_df["Duration"])/observed_female_count
std_population = np.sqrt(np.sum(pow(data["Duration"] - mean_population,2))/(total_events-1))
std_sample_women = np.sqrt(np.sum(pow(female_df["Duration"] - mean_sample_women,2))/(observed_female_count-1))
z1 = (mean_sample_women-mean_population)/(std_population/np.sqrt(observed_female_count))
print("Mean for population and sample: ", mean_population, mean_sample_women)
print("Standard deviation of population and sample: ", std_population, std_sample_women)
print("z test value: ",z1)

#-------------------------------------------------------------------------------------------------------
print("\nQ4. Are laughter events longer for callers?")
mean_sample_caller = np.sum(caller_df["Duration"])/observed_data[0]
std_sample_caller = np.sqrt(np.sum(pow(caller_df["Duration"] - mean_sample_caller,2))/(observed_data[0]-1))
z2 = (mean_sample_caller-mean_population)/(std_population/np.sqrt(observed_data[0]))
print("Mean for population and sample: ", mean_population, mean_sample_caller)
print("Standard deviation of population and sample: ", std_population, std_sample_caller)
print("z test value: ",z2)
