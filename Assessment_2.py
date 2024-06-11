import pandas as pd
import numpy as np

#----------------------------------------------------------------------------------
# Train Data
data = pd.read_csv('C:/Users/Oottu/OneDrive/Desktop/Krishna/University of Glasgow/04. Comp Social Intelligence/Assessed Exercise/Assessment 2/training-part-2.csv')

# The two classes in the dataset are smile and frown.
# Assuming smile as C1 and frown as C2.
column_class = data['Class']
n = len(data.columns)-1


# Find the probability of smile and frown in the training file
C1 = C2 = 0;
for i in range (0, column_class.count()):
    if column_class[i] == "smile":
        C1 = C1+1
    else:
        C2 = C2+1
p_C1 = C1/column_class.count()
p_C2 = C2/column_class.count()
# p(C1) and p(C2) have equal probability
# The classes are statistically independent


# Splitting the dataset in to two small matrices based on the classes
class1 = data[data["Class"] == "smile"];
class1 = np.array(class1.iloc[:,:n]) 

class2 = data[data["Class"] == "frown"];
class2 = np.array(class2.iloc[:,:n])


# Covariance calculation of Class1
mean_class1 = class1.mean(axis=0)

for i in range (0,len(class1)):
    for j in range (0,n):
        class1[i][j] = class1[i][j] - mean_class1[j]

class1_transpose = class1.transpose()
cov_class1 = np.matmul(class1_transpose,class1)


# Covariance calculation of Class2
mean_class2 = class2.mean(axis=0)

for i in range (0,len(class2)):
    for j in range (0,n):
        class2[i][j] = class2[i][j] - mean_class2[j]

class2_transpose = class2.transpose()
cov_class2 = np.matmul(class2_transpose,class2) 


# Gamma of Class1 and Class2
for i in range (0,len(class1)):
    gamma1 = (-0.5*np.log(np.linalg.det(cov_class1)))-(0.5*np.matmul(class1[i,:].transpose(),np.matmul(class1[i,:],np.linalg.inv(cov_class1))))
    gamma2 = (-0.5*np.log(np.linalg.det(cov_class2)))-(0.5*np.matmul(class1[i,:].transpose(),np.matmul(class1[i,:],np.linalg.inv(cov_class2))))

for i in range (0,len(class2)):
    gamma1 = (-0.5*np.log(np.linalg.det(cov_class1)))-(0.5*np.matmul(class2[i,:].transpose(),np.matmul(class2[i,:],np.linalg.inv(cov_class1))))
    gamma2 = (-0.5*np.log(np.linalg.det(cov_class2)))-(0.5*np.matmul(class2[i,:].transpose(),np.matmul(class2[i,:],np.linalg.inv(cov_class2))))


#---------------------------------------------------------------------
#Test file
test_data = pd.read_csv('C:/Users/Oottu/OneDrive/Desktop/Krishna/University of Glasgow/04. Comp Social Intelligence/Assessed Exercise/Assessment 2/test-part-2.csv')

# Splitting the dataset in to two small dataframe
samples = np.array(test_data.iloc[:,:n])


# Gamma of Class1 and Class2
prediction = []
for i in range (0,len(samples)):
    gamma1 = (-0.5*np.log(np.linalg.det(cov_class1)))-(0.5*np.matmul(samples[i,:].transpose(),np.matmul(samples[i,:],np.linalg.inv(cov_class1))))
    gamma2 = (-0.5*np.log(np.linalg.det(cov_class2)))-(0.5*np.matmul(samples[i,:].transpose(),np.matmul(samples[i,:],np.linalg.inv(cov_class2))))
    
    if gamma1>gamma2:
        prediction.append("smile")
    else:
        prediction.append("frown")

prediction = np.array(prediction)
print("Predictions: ", prediction)

labels = test_data['Class']
wrong_prediction_count = np.sum(labels != prediction)
print("Number of wrong predictions: ", wrong_prediction_count)
print("Error Percentage:",(wrong_prediction_count/len(test_data))*100)
