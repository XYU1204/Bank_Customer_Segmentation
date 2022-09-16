# Bank_Customer_Segmentation
**Clustering with Kmean, Dimensional deduction using Autoencoders, Visualization Using PCA, MDS, and tSNE** <br>

Customer segmentation is the approach of dividing a large and diverse customer base into smaller groups of related customers that are similar in certain ways and relevant to the marketing of a bank’s products and services. By performing segmentation, bank can offer more tailored products and services to customers.

* Using Unsupervised learning (K-Mean Clustering) to perform Customer Segmentation on bank customer data set
* Visualizing High Dimensional Data in 2D using Principle Component Analysis (PCA), Multidimensional scaling (MDS), and T-distributed Stochastic Neighbor Embedding (tSNE).
* Applying Dimensionality Reduction by training an Autoencoder neural network.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, scipy, sklearn, matplotlib, seaborn, plotly

## Data Preparation
In this project, we performed customer segmentation on a bank customer data set. The data is obtained from [kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). The original data has 23 columns from 10127 customers, we deleted the meaningless columns. The columns (variables) we work with includes:<br>
#Attrition_Flag: Internal event (customer activity) variable - if the account is closed then 1 else 0 <br>
#Customer_Age: Demographic variable - Customer's Age in Years <br>
#Gender: Demographic variable - M=Male, F=Female <br>
#Dependent_count: Demographic variable - Number of dependents <br>
#Education_Level: Demographic variable - Educational Qualification of the account holder <br>
#Marital_Status: Demographic variable - Married, Single, Divorced, Unknown <br>
#Income_Category: Demographic variable - Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, > $120k) <br>
#Card_Category: Product Variable - Type of Card (Blue, Silver, Gold, Platinum) <br>
#Months_on_book: Period of relationship with bank <br>
#Total_Relationship_Count: Total no. of products held by the customer <br>
#Months_Inactive_12_mon: No. of months inactive in the last 12 months <br>
#Contacts_Count_12_mon: No. of Contacts in the last 12 months <br>
#Credit_Limit: Credit Limit on the Credit Card <br>
#Total_Revolving_Bal: Total Revolving Balance on the Credit Card <br>
#Avg_Open_To_Buy: Open to Buy Credit Line (Average of last 12 months) <br>
#Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1) <br>
#Total_Trans_Amt: Total Transaction Amount (Last 12 months) <br>
#Total_Trans_Ct: Total Transaction Count (Last 12 months) <br>
#Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1) <br>
#Avg_Utilization_Ratio: Average Card Utilization Ratio <br>

## Exploratory Data Analysis

![[alt text]](https://github.com/XYU1204/Bank_Customer_Segmentation/blob/main/age_gender.png) <br>
customer age and gender distribution (where 0 is male and 1 is female)

![image](https://user-images.githubusercontent.com/56236129/190647083-8483f25b-c719-49b2-b198-b8ddf20fc38a.png) <br>
histogram of all the numerical columns

![image](https://user-images.githubusercontent.com/56236129/190651803-4cc3b609-9c3d-46bf-b4ee-8270863d5944.png) <br>
Credit Limit distribution divided by education level and income categories.

![image](https://user-images.githubusercontent.com/56236129/190651227-7c920d0d-412e-4b30-8804-a4116f348995.png) <br>
correlation matrix of all the numerical columns. Total Revolving Balance and Utilization ratio has the highest correlation at 0.64. Average Open to Buy and Utilization ratio has the lowest correlation at -0.54

we codified the categorical variables as following: <br>
Attrition_Flag ['Existing Customer' 'Attrited Customer'] --> [0 1] <br>
------------------------- <br>
Gender ['M' 'F'] --> [0 1] <br>
------------------------- <br>
Education_Level ['High School' 'Graduate' 'Uneducated' 'Unknown' 'College' 'Post-Graduate'  'Doctorate'] --> [2 4 1 0 3 5 6] <br>
------------------------- <br>
Marital_Status ['Married' 'Single' 'Unknown' 'Divorced'] --> [0 2 1 3] <br>
------------------------- <br>
Income_Category ['$60K - $80K' 'Less than $40K' '$80K - $120K' '$40K - $60K' '$120K +' 'Unknown'] --> [3 1 4 2 5 0] <br>
------------------------- <br>
Card_Category ['Blue' 'Gold' 'Silver' 'Platinum'] --> [0 2 1 3] <br>
------------------------- <br>

![image](https://user-images.githubusercontent.com/56236129/190653149-788ee786-514f-4c38-b40d-79c9b2c8e911.png) <br>
Correlation matrix of all columns, after codify the categorical variables. Gender and income category has the strongest negative correlation. 

## Model Buidling

**K-Mean Clustering On the Original Dataset**
![image](https://user-images.githubusercontent.com/56236129/190655479-07e26184-0f0b-4255-9628-d90741718b41.png) <br>
Finding the appropriate number of clusters using elbow method. We decide to use 7 clusters.

![image](https://user-images.githubusercontent.com/56236129/190656472-4b68596c-1ddb-4fa9-8705-b2c63e549cfe.png) <br>
7 cluster center identified. <br>
#First Customer Cluster: older age, low to median income, married, use credit card frequently <br>
#Second Customer Cluster: Churners <br>
#Third Customer Cluster: Higher income, Men <br>
#Fourth Customer Cluster: making frequent transaction <br>

**Visualizing High Dimensional Data in 2D using PCA, MDS, and tSNE**
![image](https://user-images.githubusercontent.com/56236129/190657081-73bf7355-5904-4055-bda8-95d42b59ff81.png) <br>
PCA <br>

![image](https://user-images.githubusercontent.com/56236129/190657284-2d360272-0e9f-4002-85bf-f5e306c1eef6.png) <br>
MDS <br>

![image](https://user-images.githubusercontent.com/56236129/190657342-2d7c1e29-7100-416b-8f6d-f89cc2446567.png) <br>
tSNE <br>


**K-Mean Clustering after applying autoencoder**
![image](https://user-images.githubusercontent.com/56236129/190657987-2247d14b-96d7-4e20-bb34-16de0d68f52c.png) <br>
we trained a deep neural network to build an autoencoder for the input data. The autoencoder reduced the dimensional of original data set from 17 to 10.

![image](https://user-images.githubusercontent.com/56236129/190658232-5367c402-8ba9-4366-afec-f6a66fbebb17.png) <br>
clustering score for original data set in red and reduced dimension in green. Lower score means the data within cluster are closer together. Therefore, applying autoencoder improved the performance of K-Mean Clustering. We decide to use 4 clusters after applying elbow method.

![image](https://user-images.githubusercontent.com/56236129/190658913-9d9e814b-cc00-4084-8bac-7bd1a03a5c95.png) <br>
visualizing 4 clusters in 2D using PCA

![image](https://user-images.githubusercontent.com/56236129/190659167-185e2ab9-49bb-4965-966a-366883922c38.png) <br>
location of 4 cluster centers <br>
#First Customer Cluster: women, median income
#Second Customer Cluster: Churners
#Third Customer Cluster: Higher income, Men
#Fourth Customer Cluster: higher income, high credit limits, low utilization ratio 







