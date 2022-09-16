# Bank_Customer_Segmentation
**Clustering with Kmean, Dimensional deduction using Autoencoders, Visualization Using PCA, MDS, and tSNE** <br>

Customer segmentation is the approach of dividing a large and diverse customer base into smaller groups of related customers that are similar in certain ways and relevant to the marketing of a bank’s products and services. By performing segmentation, bank can offer more tailored products and services to customers.

* Prepared stock market data from various sources. Performed portfolio analysis using CAPM (capital asset pricing model).
* Portfolio Optimization using 2000 Monte Carlo Simulations.
* Optimize arbitrary initial portfolio weights by maximizing sharpe_ratio using SLSQP (Sequential Least Squares Programming). 
* On average, the optimizer increased expected annual return by 73.99%, and increase expected sharpe ratio by 25.40%, making the investment more profitable and less violatile at the same time.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, scipy, sklearn, matplotlib, seaborn, plotly

## Data Preparation
In this project, we performed customer segmentation on a bank customer data set. The data is obtained from [kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). The original data has 23 columns from 10127 customers, we deleted the meaningless columns. The columns (variables) we work with includes:
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
customer age and gender distribution

![image](https://user-images.githubusercontent.com/56236129/190647083-8483f25b-c719-49b2-b198-b8ddf20fc38a.png) <br>
histogram of all the numerical columns
