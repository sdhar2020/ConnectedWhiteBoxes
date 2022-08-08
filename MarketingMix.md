
# Market Mix Model: Conceptual Toy Example

A marketing mix model is a modeling technique used to determine market attribution, the estimated impact of each marketing channel company uses.

Unlike attribution modeling, another technique used for marketing spend attribution, marketing mix models measure the approximate impact of your marketing channels.
Generally, your output variable will be **sales or conversions**, but it can also be things like **Vend $**. Your input variables typically consist of marketing spend by channel by period (day, week, month, quarter, etc…), but can also include other variables, which we’ll get to later



## The usefulness of marketing mix models
You can tap into the power of a marketing mix model in a number of ways, including:

- To get a better understanding of the relationships between 
your marketing channels and your target metric (i.e. conversions).
- To distinguish high ROI marketing channels from low ones and ultimately better optimize your marketing budget.
- To predict future conversions based on given inputs.
- Each of these insights can offer a ton of value as you scale your business. Let’s dive into what it takes to build one with Python



## Data Read


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df= pd.read_csv('https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv',index_col=0)
```


```python
df.head()
```





  <div id="df-69ee5967-265b-4293-9a4f-3d8b915a53e7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>Radio</th>
      <th>Newspaper</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>230.1</td>
      <td>37.8</td>
      <td>69.2</td>
      <td>22.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>44.5</td>
      <td>39.3</td>
      <td>45.1</td>
      <td>10.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17.2</td>
      <td>45.9</td>
      <td>69.3</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>151.5</td>
      <td>41.3</td>
      <td>58.5</td>
      <td>18.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>180.8</td>
      <td>10.8</td>
      <td>58.4</td>
      <td>12.9</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-69ee5967-265b-4293-9a4f-3d8b915a53e7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-69ee5967-265b-4293-9a4f-3d8b915a53e7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-69ee5967-265b-4293-9a4f-3d8b915a53e7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



## Correlation matrix

A correlation matrix is a table that shows the correlation values for each pair relationship. It’s a very fast and efficient way of understanding feature relationships. Here's the code for our matrix.




```python
plt.subplots(figsize=(15,15))
corr = df.corr()
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, annot = True, cmap = sns.diverging_palette(220, 20, as_cmap=True), )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fad8b49a110>




![png](MarketingMix_files/MarketingMix_7_1.png)


The correlation matrix above shows that there’s a strong correlation between TV and sales (0.78), a moderate correlation between radio and sales (0.58), and a weak correlation between newspaper and sales (0.23). It’s still too early to conclude anything but this is good to keep in mind as we move on with other EDAs.

## Pairplot

A pair plot is a simple way to visualize the relationships between each variable - it’s similar to a correlation matrix except it shows a graph for each pair-relationship instead of a correlation. Now let’s take a look at the code for our pair plot, as well as the result.


```python
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x7fad6c2b7710>




![png](MarketingMix_files/MarketingMix_10_1.png)


We can see some consistency between our pair plot and our original correlation matrix. It looks like there’s a strong positive relationship between TV and sales, less for radio, and even less for newspapers.

## Feature Importance

Feature importance allows you to determine how (you guessed it) “important” each input variable in predicting the output variable.

The code below first creates a random forest model with sales as the target variable and the marketing channels as the feature inputs. Once we create the model, we then calculate the feature importance of each predictor and plot it on a bar chart.


```python
# Setting X and y variables
X = df.loc[:, df.columns != 'Sales']
y = df['Sales']
# Building Random Forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)
pred = model.predict(X_test)# Visualizing Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fad5ec5fa50>




![png](MarketingMix_files/MarketingMix_14_1.png)


## Building the Marketing Mix Model

An OLS model is a type of regression model that's most commonly used when building marketing mix models. What makes Python so amazing is that it already has a library that you can use to create an OLS model:


```python
import statsmodels.formula.api as sm
model = sm.ols(formula="Sales~TV+Radio+Newspaper", data=df).fit()
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  Sales   R-squared:                       0.897
    Model:                            OLS   Adj. R-squared:                  0.896
    Method:                 Least Squares   F-statistic:                     570.3
    Date:                Tue, 25 Jan 2022   Prob (F-statistic):           1.58e-96
    Time:                        17:52:15   Log-Likelihood:                -386.18
    No. Observations:                 200   AIC:                             780.4
    Df Residuals:                     196   BIC:                             793.6
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      2.9389      0.312      9.422      0.000       2.324       3.554
    TV             0.0458      0.001     32.809      0.000       0.043       0.049
    Radio          0.1885      0.009     21.893      0.000       0.172       0.206
    Newspaper     -0.0010      0.006     -0.177      0.860      -0.013       0.011
    ==============================================================================
    Omnibus:                       60.414   Durbin-Watson:                   2.084
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              151.241
    Skew:                          -1.327   Prob(JB):                     1.44e-33
    Kurtosis:                       6.332   Cond. No.                         454.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


## Plot actual vs predicted values
Graph the predicted sales values with the actual sales values to visually see how our model performs. You'd want to do this in a business use case if you’re trying to see how well your model reflects what’s actually happening - in this case, if you’re trying to see how well your model predicts sales based on the amount spent in each marketing channel.


```python
from matplotlib.pyplot import figure

y_pred = model.predict()
labels = df['Sales']
df_temp = pd.DataFrame({'Actual': labels, 'Predicted':y_pred})
df_temp.head()

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
y1 = df_temp['Actual']
y2 = df_temp['Predicted']
plt.plot(y1, label = 'Actual')
plt.plot(y2, label = 'Predicted')
plt.legend()
plt.show()
```


![png](MarketingMix_files/MarketingMix_18_0.png)


Not Bad!!

## Interpret a marketing mix model


```python
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Sales</td>      <th>  R-squared:         </th> <td>   0.897</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.896</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   570.3</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 25 Jan 2022</td> <th>  Prob (F-statistic):</th> <td>1.58e-96</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:56:58</td>     <th>  Log-Likelihood:    </th> <td> -386.18</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   780.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   196</td>      <th>  BIC:               </th> <td>   793.6</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    2.9389</td> <td>    0.312</td> <td>    9.422</td> <td> 0.000</td> <td>    2.324</td> <td>    3.554</td>
</tr>
<tr>
  <th>TV</th>        <td>    0.0458</td> <td>    0.001</td> <td>   32.809</td> <td> 0.000</td> <td>    0.043</td> <td>    0.049</td>
</tr>
<tr>
  <th>Radio</th>     <td>    0.1885</td> <td>    0.009</td> <td>   21.893</td> <td> 0.000</td> <td>    0.172</td> <td>    0.206</td>
</tr>
<tr>
  <th>Newspaper</th> <td>   -0.0010</td> <td>    0.006</td> <td>   -0.177</td> <td> 0.860</td> <td>   -0.013</td> <td>    0.011</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>60.414</td> <th>  Durbin-Watson:     </th> <td>   2.084</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 151.241</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.327</td> <th>  Prob(JB):          </th> <td>1.44e-33</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.332</td> <th>  Cond. No.          </th> <td>    454.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



From  the output from .summary(), we can see a few areas to focus on (you can reference these insights against the OLS regression results below): 

1.   The Adj. R-squared is 0.896. This means approximately 90% of the total variation in the data can be explained by the model
2.   If you look at the column, P>|t|, you can see the p-values for each predictor. The p-values for TV and radio are less than 0.000, but the p-value for newspapers is 0.86, which indicates that newspaper spend has no significant impact on sales. Generally, you want the p-value to be less than 1% or 5%, which are the two standards in practice.
1.   The p-values for TV and radio are less than 0.000, but the p-value for newspapers is 0.86, which indicates that newspaper spend has no significant impact on sales. Generally, you want the p-value to be less than 1% or 5%, which are the two standards in practice.






## Conclusion

This is a toy model that shows the basics of Marketing Mix modeling that helps in building simulation on how much incremental vend dollars can be gotten by spending $1 on one or more of the channels

# Export


```python
!pip install nbconvert
```

    Requirement already satisfied: nbconvert in /usr/local/lib/python3.7/dist-packages (5.6.1)
    Requirement already satisfied: jupyter-core in /usr/local/lib/python3.7/dist-packages (from nbconvert) (4.9.1)
    Requirement already satisfied: jinja2>=2.4 in /usr/local/lib/python3.7/dist-packages (from nbconvert) (2.11.3)
    Requirement already satisfied: testpath in /usr/local/lib/python3.7/dist-packages (from nbconvert) (0.5.0)
    Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.7/dist-packages (from nbconvert) (0.3)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert) (0.8.4)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert) (1.5.0)
    Requirement already satisfied: nbformat>=4.4 in /usr/local/lib/python3.7/dist-packages (from nbconvert) (5.1.3)
    Requirement already satisfied: bleach in /usr/local/lib/python3.7/dist-packages (from nbconvert) (4.1.0)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.7/dist-packages (from nbconvert) (0.7.1)
    Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.7/dist-packages (from nbconvert) (5.1.1)
    Requirement already satisfied: pygments in /usr/local/lib/python3.7/dist-packages (from nbconvert) (2.6.1)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2>=2.4->nbconvert) (2.0.1)
    Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.7/dist-packages (from nbformat>=4.4->nbconvert) (0.2.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /usr/local/lib/python3.7/dist-packages (from nbformat>=4.4->nbconvert) (4.3.3)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (21.4.0)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (3.10.0.2)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (0.18.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (4.10.0)
    Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (5.4.0)
    Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.7/dist-packages (from importlib-resources>=1.4.0->jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (3.7.0)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.7/dist-packages (from bleach->nbconvert) (1.15.0)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.7/dist-packages (from bleach->nbconvert) (0.5.1)
    Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from bleach->nbconvert) (21.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->bleach->nbconvert) (3.0.6)



```python
%ls
```

    [0m[01;34msample_data[0m/



```python
from google.colab import drive
drive.mount('/content/drive')
```


```python
!pwd
```

    /content



```python
%ls
```

    [0m[01;34mdrive[0m/  [01;34msample_data[0m/



```python
from IPython.core.display import Markdown

%%shell

jupyter nbconvert --to html '/content/drive/MyDrive/Colab Notebooks/MarketingMix.ipynb'
```


      File "<ipython-input-25-f68d9a61a6f9>", line 3
        %%shell
        ^
    SyntaxError: invalid syntax


