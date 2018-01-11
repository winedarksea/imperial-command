# Colin Catlin
## Spark SQL and MLlib
### Practice examples using LAPD data and Titanic Survivorship data

```python
sc
```




    <pyspark.context.SparkContext at 0x7f1f22fbd2d0>


```python
from pyspark.sql.functions import *
```

```python
from pyspark.sql import Row
data = sc.textFile("file:/home/cloudera/sfpd.csv")

```


```python
sfpdRDD = data.map(lambda r:Row(
  incident_number=int(r.split(",")[0]), category=(r.split(",")[1]),
  description= (r.split(",")[2]),dayofweek = r.split(",")[3],
  date = (r.split(",")[4]),
  time= (r.split(",")[5]),pddistrict = (r.split(",")[6]),
  resolution = (r.split(",")[7]), address = (r.split(",")[8]),
  x = float(r.split(",")[9]), y = float(r.split(",")[10]),
  pdid = int(r.split(",")[11])
  ))
```


```python
sfpdDF = sqlContext.createDataFrame(sfpdRDD)
```


```python
sfpdDF.show(3)
```

    +--------------------+--------------+------+---------+--------------------+---------------+----------+--------------+-------------+-----+------------+-----------+
    |             address|      category|  date|dayofweek|         description|incident_number|pddistrict|          pdid|   resolution| time|           x|          y|
    +--------------------+--------------+------+---------+--------------------+---------------+----------+--------------+-------------+-----+------------+-----------+
    |JACKSON_ST/POWELL_ST|OTHER_OFFENSES|7/9/15| Thursday|POSSESSION_OF_BUR...|      150599321|   CENTRAL|15059900000000|ARREST/BOOKED|23:45|-122.4099006|37.79561712|
    |300_Block_of_POWE...| LARCENY/THEFT|7/9/15| Thursday|PETTY_THEFT_OF_PR...|      156168837|   CENTRAL|15616900000000|         NONE|23:45|-122.4083843|37.78782711|
    |JACKSON_ST/POWELL_ST|OTHER_OFFENSES|7/9/15| Thursday|DRIVERS_LICENSE/S...|      150599321|   CENTRAL|15059900000000|ARREST/BOOKED|23:45|-122.4099006|37.79561712|
    +--------------------+--------------+------+---------+--------------------+---------------+----------+--------------+-------------+-----+------------+-----------+
    only showing top 3 rows



4\. Display schema


```python
sfpdDF.schema
```




    StructType(List(StructField(address,StringType,true),StructField(category,StringType,true),StructField(date,StringType,true),StructField(dayofweek,StringType,true),StructField(description,StringType,true),StructField(incident_number,LongType,true),StructField(pddistrict,StringType,true),StructField(pdid,LongType,true),StructField(resolution,StringType,true),StructField(time,StringType,true),StructField(x,DoubleType,true),StructField(y,DoubleType,true)))



5\. Take a 10% random sample from sfpdDF using sample function (without replacement, using random seed 1234), and save the resulting dataframe as sampleDF.


```python
pandasDF = sfpdDF.toPandas()
sampleDF = pandasDF.sample(frac = 0.1,random_state= 1234)

```


```python
sampleDF.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>category</th>
      <th>date</th>
      <th>dayofweek</th>
      <th>description</th>
      <th>incident_number</th>
      <th>pddistrict</th>
      <th>pdid</th>
      <th>resolution</th>
      <th>time</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>158054</th>
      <td>19TH_AV/ULLOA_ST</td>
      <td>VANDALISM</td>
      <td>6/28/14</td>
      <td>Saturday</td>
      <td>MALICIOUS_MISCHIEF/VANDALISM</td>
      <td>140541122</td>
      <td>TARAVAL</td>
      <td>14054100000000</td>
      <td>NONE</td>
      <td>21:30</td>
      <td>-122.475539</td>
      <td>37.741187</td>
    </tr>
    <tr>
      <th>98588</th>
      <td>500_Block_of_CLEMENT_ST</td>
      <td>LARCENY/THEFT</td>
      <td>11/16/14</td>
      <td>Sunday</td>
      <td>GRAND_THEFT_FROM_A_BUILDING</td>
      <td>141022511</td>
      <td>RICHMOND</td>
      <td>14102300000000</td>
      <td>NONE</td>
      <td>16:00</td>
      <td>-122.464965</td>
      <td>37.782860</td>
    </tr>
    <tr>
      <th>357833</th>
      <td>100_Block_of_6TH_ST</td>
      <td>WARRANTS</td>
      <td>3/4/13</td>
      <td>Monday</td>
      <td>WARRANT_ARREST</td>
      <td>130183295</td>
      <td>SOUTHERN</td>
      <td>13018300000000</td>
      <td>ARREST/BOOKED</td>
      <td>9:52</td>
      <td>-122.407877</td>
      <td>37.780388</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pyspark.sql import *
```

6\. Register sampleDF as temporary table sfpd. In the subsequent steps, please use either sampleDF (if you're asked to use DataFrame API) or the sfpd temp table (if you are asked to use the SQL query approach).


```python
schemaDF = sqlContext.createDataFrame(sampleDF)
schemaDF.registerTempTable("sfpd")
```

7\. Using DataFrame APIs: show the description for incidents categorized as OTHER_OFFENSES, keeping the result distinct, ordering them in natural order, and limiting results to 10 rows. Save the resulting DF as descDF and display its content. Report the results.


```python
descDF = (sampleDF[sampleDF['category'] == 'OTHER_OFFENSES']["description"])
descDF.head(10)
```




    61400     DRIVERS_LICENSE/SUSPENDED_OR_REVOKED
    26136              MISCELLANEOUS_INVESTIGATION
    252674                TRAFFIC_VIOLATION_ARREST
    75543              MISCELLANEOUS_INVESTIGATION
    202322    DRIVERS_LICENSE/SUSPENDED_OR_REVOKED
    136637                        FALSE_FIRE_ALARM
    205242             MISCELLANEOUS_INVESTIGATION
    181675                       FALSE_PERSONATION
    87638                               CONSPIRACY
    3365                         INDECENT_EXPOSURE
    Name: description, dtype: object



8\. Replicate the previous step using a SQL query via sqlContext.sql().


```python
desc = sqlContext.sql("SELECT DISTINCT(description)
        FROM sfpd WHERE category = 'OTHER_OFFENSES'")
desc.head(10)
```




    [Row(description=u'POSSESSION_OF_BURGLARY_TOOLS_W/PRIORS'),
     Row(description=u'PHONE_CALLS/OBSCENE'),
     Row(description=u'MASSAGE_ESTABLISHMENT_PERMIT_VIOLATION'),
     Row(description=u'CONSPIRACY'),
     Row(description=u'PROBATION_VIOLATION/DV_RELATED'),
     Row(description=u'FAILURE_TO_REGISTER_AS_SEX_OFFENDER'),
     Row(description=u'VIOLATION_OF_FIRE_CODE'),
     Row(description=u'TRAFFIC_VIOLATION_ARREST'),
     Row(description=u'DOG/STRAY_OR_VICIOUS'),
     Row(description=u'FALSE_FIRE_ALARM')]



9\. Using DataFrame APIs: list top 5 PdDistricts by incidents count. Report the results.


```python
result = schemaDF.groupBy("pddistrict").agg(count("category")
.alias("cnt")).orderBy("cnt", ascending = False).show(5)
```

    +----------+----+
    |pddistrict| cnt|
    +----------+----+
    |  SOUTHERN|7393|
    |   MISSION|5082|
    |  NORTHERN|4730|
    |   CENTRAL|4173|
    |   BAYVIEW|3647|
    +----------+----+
    only showing top 5 rows


# Homework 8 - Part C. Analyze Titanic Dataset using pyspark.ml - Solution

This is a famous dataset for machine learning. A description of the dataset can be found at [kaggle website](https://www.kaggle.com/c/titanic/data). In the following, we apply the logistic regression model from pyspark.ml package to this dataset. The goal is to predict survival of passages on board titanice, and to use the pipeline and feature tools from pyspark.ml.



### Step 1: read data from local folder
- you should infer schema from the csv file


```python
sc
```




    <pyspark.context.SparkContext at 0x7ff1042e22d0>




```python
data =sqlContext.read.format('csv').options(header='true', inferSchema = 'true').load('file:/home/cloudera/titanic.csv')
```


```python
data.show(2)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+---------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+---------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|A/5 21171|   7.25|     |       S|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0| PC 17599|71.2833|  C85|       C|
    +-----------+--------+------+--------------------+------+----+-----+-----+---------+-------+-----+--------+
    only showing top 2 rows



### Step 2. Data exploration
the goal of this step is to familiarize yourself with the dataset
- detect data problems
- inform the data engineering steps
- inform the feature selection

a. Print and verify the schema


```python
data.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)



b. Print the first 10 rows from the dataset, we will use this to inform our data processing strategies
- **tip**: using Spark DataFrame's `toPandas()` for nicely formatted display of data.


```python
data.show(10)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25|     |       S|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925|     |       S|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1| C123|       S|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05|     |       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    only showing top 5 rows



c.Obtain summary statistics on the dataframe, which again will inform our data processing strategies
- pay attention to whether there are missing data
- whether the field appears to be continous or discrete


**Please print the fields in rows and stats in columns for easy viewing**


```python
import numpy as np
from pyspark.mllib.stat import Statistics
from pyspark.sql.functions import *
```


```python
dataPD = data.toPandas()
```


```python
dataPD.describe().transpose()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>891.0</td>
      <td>446.000000</td>
      <td>257.353842</td>
      <td>1.00</td>
      <td>223.5000</td>
      <td>446.0000</td>
      <td>668.5</td>
      <td>891.0000</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>891.0</td>
      <td>0.383838</td>
      <td>0.486592</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>891.0</td>
      <td>2.308642</td>
      <td>0.836071</td>
      <td>1.00</td>
      <td>2.0000</td>
      <td>3.0000</td>
      <td>3.0</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>714.0</td>
      <td>29.699118</td>
      <td>14.526497</td>
      <td>0.42</td>
      <td>20.1250</td>
      <td>28.0000</td>
      <td>38.0</td>
      <td>80.0000</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>891.0</td>
      <td>0.523008</td>
      <td>1.102743</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0</td>
      <td>8.0000</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>891.0</td>
      <td>0.381594</td>
      <td>0.806057</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>6.0000</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>891.0</td>
      <td>32.204208</td>
      <td>49.693429</td>
      <td>0.00</td>
      <td>7.9104</td>
      <td>14.4542</td>
      <td>31.0</td>
      <td>512.3292</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataRDD = data.rdd
#print(Statistics.corr(dataRDD, method="pearson"))
#can't get it to work, and not actually called for, so moving on
```

d. for each of the string columns (except `name` and `ticket`), print the count of the 10 most frequent values ordered by descending order of frequency.
- we use this to get a quick look at the number of distinct values
- useful for deciding whether the string column can be treated as a categorical variable
- useful for detecting errors, missing values in such columns


```python
data.groupby("Sex").count().show()
```

    +------+-----+
    |   Sex|count|
    +------+-----+
    |female|  314|
    |  male|  577|
    +------+-----+




```python
data.groupby("Cabin").count().orderBy('count', ascending = False).show(10)
```

    +-----------+-----+
    |      Cabin|count|
    +-----------+-----+
    |           |  687|
    |    B96 B98|    4|
    |C23 C25 C27|    4|
    |         G6|    4|
    |         F2|    3|
    |        F33|    3|
    |    C22 C26|    3|
    |          D|    3|
    |       E101|    3|
    |        B49|    2|
    +-----------+-----+
    only showing top 10 rows




```python
data.groupby("Embarked").count().orderBy('count', ascending = False).show(10)
```

    +--------+-----+
    |Embarked|count|
    +--------+-----+
    |       S|  644|
    |       C|  168|
    |       Q|   77|
    |        |    2|
    +--------+-----+



e. Based on the above, which columns you would keep/not keep as features? Why/Why not? Answer below:

     - PassengerId: Keep as ID
     - Pclass: Keep
     - Name:  Keep for interest
     - Sex: Keep
     - Age: REMOVE too many Na's
     - SibSp: Keep
     - Parch: REMOVE (mostly zeroes, and then some kids has 6 parents?)
     - Ticket: REMOVE (info contained in name, id)
     - Fare: Keep
     - Cabin: REMOVE (lots of empty)
     - Embarked: Keep, although it may be correlated with class



### Step 3. Some feature engineering
the goal of this step is to do necessary feature engineering that is not provided as part of pyspark.ml.features. Note that currently pyspark.ml.features provide
- stringindexer: for convert string labels into numerical labels (0, 1,...), ordered by label frequencies
- one-hot-encoder: mapping a column of category indices to a column of binary vectors.
- vector assembler: merges multiple columns into one vector columns needed for most algorithms

here we will
- deal with missing values
- creating new columns
- convert data types

let's first import the SQL functions


```python
from pyspark.sql.functions import *
```

a. Select all feature columns you plan to use plus `Survived`, in the mean time, convert all numerical columns into double type (**hint**: use sql function .cast())
- the reason we convert all numeric types to doubles is because PySpark does not like other numerical types


```python
my_cols = ["Survived", "PassengerId","Name","Pclass",
          "Sex","SibSp","Fare","Embarked"]
```


```python
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
udf1 = udf(lambda items: 1 if items.isnull() else 0)
```


```python
dataDF = data.select(data.Survived.cast("double"),
          data.PassengerId.cast("double"),
            data.Name,data.Pclass.cast("double"),
            data.Sex,data.Age,data.SibSp.cast("double"),
           data.Fare,data.Embarked)
#I think making INTEGERs to DOUBLEs is SILLY, but maybe there's some greater purpose..
```

b. We notice that there are missing values in the Age column. One way is to drop such rows. Alternatively, we may use a mean replacement strategy.

- Create a new binary indicator column **AgeNA**, converting it to a double type column (**hint**: use isnull() function).
- Also replace the null values in the Age column with the mean age (refer to the summary statistics you've done).



```python
#SO it was my plan to DROP age, but whatever, I'll keep it since you ask so nicely...
#data_mr = dataDF.na.fill("AgeNA")
#df.select('user', 'item', 'fav_items', function(col('item'), col('fav_items')).alias('result')).show()
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import when
dataDF2 = dataDF.withColumn("AgeNA",
              when(isnull(dataDF.Age), (1)).otherwise((0)))
dataDF3 = dataDF2.withColumn("Age",
              when(isnull(dataDF2.Age), 30).otherwise(dataDF2.Age))
#data_mr.select(isnull(data_mr.Age)).show(10)
```

c. Verify the revised DataFrame

print the new schema


```python
dataDF3.printSchema()
```

    root
     |-- Survived: double (nullable = true)
     |-- PassengerId: double (nullable = true)
     |-- Name: string (nullable = true)
     |-- Pclass: double (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: double (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Embarked: string (nullable = true)
     |-- AgeNA: integer (nullable = false)



print the first 10 rows to verify


```python
dataDF3.show(10)
```

+--------+-----------+--------------------+------+------+----+-----+-------+--------+-----+
|Survived|PassengerId|                Name|Pclass|   Sex| Age|SibSp|   Fare|Embarked|AgeNA|
+--------+-----------+--------------------+------+------+----+-----+-------+--------+-----+
|     0.0|        1.0|Braund, Mr. Owen ...|   3.0|  male|22.0|  1.0|   7.25|       S|    0|
|     1.0|        2.0|Cumings, Mrs. Joh...|   1.0|female|38.0|  1.0|71.2833|       C|    0|
|     1.0|        3.0|Heikkinen, Miss. ...|   3.0|female|26.0|  0.0|  7.925|       S|    0|
|     1.0|        4.0|Futrelle, Mrs. Ja...|   1.0|female|35.0|  1.0|   53.1|       S|    0|
|     0.0|        5.0|Allen, Mr. Willia...|   3.0|  male|35.0|  0.0|   8.05|       S|    0|
|     0.0|        6.0|    Moran, Mr. James|   3.0|  male|30.0|  0.0| 8.4583|       Q|    1|
|     0.0|        7.0|McCarthy, Mr. Tim...|   1.0|  male|54.0|  0.0|51.8625|       S|    0|
|     0.0|        8.0|Palsson, Master. ...|   3.0|  male| 2.0|  3.0| 21.075|       S|    0|
|     1.0|        9.0|Johnson, Mrs. Osc...|   3.0|female|27.0|  0.0|11.1333|       S|    0|
|     1.0|       10.0|Nasser, Mrs. Nich...|   2.0|female|14.0|  1.0|30.0708|       C|    0|
+--------+-----------+--------------------+------+------+----+-----+-------+--------+-----+
only showing top 10 rows



recalculate the summary statistics to verify that we no longer have missing values among the numerical columns


```python
dataPD3 = dataDF3.toPandas()
dataPD3.describe().transpose()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>891.0</td>
      <td>0.383838</td>
      <td>0.486592</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>PassengerId</th>
      <td>891.0</td>
      <td>446.000000</td>
      <td>257.353842</td>
      <td>1.00</td>
      <td>223.5000</td>
      <td>446.0000</td>
      <td>668.5</td>
      <td>891.0000</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>891.0</td>
      <td>2.308642</td>
      <td>0.836071</td>
      <td>1.00</td>
      <td>2.0000</td>
      <td>3.0000</td>
      <td>3.0</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>891.0</td>
      <td>29.758889</td>
      <td>13.002570</td>
      <td>0.42</td>
      <td>22.0000</td>
      <td>30.0000</td>
      <td>35.0</td>
      <td>80.0000</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>891.0</td>
      <td>0.523008</td>
      <td>1.102743</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0</td>
      <td>8.0000</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>891.0</td>
      <td>32.204208</td>
      <td>49.693429</td>
      <td>0.00</td>
      <td>7.9104</td>
      <td>14.4542</td>
      <td>31.0</td>
      <td>512.3292</td>
    </tr>
    <tr>
      <th>AgeNA</th>
      <td>891.0</td>
      <td>0.198653</td>
      <td>0.399210</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



Let's call this dataframe **my_final_data**


```python
my_final_data = dataDF3
```

### Step 4: Encoding string columns (for use with pipeline)

String columns cannot be directly used with LogisticRegression. Neither is categorical column types. Our strategy is to convert string categorical variables into a series of binary dummies (drop the last one). Fortunately, these steps are already provided as part of pyspark.ml.features. Click on each below to familiar with them if you need further information.
- [StringIndexer](https://spark.apache.org/docs/1.6.0/api/python/pyspark.ml.html#pyspark.ml.feature.StringIndexer): for convert string labels into numerical labels (0, 1,...), ordered by label frequencies
- [OneHotEncoder](https://spark.apache.org/docs/1.6.0/api/python/pyspark.ml.html#pyspark.ml.feature.OneHotEncoder): mapping a column of category indices to a column of binary vectors (the least frequent one will be dropped by default).
- [VectorAssembler](https://spark.apache.org/docs/1.6.0/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler): merges multiple columns into one vector columns needed for most algorithms

a. import necessary functions (note the parenthesis here is for breaking long lines)


```python
from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
                                OneHotEncoder,StringIndexer)
```

b. Create indexers and encoders for categorical string variables:
- call them `[field]_indexer`, `[field]_encoder` respectively (e.g. gender_indexer, gender_encoder)


```python
from pyspark.ml.feature import OneHotEncoder, StringIndexer

sex_indexer = StringIndexer(inputCol="Sex", outputCol="sex_indexer")
sex_model = sex_indexer.fit(my_final_data)
sex_indexed = sex_model.transform(my_final_data)

gender_encoder=OneHotEncoder(inputCol="sex_indexer", outputCol="gender_encoder")
sex_encoded = gender_encoder.transform(sex_indexed)
sex_encoded.show(3)
#I believe that the index is sufficient (0,1) to contain information of the M/F of the gender
```

+--------+-----------+--------------------+------+------+----+-----+-------+--------+-----+-----------+--------------+
|Survived|PassengerId|                Name|Pclass|   Sex| Age|SibSp|   Fare|Embarked|AgeNA|sex_indexer|gender_encoder|
+--------+-----------+--------------------+------+------+----+-----+-------+--------+-----+-----------+--------------+
|     0.0|        1.0|Braund, Mr. Owen ...|   3.0|  male|22.0|  1.0|   7.25|       S|    0|        0.0| (1,[0],[1.0])|
|     1.0|        2.0|Cumings, Mrs. Joh...|   1.0|female|38.0|  1.0|71.2833|       C|    0|        1.0|     (1,[],[])|
|     1.0|        3.0|Heikkinen, Miss. ...|   3.0|female|26.0|  0.0|  7.925|       S|    0|        1.0|     (1,[],[])|
+--------+-----------+--------------------+------+------+----+-----+-------+--------+-----+-----------+--------------+
only showing top 3 rows




```python
#BUT that is stupid and DIFFICULT, let's do this:
import pandas as pd
df_sex = pd.get_dummies(dataPD3['Sex'])
df_new = pd.concat([dataPD3, df_sex], axis=1)

df_embarked = pd.get_dummies(df_new['Embarked'])
df_new = pd.concat([df_new, df_embarked], axis=1)
df_new1 = df_new.drop("", axis=1)
df_new.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>PassengerId</th>
      <th>Name</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>AgeNA</th>
      <th>female</th>
      <th>male</th>
      <th></th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>3.0</td>
      <td>male</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1.0</td>
      <td>female</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>3.0</td>
      <td>female</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1.0</td>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>5.0</td>
      <td>Allen, Mr. William Henry</td>
      <td>3.0</td>
      <td>male</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>6.0</td>
      <td>Moran, Mr. James</td>
      <td>3.0</td>
      <td>male</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>7.0</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>1.0</td>
      <td>male</td>
      <td>54.0</td>
      <td>0.0</td>
      <td>51.8625</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>8.0</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>3.0</td>
      <td>male</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>21.0750</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>9.0</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>3.0</td>
      <td>female</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>11.1333</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>10.0</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>2.0</td>
      <td>female</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>30.0708</td>
      <td>C</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



If you want, you can test these on the dataframe, use their .fit, then .transform() function. e.g. `gender_indexer.fit(df).transform(df)`


```python

```

### Step 5: Assemble feature columns into a feature vector (for use with pipeline)
- use VectorAssember to do this, we shall the new column `features`


```python
assembled = df_new1.drop(["PassengerId","Sex","Embarked", "Name"], axis=1)
assembled.head(3)
final_data = sqlContext.createDataFrame(assembled)

```


```python
from pyspark.ml.feature import VectorAssembler
ignore = ['Survived']
assembler = VectorAssembler(
    inputCols=[x for x in final_data.columns if x not in ignore],
    outputCol='features')
assembled = assembler.transform(final_data).select("Survived", "features")
assembled.show(10)
```

    +--------+--------------------+
    |Survived|            features|
    +--------+--------------------+
    |     0.0|[3.0,22.0,1.0,7.2...|
    |     1.0|[1.0,38.0,1.0,71....|
    |     1.0|(10,[0,1,3,5,9],[...|
    |     1.0|[1.0,35.0,1.0,53....|
    |     0.0|(10,[0,1,3,6,9],[...|
    |     0.0|[3.0,30.0,0.0,8.4...|
    |     0.0|(10,[0,1,3,6,9],[...|
    |     0.0|[3.0,2.0,3.0,21.0...|
    |     1.0|(10,[0,1,3,5,9],[...|
    |     1.0|[2.0,14.0,1.0,30....|
    +--------+--------------------+
    only showing top 10 rows



### Step 6: Create the LogisticRegression model (for use with pipeline)
- the documentation of LogisticRegression from pyspark.ml can be [found here](https://spark.apache.org/docs/1.6.0/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression).


```python
# import LogisticRegression from ml package

from pyspark.ml.classification import LogisticRegression
```


```python
log_reg_titanic = LogisticRegression(maxIter=10,
        regParam=0.3, featuresCol="features", labelCol="Survived")
```

### Step 7. Assemble Pipelines
put together the stages of the pipeline in the order you want
- the last step should be `log_reg_titanic`


```python
# import Pipeline from ml package
from pyspark.ml import Pipeline
```


```python
pipeline = Pipeline(stages=[assembler, assembled, log_reg_titanic])
```

### Step 8: prepare training and test datasets
a. Please use a 70-30 random split here for training and testing data sets respectively


```python
train_titanic_data, test_titanic_data = assembled.randomSplit([0.7, 0.3])
```

b. Verify the sizes of each datasets after the split


```python
train_titanic_data.count()
```




    612




```python
test_titanic_data.count()
```




    279



### Step 9: Fit the model and use it on the test dataset

a. Fit the model using the predefined pipeline on the training set


```python
fit_model = log_reg_titanic.fit(train_titanic_data)
```

b. Use the fitted model for prediction on the test set


```python
results = fit_model.transform(test_titanic_data)
```

c. Obtain the logistic regression model
- you can easily obtain the model for each stage of the pipeline using **.stages[index]**


```python
logistic_regression_model = fit_model
```

d. Report the logistic regression coefficients


```python
print("Coefficients: " + str(logistic_regression_model.coefficients))
print("Intercept: " + str(logistic_regression_model.intercept))
```

Coefficients: [-0.304104596739,-0.00626826436218,-0.0677781924546,0.00218739572191,
-0.188337964339,0.68017283275,-0.678331949829,0.181673566654,0.0454134013889,
-0.164890847956]
Intercept: 0.602960370762



```python
final_data.columns
```




    ['Survived',
     'Pclass',
     'Age',
     'SibSp',
     'Fare',
     'AgeNA',
     'female',
     'male',
     'C',
     'Q',
     'S']



e. Interpret the obtained coefficients
- **Hint**: it is easier to interpret the odds ratio (OR), which is computed as exp(raw_coefficient)

Pclass = well, no surprise, being rich (and being therefore closer to the lifeboats on deck as well) meant more likely to survive
Age = not so strong an effect here. Not so surprising as there aren't a lot of children here overall
SibSp = no strong effect here. Any slight effect perhaps related to correlation of larger families with lower classes
Fare = huh, little effect, so that goes against "Class" alone making sense due to wealth
AgeNa = some effect, perhaps poor/lower classes were more likely to have no known age recorded, hence this
Female = a massive boon to be female. Clearly women were let first into the life boats, confirming what's said
...and conversely less good to be male
As for the different embarkation points... It's really hard to say for sure. Cherbourg seems to be higher, perhaps here more first class passengers boarded?

### Step 10. Evaluate model performance

It is useful to see how is the results DataFrame after applying the model.

a. print first 5 rows of the results


```python
results.show(5)
```

+--------+--------------------+--------------------+--------------------+----------+
|Survived|            features|       rawPrediction|         probability|prediction|
+--------+--------------------+--------------------+--------------------+----------+
|     0.0|(10,[0,1,3,5,9],[...|[-0.0287926135900...|[0.49280234384245...|       1.0|
|     0.0|(10,[0,1,3,6,7],[...|[0.03902072495165...|[0.50975394364263...|       0.0|
|     0.0|(10,[0,1,3,6,7],[...|[0.32515030873467...|[0.58057890649894...|       0.0|
|     0.0|(10,[0,1,3,6,7],[...|[0.33569087284911...|[0.58314340617042...|       0.0|
|     0.0|(10,[0,1,3,6,7],[...|[0.38789682431156...|[0.59577630020777...|       0.0|
+--------+--------------------+--------------------+--------------------+----------+
only showing top 15 rows



b. Obtain the RUC for the model


```python
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(results)
print("Test Error = %g " % (1.0 - accuracy))
```

    Test Error = 0.166667


c. Does including **AgeNA** increase or decrease model performance in terms of AUC?

- AgeNA does actually decrease the performance when it is removed from the model and the model is run again. That said, it doesn't make a very big different going from 16.7% to 18.9% error. So yes, it makes a difference. No, that is not a profound difference.
