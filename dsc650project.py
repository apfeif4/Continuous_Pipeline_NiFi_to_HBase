# from pyspark.sql.functions import regex_replace, col
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
import gc
import happybase

spark = SparkSession.builder.appName("MLlib Heart attack Prediction").enableHiveSupport().getOrCreate()

def write_to_hbase(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('hattack_table')
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()
    
# Load the Hive data into a dataframe.
df = spark.sql("SELECT age, gender, rate, sbp, dbp, bs, ckmb, trop, result FROM hattack")


# Drop NA value.
df = df.na.drop()

# Obtain the features.
assembler = VectorAssembler(inputCols= ["age", "gender", "rate", "sbp", "dbp", "bs", "ckmb", "trop"], outputCol= "features")
output = assembler.transform(df)
modelDat = output.select("features", "result")

# Split the data into training and testing data and fit the model.
training_df, test_df = modelDat.randomSplit([0.8, 0.2])
spark.sparkContext._jvm.System.gc() # Run garbage collction
logReg = LogisticRegression(maxIter = 3, labelCol="result").fit(training_df)

# Get training and testing predicitions.
training_pred = logReg.evaluate(training_df).predictions
result = logReg.evaluate(test_df).predictions

# Evaluate the model,
tp = result[(result.result ==1) & (result.prediction==1)].count()
tn = result[(result.result ==0) & (result.prediction==0)].count()
fp = result[(result.result ==0) & (result.prediction==1)].count()
fn = result[(result.result ==1) & (result.prediction==0)].count()
accuracy = float((tp + tn)/(result.count()))
recall = float(tn)/(tp+tn)
precision = float(tp)/(tp+fn)
f1Score = float(2*(precision*recall)/(precision+recall))



# Load the results to hbase.
data = [('row1', 'cf:True Positive', str(tp)), ('row2', 'cf:True Negative', str(tn)), ('row3', 'cf:False Positive', str(fp)), ('row4', 'cf:False Negative', str(fn)), ('row5', 'cf:Accuracy', str(accuracy)), ('row6', 'cf:Recall', str(recall)), ('row7', 'cf:Precision', str(precision)), ('row8', 'cf:F1-score', str(f1Score))]

 
rdd = spark.sparkContext.parallelize(data)
spark.sparkContext._jvm.System.gc() # Run garbage collction
rdd.foreachPartition(write_to_hbase)
