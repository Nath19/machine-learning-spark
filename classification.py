# Remove the 'flight' column
flights = flights.drop('flight')

# Number of records with missing 'delay' values
flights.filter('delay IS NULL').count()

# Remove records with missing 'delay' values
flights = flights.filter('delay IS NOT NULL')

# Remove records with missing values in any column and get the number of remaining rows
flights = flights.dropna()
print(flights.count())

#########################################################################################

#SPLIT


# Split into training and testing sets in a 80:20 ratio
flights_train, flights_test = flights.randomSplit([0.8, 0.2], seed=17)

# Check that training set has around 80% of records
training_ratio = flights_train.count() / flights_test.count()
print(training_ratio)


#########################################################################################


# Import the Decision Tree Classifier class
from pyspark.ml.classification import DecisionTreeClassifier

# Create a classifier object and fit to the training data
tree = DecisionTreeClassifier()
tree_model = tree.fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
prediction = tree_model.transform(flights_test)
prediction.select('label', 'prediction', 'probability').show(5, False)





#########################################################################################




#Evaluate the decision tree


# Create a confusion matrix
prediction.groupBy("label", "prediction").count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = prediction').count()
TP = prediction.filter('prediction = 1 AND label = prediction').count()
FN = prediction.filter('prediction = 0 AND label = prediction').count()
FP = prediction.filter('prediction = 1 AND label = prediction').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TP+TN)/ ( TP + TN + FP + FN)
print(accuracy)


#########################################################################################


#Logistic regression



# Import the logistic regression class
from pyspark.ml.classification import LogisticRegression

# Create a classifier object and train on training data
logistic = LogisticRegression().fit(flights_train)

# Create predictions for the testing data and show confusion matrix
prediction = logistic.transform(flights_test)
prediction.groupBy("label", "prediction").count().show()





#########################################################################################




from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print('precision = {:.2f}\nrecall    = {:.2f}'.format(precision, recall))

# Find weighted precision
multi_evaluator =  MulticlassClassificationEvaluator()
weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(prediction,{binary_evaluator.metricName: "areaUnderROC"})




#########################################################################################


#Turning Text into Tables
""" 
- remove punctuation and numbers
- tokenize (split into individual words)
- remove stop words
- apply the hashing trick
- convert to TF-IDF representation.
"""

# Import the necessary functions
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer

# Remove punctuation (REGEX provided) and numbers
wrangled = sms.withColumn('text', regexp_replace(sms.text, '[_():;,.!?\\-]', ' '))
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, '[0-9]', ' '))

# Merge multiple spaces
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, ' +', ' '))

# Split the text into words
wrangled = Tokenizer(inputCol='text', outputCol='words').transform(wrangled)

wrangled.show(4, truncate=False)


#########################################################################################


#Stop words and hashing



from pyspark.ml.feature import StopWordsRemover , HashingTF, IDF

# Remove stop words.
wrangled = StopWordsRemover(inputCol='words', outputCol='terms')\
      .transform(sms)

# Apply the hashing trick
wrangled = HashingTF(inputCol='terms', outputCol='hash', numFeatures=1024)\
      .transform(wrangled)

# Convert hashed symbols to TF-IDF
tf_idf = IDF(inputCol='hash', outputCol='features')\
      .fit(wrangled).transform(wrangled)
      
tf_idf.select('terms', 'features').show(4, truncate=False)



#########################################################################################


# Training a spam classifier

# Split the data into training and testing sets
sms_train, sms_test = sms.randomSplit([0.8, 0.2], seed=13)

# Fit a Logistic Regression model to the training data
logistic = LogisticRegression(regParam=0.2).fit(sms_train)

# Make predictions on the testing data
prediction = logistic.transform(sms_test)

# Create a confusion matrix, comparing predictions to known labels
prediction.groupBy('label', 'prediction').count().show()










