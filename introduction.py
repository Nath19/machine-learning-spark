# Import the PySpark module
from pyspark.sql import SparkSession
# Create SparkSession object
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('test') \
                    .getOrCreate()

# What version of Spark?
print(spark.version)

# Terminate the cluster
spark.stop()




#########################################################################################


# Read data from CSV file
flights = spark.read.csv('flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

# Check column data types
flights.dtypes



#########################################################################################

#Loading data with specific schema

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
sms = spark.read.csv("sms.csv", sep=';', header=False, schema=schema)

# Print schema of DataFrame
sms.printSchema()

#########################################################################################

# Removing columns and rows




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



# Column Manipulation 


# Import the required function
from pyspark.sql.functions import round

# Convert 'mile' to 'km' and drop 'mile' column
flights_km = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                    .drop('mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights_km = flights_km.withColumn('label', (flights_km.delay >= 15).cast('integer'))

# Check first five records
flights_km.show(5)





#########################################################################################



# Assembling Columns 




# Import the necessary class
from pyspark.ml.feature import VectorAssembler

# Create an assembler object
assembler = VectorAssembler(inputCols=[
    'mon','dom','dow','carrier_idx','org_idx','km','depart','duration'
], outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flights)

# Check the resulting column
flights_assembled.select('features', 'delay').show(5, truncate=False)







