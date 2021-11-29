
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyspark
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import RobustScaler

from test2 import getMarker

def getMarker2(q):
	return 'X' if q<=5 else 'o'

# def getMarker3(q):
# 	return 'X' if q<=4 else 'o' if q<=5 else '+'

# def getMarker2(q):
# 	return 'X' if q<=5 else 'o'

# def getMarker2(q):
# 	return 'X' if q<=5 else 'o'

# def getMarker2(q):
# 	return 'X' if q<=5 else 'o'

# def getMarker2(q):
# 	return 'X' if q<=5 else 'o'

# getMarkerDict = {
# 	2:getMarker2,
# 	3:getMarker3,
# }

# func = getMarkerDict(k)

# func()

file_path = "./winequality-white.csv" 
spark = SparkSession.builder.appName("test").getOrCreate()
sqlContext = SQLContext(sparkContext = spark.sparkContext, sparkSession = spark)

"""
 |-- fixed acidity: string (nullable = true)
 |-- volatile acidity: string (nullable = true)
 |-- citric acid: string (nullable = true)
 |-- residual sugar: string (nullable = true)
 |-- chlorides: string (nullable = true)
 |-- free sulfur dioxide: string (nullable = true)
 |-- total sulfur dioxide: string (nullable = true)
 |-- density: string (nullable = true)
 |-- pH: string (nullable = true)
 |-- sulphates: string (nullable = true)
 |-- alcohol: string (nullable = true)
 |-- quality: string (nullable = true)
"""

scale_data = False
# Load white wine data
nullable = True
schema = StructType([
    StructField("fixed acidity", FloatType(), nullable), 
    StructField("volatile acidity", FloatType(), nullable), #
    StructField("citric acid", FloatType(), nullable), 
    StructField("residual sugar", FloatType(), nullable), #
	StructField("chlorides", FloatType(), nullable),
    StructField("free sulfur dioxide", FloatType(), nullable), #
    StructField("total sulfur dioxide", FloatType(), nullable), #
    StructField("density", FloatType(), nullable), #
	StructField("pH", FloatType(), nullable), #
    StructField("sulphates", FloatType(), nullable),
    StructField("alcohol", FloatType(), nullable), #
    StructField("quality", FloatType(), nullable),

])

data = spark.read.format("csv").option("header", "true").option("delimiter",";").schema(schema).load(file_path)
data.show()
data.printSchema()



print('########################################################################################')

# qualities available
qualities = [e.quality for e in data.select("quality").distinct().collect()]
print(qualities)
print('########################################################################################')
print('########################################################################################')
print('########################################################################################')
data.groupBy("quality").count().show()
# Create features column, assembling together the numeric data
vecAssembler = VectorAssembler(
    inputCols=['volatile acidity', 'residual sugar',
		'free sulfur dioxide', 'total sulfur dioxide', 'density',
		'pH', 'alcohol'],
    outputCol="features")

wine_with_features = vecAssembler.transform(data)
scaler = RobustScaler(inputCol="features", outputCol="scaled features",
                      withScaling=True, withCentering=False,
                      lower=0.25, upper=0.75)
#scaler = StandardScaler(inputCol="features", outputCol="scaled features",
#                        withStd=False, withMean=True)

scalerModel = scaler.fit(wine_with_features)

wine_with_features = scalerModel.transform(wine_with_features) ## 5otra
wine_with_features.select('features').show()

print('##########################################################################')

# Do K-means
k = 7 # TODO: test several k, elbow method
kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("scaled features" if scale_data else "features")
model = kmeans_algo.fit(wine_with_features)
centers = model.clusterCenters()
# Assign clusters to wines
wine_with_clusters = model.transform(wine_with_features)

print("Centers", centers)

# Convert Spark Data Frame to Pandas Data Frame
wine_for_viz = wine_with_clusters.toPandas()

# Vizualize
# Marker styles are calculated from Wine qualities
wine_quality_repartitions = [wine_for_viz [ wine_for_viz['quality'] == e ] for e in qualities]
print([len(e) for e in wine_quality_repartitions])

# qualities code k-means results, cluster numbers

rgbs = np.random.rand(len(wine_quality_repartitions),3)
colors = {}
for i in range(k):
	colors[i] = (rgbs[i][0], rgbs[i][1], rgbs[i][2])


print(colors)

markers = ['o', 'v', '^', '>', '<', '1', 's', 'p', 'P', '*', 'X']
print(len(qualities))
print(len(markers))
fig = plt.figure().gca(projection='3d')
for i in range(len(qualities)):
	fig.scatter(wine_quality_repartitions[i]['volatile acidity'],
    	         wine_quality_repartitions[i]['volatile acidity'],
    	         wine_quality_repartitions[i]['free sulfur dioxide'],
    	         c = wine_quality_repartitions[i].prediction.map(colors),
				 marker = markers[i]) 
	print('!!!!!!!!!!!!!!!!!!!!!!')
	print(i)

fig.set_xlabel('volatile acidity')
fig.set_ylabel('volatile acidity')
fig.set_zlabel('free sulfur dioxide')
plt.show()



### PCA

# Dimenstion reduction. From 11D to 3D
# by PCA method
datamatrix =  RowMatrix(data.select(['volatile acidity', 'residual sugar',
		'free sulfur dioxide', 'total sulfur dioxide', 'density',
		'pH', 'alcohol']).rdd.map(list))

# Compute the top 3 principal components. The "best" hyperplane.
pc = datamatrix.computePrincipalComponents(3)
print ("***** 3 Principal components *****")
print(pc)

# project data
projected = datamatrix.multiply(pc)
new_X = pd.DataFrame(
    projected.rows.map(lambda x: x.values[0]).collect()
)
new_Y = pd.DataFrame(
    projected.rows.map(lambda x: x.values[1]).collect()
)
new_Z = pd.DataFrame(
    projected.rows.map(lambda x: x.values[2]).collect()
)
print(new_X)
# Vizualize with PCA, 3 components
# Colors code k-means results, cluster numbers

fig = plt.figure().gca(projection='3d')

for i in range(len(wine_quality_repartitions)):
	print(wine_quality_repartitions[i].prediction.map(colors))
	fig.scatter(new_X [wine_for_viz['quality'] == qualities[i]],
    	         new_Y [wine_for_viz['quality'] == qualities[i]],
    	         new_Z [wine_for_viz['quality'] == qualities[i]],
    	         c = wine_quality_repartitions[i].prediction.map(colors),
				 marker = markers[i]) 

fig.set_xlabel('Component 1')
fig.set_ylabel('Component 2')
fig.set_zlabel('Component 3')
plt.show()


# Dimenstion reduction. From 11D to 2D
# by PCA method
# Compute the top 3 principal components. The "best" hyperplane.
pc = datamatrix.computePrincipalComponents(2)
print ("***** 2 Principal components. The same as first 2 of 3 principal components *****")
print (pc)

# project data
projected = datamatrix.multiply(pc)
new_X = pd.DataFrame(
    projected.rows.map(lambda x: x.values[0]).collect()
)
new_Y = pd.DataFrame(
    projected.rows.map(lambda x: x.values[1]).collect()
)

# Vizualize with PCA, 2 components
# Colors code k-means results, cluster numbers
# fig = plt.figure().gca()

# for i in range(len(wine_quality_repartitions)):
# 	fig.scatter(new_X [wine_for_viz['quality'] == qualities[i]],
#     	         new_Y [wine_for_viz['quality'] == qualities[i]],
#     	         c = wine_quality_repartitions[i].prediction.map(colors),
# 				 marker = markers[i]) 


# fig.set_xlabel('Component 1')
# fig.set_ylabel('Component 2')
# plt.show()

###################### Density map ######################

# fig = plt.figure().gca()
# fig.hist2d(wine_for_viz['quality'],
# 	        wine_for_viz['prediction']) 
# fig.set_xlabel('Component 1')
# fig.set_ylabel('Component 2')
# plt.show()

# for Q in qualities:
#   fig = plt.figure().gca()
#   fig.scatter(new_X [wine_for_viz['quality'] == Q],
#     	         new_Y [wine_for_viz['quality'] == Q],
#     	         c = wine_for_viz [ wine_for_viz['quality'] == Q ].prediction.map(colors),
# 				 marker = markers[0]) 
#   fig.set_xlabel('Component 1')
#   fig.set_ylabel('Component 2')
#   plt.show()


###################### Sijhoutte ######################
from pyspark.ml.evaluation import ClusteringEvaluator
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(wine_with_clusters)
print("Silhouette with squared euclidean distance = " + str(silhouette))

##################################################################################################

# Spliting into two categories:
# 1. Quality smaller than 6
# 2. Quality bigger or equal to 6

# fig = plt.figure().gca()
# fig.scatter(new_X [wine_for_viz['quality'] < 6],
# 				new_Y [wine_for_viz['quality'] < 6],
# 				marker = markers[0]) 
# fig.scatter(new_X [wine_for_viz['quality'] >= 6],
# 				new_Y [wine_for_viz['quality'] >= 6],
# 				marker = markers[1]) 
# fig.set_xlabel('Component 1')
# fig.set_ylabel('Component 2')
# plt.show()


###########################################################################################

# Spliting into 4 categories:
# 1. Quality smaller than 5
# 2. Quality equal to 5
# 3. Quality equal to 6
# 4. Quality bigger than 6

# fig = plt.figure().gca()
# fig.scatter(new_X [wine_for_viz['quality'] < 5],
# 				new_Y [wine_for_viz['quality'] < 5],
# 				marker = markers[0]) 
# fig.scatter(new_X [wine_for_viz['quality'] == 5],
# 				new_Y [wine_for_viz['quality'] == 5],
# 				marker = markers[1])
# fig.scatter(new_X [wine_for_viz['quality'] == 6],
# 				new_Y [wine_for_viz['quality'] == 6],
# 				marker = markers[2]) 
# fig.scatter(new_X [wine_for_viz['quality'] > 6],
# 				new_Y [wine_for_viz['quality'] > 6],
# 				marker = markers[3])  
# fig.set_xlabel('Component 1')
# fig.set_ylabel('Component 2')
# plt.show()

# wine_cluter_filter = wine_with_clusters.where(wine_with_clusters.quality < 5)
# silhouette = evaluator.evaluate(wine_cluter_filter)
# print("Silhouette with squared euclidean distance = " + str(silhouette))
# wine_cluter_filter = wine_with_clusters.where(wine_with_clusters.quality == 5)
# silhouette = evaluator.evaluate(wine_cluter_filter)
# print("Silhouette with squared euclidean distance = " + str(silhouette))
# wine_cluter_filter = wine_with_clusters.where(wine_with_clusters.quality == 6)
# silhouette = evaluator.evaluate(wine_cluter_filter)
# print("Silhouette with squared euclidean distance = " + str(silhouette))
# wine_cluter_filter = wine_with_clusters.where(wine_with_clusters.quality > 6)
# silhouette = evaluator.evaluate(wine_cluter_filter)
# print("Silhouette with squared euclidean distance = " + str(silhouette))

preds = wine_for_viz[wine_for_viz['quality'] < 6].prediction
actuals = wine_for_viz[wine_for_viz['quality'] < 6]['quality']
under_6_incluster0 = 0
under_6_incluster1 = 0
over_6_incluster0 = 0
over_6_incluster1 = 0

for p in preds:
	if p == 0:
		under_6_incluster0 += 1
	else: 
		under_6_incluster1 += 1

preds = wine_for_viz[wine_for_viz['quality'] >= 6].prediction
actuals = wine_for_viz[wine_for_viz['quality'] >= 6]['quality']
for p in preds:
	if p == 0:
		over_6_incluster0 += 1
	else:
		over_6_incluster1 += 1
