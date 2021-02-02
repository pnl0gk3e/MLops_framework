# Databricks notebook source
#import torch
#Check what are mounted
dbutils.fs.ls('/mnt/input-data/')
#input-data is here by unchecked creation
#items = spark.read.format("delta").load("/mnt/input-data/items")
/mnt/input-data

# COMMAND ----------

#customers = spark.read.format("delta").load("/mnt/input-data/transactions")
customers.filter(F.col("TransactionDt") < datetime.now()).count()

# COMMAND ----------

customers.filter

# COMMAND ----------

#Check details within the mounted dataset
#dbutils.fs.ls('/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/')
customers = spark.read.format("delta").load( \
                      "/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/als_user_factors/")
rules = spark.read.format("parquet").load( \
                      "/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/combined_rules_df.parquet")

# COMMAND ----------

#customers.show(5)
rules.show(5)
#dbutils.fs.ls('/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/')

# COMMAND ----------

import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

# read the file - (21892, 9624) - (cust, rules)
input_recommender= spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/relevance_2/")

df_input_recommender= input_recommender.toPandas()
df_input_recommender["cust"] = df_input_recommender["cust"].astype("str")
df_input_recommender["rule_id"] = df_input_recommender["rule_id"].astype("str")
df_input_recommender["relevance"] = df_input_recommender["relevance"].astype("float")
df_input_recommender= df_input_recommender[df_input_recommender["relevance"]>0].sample(frac =1)
unique_user_ids = df_input_recommender.groupby(["cust"]).agg({"rule_id":"nunique"}).reset_index()["cust"].values
unique_rule_ids = df_input_recommender.groupby(["rule_id"]).agg({"cust":"nunique"}).reset_index()["rule_id"].values

features = df_input_recommender.filter(["cust","rule_id","relevance"], axis=1)
#labels = df_input_recommender.filter(["full_count"], axis=1)
df_rules = df_input_recommender.filter(["rule_id"], axis=1)

dataset = tf.data.Dataset.from_tensor_slices((dict(features)))
rules = tf.data.Dataset.from_tensor_slices((dict(df_rules))).map(lambda x: x["rule_id"])


class MovielensModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

    embedding_dimension = 8

    # User and movie models.
    self.rule_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_rule_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_rule_ids) + 1, embedding_dimension)
    ])
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=rules.batch(32).map(self.rule_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["cust"])
    # And pick out the movie features and pass them into the movie model.
    rule_embeddings = self.rule_model(features["rule_id"])

    return (
        user_embeddings,
        rule_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.rating_model(
            tf.concat([user_embeddings, rule_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("relevance")
    
    user_embeddings, rule_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, rule_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss)
  
model = MovielensModel(rating_weight=0.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))




tf.random.set_seed(42)
shuffled = dataset.shuffle(features.shape[0], seed=42, reshuffle_each_iteration=False)

train = shuffled.take(features.shape[0])
test = shuffled.take(10_000)

cached_train = train.batch(320).cache()
cached_test = test.batch(160).cache()

model.fit(cached_train, epochs=1)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")




# COMMAND ----------

#customers.filter(F.col("cust") == 88269765).show(10)
transactions = spark.read.format("delta").load("/mnt/input-data/transactions")
products = spark.read.format("delta").load("/mnt/input-data/items_reformat")
master = transactions.join(products,"RetailItemNbr", "inner").withColumnRenamed("MemberNbr", "cust")
online_customers = master.withColumn("in_store", (F.col("Channel") == "Store").astype("int")) \
                    .groupBy("cust").agg(F.mean("in_store").alias("in_store_percentage")) \
                    .filter(F.col("in_store_percentage") <= 1.0) \
                    .withColumn("online_customer", F.lit(True)) \
                    .select(F.col("cust").alias("online_cust_nbr"), F.col("online_customer"))

master = master.join(online_customers, master.cust == online_customers.online_cust_nbr, "left")

'''

online_prods = transactions.filter(F.col('Channel') ==  "Online").select(F.col("RetailItemNbr").alias('online_prods')).dropDuplicates()
transactions_next = transactions.join(online_prods, F.col("RetailItemNbr") == F.col('online_prods'), "inner")
#transactions.filter(F.col("MemberNbr") == 88269765).orderBy("TransactionDt", ascending =False).show(100)
online_customers = transactions.withColumn("in_store", (F.col("Channel") == "Store").astype("int")) \
                .groupBy("MemberNbr").agg(F.mean("in_store").alias("in_store_percentage")) \
                .filter(F.col("in_store_percentage") <= 1.0) \
                .withColumn("online_customer", F.lit(True)) \
                .select(F.col("MemberNbr").alias("online_cust_nbr"), F.col("online_customer"))
transactions = transactions.join(online_customers, transactions.MemberNbr == online_customers.online_cust_nbr, "left")
'''


# COMMAND ----------

#transactions.filter(col("TransactionDt") >= datetime.now() - timedelta(days=700)).groupBy("TransactionId").count().count(),

#master.groupBy("TransactionId").count().count()
#online_customers.groupBy("online_cust_nbr").count().count()
online_customers.filter(F.col("online_cust_nbr")==88269765).show(10)

# COMMAND ----------

transactions.groupBy("TransactionId").count().count()

# COMMAND ----------

a = spark.read.format("delta").load("/mnt/input-data/master_before_sample/")
b = spark.read.format("delta").load("/mnt/input-data/master_after_sample")
c = spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/output_intersection/")
d = spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/output_add_relevance/")
tmp = c.filter(F.col("cust") == 88269765).show(100)
#relevance.show(10)
a.groupBy("cust").count().count(), b.groupBy("cust").count().count() , c.groupBy("cust").count().count(),d.groupBy("cust").count().count()

# COMMAND ----------


from datetime import datetime, timedelta
import pyspark.sql.functions as F
from pyspark.sql.functions import col, size, sum, array_except, approx_count_distinct, count, avg, stddev, skewness
import numpy as np
from scipy.spatial.distance import cosine
online = a.filter(F.col("online_customer") == F.lit(True))
online.count()

# COMMAND ----------

#a.groupBy("Channel").count().show(10)
a.filter(F.col("cust") == 88269765).show(100)



# COMMAND ----------

#online.groupBy("cust").count().count()
transactions.filter(col("businessdate") >= datetime.now() - timedelta(days=700)).groupBy("")


# COMMAND ----------

import numpy as np
# read the file - (21892, 9624) - (cust, rules)
input_recommender= spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/output_add_relevance")

df_input_recommender= input_recommender.toPandas()
df_input_recommender["cust"] = df_input_recommender["cust"].astype("str")
df_input_recommender["rule_id"] = df_input_recommender["rule_id"].astype("str")
df_input_recommender["full_count"] = df_input_recommender["full_count"].astype("float")

cust_vocab = df_input_recommender.groupby(["cust"]).agg({"rule_id":"nunique"}).reset_index()["cust"].values
rule_vocab = df_input_recommender.groupby(["rule_id"]).agg({"cust":"nunique"}).reset_index()["rule_id"].values

features = df_input_recommender.filter(["cust","rule_id"], axis=1)
labels = df_input_recommender.filter(["full_count"], axis=1)
dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

'''
ds_input_recommender = (
  tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(df_input_recommender["cust"].values, tf.string),
            tf.cast(df_input_recommender['rule_id'].values,tf.string),
            tf.cast(df_input_recommender["full_count"].values, tf.float32)
    )
    )
)

'''



# COMMAND ----------

features = df_input_recommender.filter(["cust","rule_id"], axis=1)
labels = df_input_recommender.filter(["full_count"], axis=1)
dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))



# COMMAND ----------

ratings = tfds.load('movielens/100k-ratings', split="train")
movies = tfds.load('movielens/100k-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
})
movies = movies.map(lambda x: x["movie_title"])

#And repeat our preparatins for building vocabularies and splitting the data into a train and a test set:

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# COMMAND ----------

x= spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/input_relevance_master")
y = spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/input_relevance_rules")
als_user = spark.read.format("delta").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/als_user_factors/")
als_rules = spark.read.format("delta").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/als_item_factors/")


# COMMAND ----------

 master= spark.read.format("delta").load("/mnt/input-data/master_before_sample/")
master.orderBy("TransactionDt", ascending = True).show(10)


# COMMAND ----------


# testing what rules are hit for the test basket
import pandas as pd
query_basket = pd.DataFrame([120452335,120126741,111011141,112054046,120363148], columns = ["basket"])
query_basket["id"] ="test"
query_basket = spark.createDataFrame(query_basket)
query_basket = query_basket.groupBy("id").agg(F.collect_list("basket").alias("query"))
query_result = y.crossJoin(query_basket).withColumn("full_ant_match",
                  (F.size(F.array_except(F.col("antecedent_formatted_peapod_id"),
                                     F.col("query")))==0).cast('integer')).withColumn("cons_match",
                  (F.size(F.array_except(F.col("consequent"),
                                     F.col("query")))==0).cast('integer')) 






# COMMAND ----------


import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

# read the file - (21892, 9624) - (cust, rules)
input_recommender= spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/relevance_2/")

df_input_recommender= input_recommender.toPandas()
df_input_recommender["cust"] = df_input_recommender["cust"].astype("str")
df_input_recommender["rule_id"] = df_input_recommender["rule_id"].astype("str")
df_input_recommender["relevance"] = df_input_recommender["relevance"].astype("float")
df_input_recommender= df_input_recommender[df_input_recommender["relevance"]>0].sample(frac =.1)
unique_user_ids = df_input_recommender.groupby(["cust"]).agg({"rule_id":"nunique"}).reset_index()["cust"].values
unique_rule_ids = df_input_recommender.groupby(["rule_id"]).agg({"cust":"nunique"}).reset_index()["rule_id"].values

features = df_input_recommender.filter(["cust","rule_id","relevance"], axis=1)
#labels = df_input_recommender.filter(["full_count"], axis=1)
df_rules = df_input_recommender.filter(["rule_id"], axis=1)

# COMMAND ----------



# COMMAND ----------

import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

# read the file - (21892, 9624) - (cust, rules)
input_recommender= spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/relevance_2/")

df_input_recommender= input_recommender.toPandas()
df_input_recommender["cust"] = df_input_recommender["cust"].astype("str")
df_input_recommender["rule_id"] = df_input_recommender["rule_id"].astype("str")
df_input_recommender["relevance"] = df_input_recommender["relevance"].astype("float")
df_input_recommender= df_input_recommender[df_input_recommender["relevance"]>0].sample(frac =.1)
unique_user_ids = df_input_recommender.groupby(["cust"]).agg({"rule_id":"nunique"}).reset_index()["cust"].values
unique_rule_ids = df_input_recommender.groupby(["rule_id"]).agg({"cust":"nunique"}).reset_index()["rule_id"].values

features = df_input_recommender.filter(["cust","rule_id","relevance"], axis=1)
#labels = df_input_recommender.filter(["full_count"], axis=1)
df_rules = df_input_recommender.filter(["rule_id"], axis=1)

dataset = tf.data.Dataset.from_tensor_slices((dict(features)))
rules = tf.data.Dataset.from_tensor_slices((dict(df_rules))).map(lambda x: x["rule_id"])


class MovielensModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

    embedding_dimension = 8

    # User and movie models.
    self.rule_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_rule_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_rule_ids) + 1, embedding_dimension)
    ])
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=rules.batch(32).map(self.rule_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["cust"])
    # And pick out the movie features and pass them into the movie model.
    rule_embeddings = self.rule_model(features["rule_id"])

    return (
        user_embeddings,
        rule_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.rating_model(
            tf.concat([user_embeddings, rule_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("relevance")
    
    user_embeddings, rule_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, rule_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss)
  
model = MovielensModel(rating_weight=0.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.001))




tf.random.set_seed(42)
shuffled = dataset.shuffle(features.shape[0], seed=42, reshuffle_each_iteration=False)

train = shuffled.take(features.shape[0])
test = shuffled.take(10_000)

cached_train = train.batch(3200).cache()
cached_test = test.batch(160).cache()

model.fit(cached_train, epochs=1)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")


# COMMAND ----------

# map to the dictionary
rule_vectors = model.rule_model
user_vectors = model.user_model
rule_index = rule_vectors.layers[0].get_weights()[0]
rule_embeddings = rule_vectors.layers[1].get_weights()[0]
user_index = user_vectors.layers[0].get_weights()[0]
user_embeddings = user_vectors.layers[1].get_weights()[0]
rule_dict = {}
user_dict = {}
for index,item in enumerate(rule_index):
  rule_dict[item] = rule_embeddings[index]
for index,item in enumerate(user_index):
  user_dict[item] = user_embeddings[index]

# COMMAND ----------

#user_dict
df_input_recommender = df_input_recommender[df_input_recommender["relevance"]>0].sample(frac =.1)

# COMMAND ----------

 spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
# get the vectors and TSNE
rule_vectors = model.rule_model
user_vectors = model.user_model
rule_index = rule_vectors.layers[0].get_weights()[0]
embeddings = rule_vectors.layers[1].get_weights()[0]
X_embedded = TSNE(n_components=2).fit_transform(embeddings)
df_tsne = pd.DataFrame(X_embedded, columns = ["X",'Y'])
sns.scatterplot(data=df_tsne, x="X", y="Y")

# COMMAND ----------

#model.evaluate(cached_test, return_dict=True)
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index(rules.batch(100).map(model.rule_model), rules)

# Get recommendations.
_, titles = index(tf.constant(["99788191"]))
print(f"Recommendations for user 99788191: {titles[0, :20]}")

# COMMAND ----------

# get the table here with als matrix
from datetime import datetime, timedelta
import pyspark.sql.functions as F
from pyspark.sql.functions import col, size, sum, array_except, approx_count_distinct, count, avg, stddev, skewness
import numpy as np
from scipy.spatial.distance import cosine
rule_table = spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/input_relevance_rules")
als_user = spark.read.format("delta").load("/mnt/input-data/all_customers/")
als_rules = spark.read.format("delta").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/als_item_factors/")
rule_table = rule_table.join(als_rules, rule_table.rule_id == als_rules.id, "inner")
query_vector = als_user.filter(F.col("cust") == 88269765).select(F.col("features")).rdd.flatMap(list).first()
rule_table = rule_table.withColumn("query", F.array([F.lit(x) for x in query_vector]))
# get the table here with tf matrix

# COMMAND ----------

from datetime import datetime, timedelta
import pyspark.sql.functions as F
from pyspark.sql.functions import col, size, sum, array_except, approx_count_distinct, count, avg, stddev, skewness
import numpy as np
from scipy.spatial.distance import cosine


rule_table = rule_table.toPandas()
# write our UDF for cosine similarity
def sim_cos(a,b):
  return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
  


#result = rule_table.select(sim_cos("features", "query").alias("sim_cosine"))


#rule_table = rule_table.withColumn("query", (F.col("Channel") == "Store").astype("int"))

# COMMAND ----------

#als_user.show(10)
x.filter(F.col("cust") == 88269765).show(10)


#rule_table['cosine'] = rule_table.apply(lambda row: 1 - cosine(row['features'], row['query']), axis=1)
#rule_table.sort_values(["cosine"], ascending = False)

# COMMAND ----------

input_recommender= spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/output_add_relevance")

df_input_recommender= input_recommender.toPandas()
df_input_recommender["cust"] = df_input_recommender["cust"].astype("str")
df_input_recommender["rule_id"] = df_input_recommender["rule_id"].astype("str")
df_input_recommender["full_count"] = df_input_recommender["full_count"].astype("float")

unique_user_ids = df_input_recommender.groupby(["cust"]).agg({"rule_id":"nunique"}).reset_index()["cust"].values
unique_rule_ids = df_input_recommender.groupby(["rule_id"]).agg({"cust":"nunique"}).reset_index()["rule_id"].values

features = df_input_recommender.filter(["cust","rule_id","full_count"], axis=1)
labels = df_input_recommender.filter(["full_count"], axis=1)
df_rules = df_input_recommender.filter(["rule_id"], axis=1)
dataset = tf.data.Dataset.from_tensor_slices((dict(features)))
rules = tf.data.Dataset.from_tensor_slices((dict(df_rules))).map(lambda x: x["rule_id"])



# COMMAND ----------

rules.batch(100)

# COMMAND ----------

# training the movie lesns model
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

ratings = tfds.load('movielens/100k-ratings', split="train")
movies = tfds.load('movielens/100k-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
})
movies = movies.map(lambda x: x["movie_title"])

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

class MovielensModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

    embedding_dimension = 32

    # User and movie models.
    self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.movie_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model.
    movie_embeddings = self.movie_model(features["movie_title"])

    return (
        user_embeddings,
        movie_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.rating_model(
            tf.concat([user_embeddings, movie_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("user_rating")

    user_embeddings, movie_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss

            + self.retrieval_weight * retrieval_loss)
  
model = MovielensModel(rating_weight=0.5, retrieval_weight=0.5)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

# get the vectors and TSNE
movie_vectors = model.movie_model
user_vectors = model.user_model

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
embeddings = user_vectors.layers[1].get_weights()[0]
X_embedded = TSNE(n_components=2).fit_transform(embeddings)
df_tsne = pd.DataFrame(X_embedded, columns = ["X",'Y'])
sns.scatterplot(data=df_tsne, x="X", y="Y")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from typing import Dict, Text

import numpy as npdd
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
# Ratings data and movie to get pandas dataframe
ratings, info_ratings = tfds.load('movielens/100k-ratings', split="train", with_info=True)
movies, info_movies = tfds.load('movielens/100k-movies', split="train",with_info=True)


# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x: x["movie_title"])

#Build vocabularies to convert user ids and movie titles into integer indices for embedding layers:

user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies)


# COMMAND ----------


tfds.as_dataframe(movies.take(4), info_ratings)


# COMMAND ----------

ratings = ds_input_recommender.take(-1)  # Only take a single example
for index,example in enumerate(ratings):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
  print(type(example))
  print(list(example.keys()), index)
  movie = example["cust"]
  user = example["rule_id"]
  print(movie, user)
  

# COMMAND ----------

class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(self,user_model: tf.keras.Model,movie_model: tf.keras.Model,task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["cust"])
    movie_embeddings = self.movie_model(features["rule_id"])

    return self.task(user_embeddings, movie_embeddings)
  
# Define user and movie models.


user_model = tf.keras.Sequential([
    user_ids_vocabulary,
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
])
movie_model = tf.keras.Sequential([
    movie_titles_vocabulary,
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    movies.batch(128).map(movie_model)
  )
)
# Create a retrieval model.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(ratings.batch(4096), epochs=10)

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index(movies.batch(100).map(model.movie_model), movies)

# Get some recommendations.
_, titles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")


# COMMAND ----------

x

# COMMAND ----------

# See what is in a file
#c= spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/output_dataloading_master")
#b =spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/category_mining")
#a = spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/input_mining_master")
x= spark.read.format("delta").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/relevance")
# to get the dynamimc flter in the rule relevance
#x = x.filter(col("businessdate") >= datetime.now() - timedelta(days=100))
#y= spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/input_relevance_rules")

# COMMAND ----------

f= spark.read.format("delta").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/als_user_factors/")
f.groupBy("cust").count().count()

# COMMAND ----------

import json
import logging

import matplotlib.pyplot as plt
import pyspark
import pyspark.sql.functions as F
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, size, array_union, collect_list, element_at, explode, lit, when, \
    array_repeat, monotonically_increasing_id, dense_rank, approx_count_distinct 
fp_growth_prod = FPGrowth(itemsCol="formatted_peapod_id",
                             minSupport=200/400000,
                             minConfidence=.0025)
fp_growth_prod_model= fp_growth_prod.fit(x)
fp_growth_cat = FPGrowth(itemsCol="prdt_lvl_6_dsc",
                             minSupport=200/400000,
                             minConfidence=.025)
fp_growth_cat_model= fp_growth_cat.fit(x)


# COMMAND ----------

 a_rules = fp_growth_prod_model.associationRules \
            .select("antecedent",
                    "consequent",
                    "lift",
                    "confidence",
                    size("antecedent").alias("lhs_len"),
                    size("consequent").alias("rhs_len")
                    )
  
a_rules = a_rules.filter(col("rhs_len") == 1) \
            .filter(col("lift") >= 10) \
            .withColumn("full_rule", array_union("antecedent", "consequent"))
item_sets_a = fp_growth_prod_model.freqItemsets
a_rules = a_rules.join(item_sets_a.select("freq", "items"),
                             item_sets_a.items == a_rules.full_rule, 'inner') \
            .select('antecedent',
                    'consequent',
                    'lhs_len',
                    item_sets_a['freq'],
                    'lift',
                    'confidence') \
            .withColumnRenamed("freq", "rule_freq") \
            .withColumnRenamed("antecedent", "antecedent_formatted_peapod_id") \
            .withColumn("level", lit("formatted_peapod_id")) 

b_rules = fp_growth_cat_model.associationRules \
            .select("antecedent",
                    "consequent",
                    "lift",
                    "confidence",
                    size("antecedent").alias("lhs_len"),
                    size("consequent").alias("rhs_len")
                    )
b_rules = b_rules.filter(col("rhs_len") == 1) \
            .filter(col("lift") >= 20) \
            .withColumn("full_rule", array_union("antecedent", "consequent"))
item_sets_b = fp_growth_cat_model.freqItemsets
b_rules = b_rules.join(item_sets_b.select("freq", "items"),
                             item_sets_b.items == b_rules.full_rule, 'inner') \
            .select('antecedent',
                    'consequent',
                    'lhs_len',
                    item_sets_b['freq'],
                    'lift',
                    'confidence') \
            .withColumnRenamed("freq", "rule_freq") \
            .withColumnRenamed("antecedent", "antecedent_prdt_lvl_6_dsc") \
            .withColumn("level", lit("prdt_lvl_6_dsc")) \



# COMMAND ----------


rules_to_convert = b_rules
# Limit to the top_n items per category
top_products = b.filter(col("rank") <= 2)
# Generate a df containing categories, and for each the list of top_n UPCs
replacement_upcs = top_products.groupBy("prdt_lvl_6_dsc").agg(
    collect_list("formatted_peapod_id").alias("replacement_items"))
# Do an inner join with the rules to match replacement upc lists with rule antecedents
replaced = rules_to_convert.join(replacement_upcs,
                                 col("prdt_lvl_6_dsc") == element_at(col("consequent"), 1))
# Now use explode to seperate each rule with a list of antecedent options to multiple copies of the rule with
# 1 antecedent each. Use array_repeat to pack back into an array to match format of product rules
replaced = replaced.withColumn("consequent", explode(col("replacement_items"))) \
    .withColumn("consequent", array_repeat(col("consequent"), 1)) \
    .drop("replacement_items", "prdt_lvl_6_dsc") 
category_rules = replaced


# COMMAND ----------

# rule-merge
i = "formatted_peapod_id"
j = "prdt_lvl_6_dsc"
a_rules = a_rules.withColumn(f"antecedent_{j}", lit(None).astype("array<string>"))
category_rules = category_rules.withColumn(f"antecedent_{i}", lit(None).astype("array<int>"))
c = category_rules.columns
combined_rules = a_rules[c].union(category_rules[c])

max_rules_per_antecedent = 10
combined_rules = combined_rules \
    .withColumn("lift", col("lift") * when(col("level") == "formatted_peapod_id" , lit(1)).otherwise(lit(.7))) \
     .withColumn('rule_id', dense_rank().over(Window.orderBy(monotonically_increasing_id()))).drop("confidence") \
     .withColumn('rank',F.dense_rank().over(Window.partitionBy('antecedent_prdt_lvl_6_dsc','antecedent_formatted_peapod_id') \
                   .orderBy(F.desc('lift')))).filter(F.col('rank') <= max_rules_per_antecedent)

# COMMAND ----------

# rule relevance
from pyspark.sql.functions import array_except, array_intersect, explode, coalesce
intersections = x.select("cust", "formatted_peapod_id", "prdt_lvl_6_dsc") \
      .crossJoin(y.select("antecedent_formatted_peapod_id", 'antecedent_prdt_lvl_6_dsc',"consequent", 'rule_id')) \
       .withColumn("inter_LHS_product",
                  (size(array_except(col("antecedent_formatted_peapod_id"),
                                     col("formatted_peapod_id")))==0).cast('integer')) \
      .withColumn("inter_LHS_category",
                  (size(array_except(col('antecedent_prdt_lvl_6_dsc'),
                                     col('prdt_lvl_6_dsc'))) == 0).cast('integer')) \
      .withColumn("inter_RHS",
                  (size(array_except(col("consequent"), col("formatted_peapod_id"))) == 0).cast('integer')) \
      .withColumn("inter_full_category", 
                  (col("inter_LHS_category") * col("inter_RHS"))) \
      .withColumn("inter_full_product", 
                  (col("inter_LHS_product") * col("inter_RHS"))) \
      .filter((col("inter_LHS_product") == 1) | (col("inter_LHS_category") == 1) | (col("inter_RHS") == 1)) \
      .groupBy('cust', 'rule_id') \
      .agg(F.sum('inter_LHS_product').alias('ant_count_product'), \
           F.sum('inter_LHS_category').alias('ant_count_category'), \
           F.sum('inter_RHS').alias('cons_count'), \
           F.sum('inter_full_category').alias('full_count_category'), \
           F.sum('inter_full_product').alias('full_count_product')) \
          .filter((col('full_count_product') > 0) | (col('full_count_category') > 0))
  

# COMMAND ----------

r= intersections.filter(col("full_count_product")>0).groupBy("rule_id").count().count()
c= intersections.filter(col("full_count_product")>0).groupBy("cust").count().count()

# COMMAND ----------

r,c

# COMMAND ----------

r,c

# COMMAND ----------

r,c

# COMMAND ----------

#Rules which only show in category antecedent but not in consequent - rules(5584) cust(8759)
intersections.filter((col("ant_count_category")>0) & (col("cons_count")==0)).groupBy("rule_id").count().count()
intersections.filter((col("ant_count_category")>0) & (col("cons_count")==0)).groupBy("cust").count().count()
#Rules which only show in product antecedent but not in consequent - rules(14235), cust(9852)
intersections.filter((col("ant_count_product")>0) & (col("cons_count")==0)).groupBy("rule_id").count().count()
intersections.filter((col("ant_count_product")>0) & (col("cons_count")==0)).groupBy("cust").count().count(
#Rules which does not show in category antecedent but in consequent- rules(24723) cust(9865)
intersections.filter((col("ant_count_category")==0) & (col("cons_count")>0)).groupBy("rule_id").count().count()
intersections.filter((col("ant_count_category")==0) & (col("cons_count")>0)).groupBy("cust").count().count()
#Rules which does not show in product antecedent but in consequent - rules(24669), cust(9768)
intersections.filter((col("ant_count_product")==0) & (col("cons_count")>0)).groupBy("rule_id").count().count()
intersections.filter((col("ant_count_product")==0) & (col("cons_count")>0)).groupBy("cust").count().count()
#Rules which shows in both in category antecedent and consequent - rules(1301) , cust(1110)
intersections.filter(col("full_count_category")>0).groupBy("rule_id").count().count()
intersections.filter(col("full_count_category")>0).groupBy("cust").count().count()
#Rules which shows in both product antecedent and consequent - rules(5566)  , count(5454)
intersections.filter(col("full_count_product")>0).groupBy("rule_id").count().count()
intersections.filter(col("full_count_product")>0).groupBy("cust").count().count()
# total customer for whom rules where found to be relevant - rules (6867) ,cust(5616)
intersections.filter((col("full_count_category")>0) | (col("full_count_product")>0)).groupBy("cust").count().count()


# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
als_rules =spark.read.format("delta").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/als_item_factors/")
als_rules_col = als_rules.select(['id']+[F.expr('features[' + str(x) + ']') for x in range(0, 20)])
df_als_rules = als_rules_col.toPandas()
X_embedded = TSNE(n_components=2).fit_transform(df_als_rules.filter(df_als_rules.columns[1:]))
df_tsne = pd.DataFrame(X_embedded, columns = ["X",'Y'])
sns.scatterplot(data=df_tsne, x="X", y="Y")

# COMMAND ----------

!from __future__ import print_function
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from collections import namedtuple
import tensorflow as tf
import tensorflow.summary
from tensorflow.summary import scalar
from tensorflow.summary import histogram
from chardet.universaldetector import UniversalDetector

# COMMAND ----------

max_rules_per_antecedent = 10

combined_rules = combined_rules \
    .withColumn("lift", col("lift") * when(col("level") == self.product_identifier, lit(1))
                .otherwise(lit(category_lift_discount_coef))) \
    .withColumn('rule_id', dense_rank().over(Window.orderBy(monotonically_increasing_id()))) \
    .withColumn('rank',
                F.dense_rank()
                .over(Window.partitionBy('antecedent_prdt_lvl_6_dsc',
                                         'antecedent_formatted_peapod_id')
                      .orderBy(F.desc('lift')))) \
    .filter(F.col('rank') <= max_rules_per_antecedent)  


# COMMAND ----------

replacement_upcs.show(10)

# COMMAND ----------

rules_to_convert.join(replacement_upcs,
                                 col("prdt_lvl_6_dsc") == element_at(col("consequent"), 1)).show(10)

# COMMAND ----------

# Getting the frequency of each product 
import pyspark.sql.functions as F
from pyspark.sql import Window
transactions = spark.read.format("delta").load("/mnt/input-data/transactions")
window = Window.partitionBy(['MemberNbr','RetailItemNbr']).orderBy('TransactionDt')
trasactions_part = transactions.withColumn("day_diff_prod_pur_per_cust", F.datediff(transactions.TransactionDt,F.lag(transactions.TransactionDt, 1).over(window))).groupby(["RetailItemNbr"]).agg({"day_diff_prod_pur_per_cust":"mean"}).filter(F.col("avg(day_diff_prod_pur_per_cust)").isNotNull())
combined_rules = spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/combined_rules.parquet").withColumn("RetailItemNbr", F.explode(F.col("consequent")))
join = combined_rules.join(trasactions_part, on='RetailItemNbr', how='inner').select(["consequent","avg(day_diff_prod_pur_per_cust)","lift","confidence"]).orderBy(["avg(day_diff_prod_pur_per_cust)"], ascending = True)


# COMMAND ----------

trasactions_part_alter = transactions.withColumn("day_diff_prod_pur_per_cust", F.datediff(transactions.TransactionDt,F.lag(transactions.TransactionDt, 1).over(window))).groupby(["RetailItemNbr"]).agg({"day_diff_prod_pur_per_cust":"mean"}).filter(F.col("avg(day_diff_prod_pur_per_cust)").isNull())

# COMMAND ----------

join.agg({"avg(day_diff_prod_pur_per_cust)":"mean"}).show()

# COMMAND ----------

from pyspark.sql.functions import col, size, sum, array_except, approx_count_distinct, count, avg, stddev, skewness
#Total unique customers in transaction data - 1058388
transactions = spark.read.format("delta").load("/mnt/input-data/transactions")
cust1 = transactions.groupby(["MemberNbr"]).agg({"TransactionId":"approx_count_distinct"}).count()
# total unique customers after selecting data from mining time window -  579862
master = spark.read.format("parquet").load('/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/input_master_fpg.parquet/')
cust2=master.groupby(["MemberNbr"]).agg({"TransactionId":"approx_count_distinct"}).count()
# total unique customers after selecting data for relevance - 1018
rules = spark.read.format("parquet").load('/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/rules.parquet')
cust3=rules.groupby(["MemberNbr"]).agg({"rule_id":"approx_count_distinct"}).count()
# total unique customers after ALS filter - 185
als = spark.read.parquet('/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/als_user_factors.parquet')
cust4=als.count()

# COMMAND ----------

# 100 days
cust1, cust2, cust3, cust4

# COMMAND ----------

# 300 days
cust1, cust2, cust3, cust4

# COMMAND ----------

# 300 days + lot more rules
cust1, cust2, cust3, cust4

# COMMAND ----------

# 300 days + lot more rules but different rules
cust1, cust2, cust3, cust4

# COMMAND ----------

combined_rules = spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/combined_rules.parquet")

# COMMAND ----------

combined_rules.count()

# COMMAND ----------

from datetime import datetime, timedelta
import pyspark.sql.functions as F
master.filter(F.col("TransactionDt") >= datetime.now() - timedelta(days=150)).groupby(["MemberNbr"]).agg({"TransactionId":"approx_count_distinct"}).count()

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
als_rules = spark.read.parquet('/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/als_item_factors.parquet')
als_rules_col = als_rules.select(['id']+[F.expr('features[' + str(x) + ']') for x in range(0, 20)])
df_als_rules = als_rules_col.toPandas()
X_embedded = PCA(n_components=2).fit_transform(df_als_rules.filter(df_als_rules.columns[1:]))
df_tsne = pd.DataFrame(X_embedded, columns = ["X",'Y'])
sns.scatterplot(data=df_tsne, x="X", y="Y")



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

master.groupby(["MemberNbr","TransactionId"]).agg({"RetailItemNbr":"count"}).count()

# COMMAND ----------

master.count()


# COMMAND ----------

combined_rules = spark.read.format("parquet").load("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/combined_rules.parquet")
combined_rules.count()

# COMMAND ----------

tmp.filter(F.col("RetailItemNbr") == 120506735).orderby(["MemberNbrshow(100)

# COMMAND ----------

b

# COMMAND ----------

# Read the first FPG model output on products and see number of rules
tmp = spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/rules.parquet")
#tmp = tmp.withColumn("monotonically_increasing_id", F.monotonically_increasing_id())  
#tmp.agg({"monotonically_increasing_id": "min"}).show()
#tmp.show(10) 
tmp = tmp.groupBy(["MemberNbr"]).agg({"rule_id":"count"}).orderBy(["count(rule_id)"],ascending = True)
#tmp.orderBy("confidence", ascending=False).show(20)

# COMMAND ----------

tmp.show(100)

# COMMAND ----------

# Read the first FPG model output on categories and see number of rules
tmp = spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/cat_id_rules_fpg.parquet")
#tmp = spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/pr_id_rules_fpg.parquet")
#tmp = tmp.withColumn("monotonically_increasing_id", F.monotonically_increasing_id())  
#tmp.agg({"monotonically_increasing_id": "min"}).show()
#tmp.count()
tmp.orderBy("confidence", ascending=False).show(20)


# COMMAND ----------

# Read the frequency in rule_mining fit
tmp =spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/combined_rules.parquet")
#tmp.groupBy("antecedent_CombinedCategoryID"). agg({"consequent":"count"}).orderBy("count(consequent)", descending = True).show(100)
#tmp.show(10)
tmp.show(1)

# COMMAND ----------

# Read the first FPG model output on categories and see number of rules
tmp = spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/top_products.parquet")
#tmp = tmp.withColumn("monotonically_increasing_id", F.monotonically_increasing_id())  
#tmp.agg({"monotonically_increasing_id": "min"}).show()
#tmp.count()
tmp.groupBy("CombinedCategoryId","rank").agg({"RetailItemNbr":"count"}).orderBy("count(RetailItemNbr)",ascending=False).show(20)
#tmp.show(20)


# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, size, array_union, collect_list
tmp = spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/top_products.parquet")
#tmp.groupBy("RetailItemNbr").agg({"rank": "count"}).show(10)
tmp.groupBy("CombinedCategoryId").agg( \
       collect_list("RetailItemNbr").alias("replacement_items")).withColumn("frequency",F.size(F.col("replacement_items"))).show(10)  
#tmp.agg({"monotonically_increasing_id": "min"}).show()


# COMMAND ----------

tmp.show(10)

# COMMAND ----------

from datetime import datetime, timedelta
import pyspark.sql.functions as F
from pyspark.sql.functions import col, size, sum, array_except, approx_count_distinct, count, avg, stddev, skewness

tmp = spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/input_master_fpg.parquet")

class DynamicDates:
    def __init__(self):
        self.date_array = []

    def add_date_window(self, n_days, end=datetime.now(), lag=None):
        if lag is not None:
            end = end - timedelta(days=lag)
        dates = [(end - timedelta(days=x))
                 for x in range(n_days)]
        self.date_array += dates

    def clear_dates(self):
        self.date_array = []

    def return_dates(self, date_format='%Y-%m-%d') -> [str]:
        return [i.strftime(date_format) for i in self.date_array]
      
transaction_dates = DynamicDates()
transaction_dates.add_date_window(n_days=80, lag=0)
master_tmp =tmp.filter(F.col("TransactionDt").isin(transaction_dates.return_dates())).cache()

rules_tmp =spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/combined_rules.parquet")




# COMMAND ----------

#tmp = spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/rules.parquet")
tmp = spark.read.parquet('/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/als_user_factors.parquet')

# COMMAND ----------

tmp.count()

# COMMAND ----------

rules_tmp =spark.read.parquet("/mnt/input-data/HyperPersonalization/XSellModelCheckpoints/combined_rules.parquet")

rules_tmp.groupby(["antecedent_RetailItemNbr"]).agg({"consequent":"count"}).orderBy(["count(consequent)"], ascending = False).show()


# COMMAND ----------

.withColumn("inter_LHS_product",(size(array_except(col("antecedent_RetailItemNbr"), \ 
                                       col("RetailItemNbr"))) == 0).cast('integer')) \
        .withColumn("inter_LHS_category",(size(array_except(col("antecedent_CombinedCategoryID"), \
                                       col("CombinedCategoryID"))) == 0).cast('integer')) \
        .withColumn("inter_RHS",(size(array_except(col("consequent"), col("RetailItemNbr"))) == 0).cast('integer')) \
        .withColumn("inter_LHS", coalesce(col("inter_LHS_product"), col("inter_LHS_category"))) \
        .withColumn("inter_full", (col("inter_LHS") * col("inter_RHS"))) \
        .filter((col("inter_LHS") == 1) | (col("inter_RHS") == 1)) \
        .groupBy('MemberNbr', 'rule_id') \
        .agg(sum('inter_LHS').alias('ant_count'), sum('inter_RHS').alias('cons_count'),sum('inter_full').alias('full_count')) \
        .withColumn('cons_only', col('cons_count') - col('full_count')) \
        .filter((col('ant_count') > min_lhs_threshold) & (col('cons_count') > min_rhs_threshold))



# COMMAND ----------

