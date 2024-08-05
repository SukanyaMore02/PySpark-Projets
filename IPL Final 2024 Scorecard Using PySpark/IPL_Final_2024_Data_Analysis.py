"""
Project Description : In this project, we will build the scorecard for IPL final 2024, KKR vs SRH match by analysing deliveries.csv dataset. This dataset contains deliveries deatils for each and every match and inning. I have built the scorecard for first inning of IPL 2024 match for both bowlers and batters.
"""

#Install pyspark & initialse the same.
!pip install pyspark findspark

import findspark
findspark.init()

#Import SparkSession and start the session. 
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("IPLFinal2024").getOrCreate()

#To read the large file use files.upload() in Google Colab Notebook.
from google.colab import files
uploaded = files.upload()

#Create the dataframe by reading deliveries dataset.
deliveries_df = spark.read.option("header", "true").csv("deliveries.csv")

# Shows only top 20 rows
deliveries_df.show()  

# Set truncate=False, if not set it will truncate strings more than 20 chars eg. Royal Challegers ...
deliveries_df.show(deliveries_df.count(),truncate=False) 

#Steps for building Scorecard
#Step1 : Let's find final match ID

import pyspark.sql.functions as func

# The column match id is read as STRING by default as we have not defined schema, so it was not giving correct final match ID.
deliveries_df.select("match_id").distinct().orderBy(func.col("match_id").desc()).show()
deliveries_df.printSchema()

#Let's define schema now
from pyspark.sql.types import StringType, StructField
from pyspark.sql.types import *

fields = StructType([ StructField("match_id", IntegerType(),nullable=True) ,  StructField("inning", IntegerType(),nullable=True), 
                      StructField("batting_team", StringType(),nullable=True),StructField("bowling_team", StringType(),nullable=True),
                      StructField("over", IntegerType(),nullable=True),       StructField("ball", IntegerType(),nullable=True), 
                      StructField("batter", StringType(),nullable=True),      StructField("bowler", StringType(),nullable=True),
                      StructField("non_striker", IntegerType(),nullable=True), StructField("batsman_runs", IntegerType(),nullable=True),
                      StructField("extra_runs", IntegerType(),nullable=True), StructField("total_runs", IntegerType(),nullable=True), 
                      StructField("extras_type", StringType(),nullable=True), StructField("is_wicket", IntegerType(),nullable=True),
                      StructField("player_dismissed", StringType(),nullable=True),StructField("dismissal_kind", StringType(),nullable=True),
                      StructField("fielder", StringType(),nullable=True)
                  ])

fields

#Let's create dataframe with this schema now
deliveries_df = spark.read.schema(fields).option("header",True).csv("deliveries.csv")
deliveries_df.printSchema()


#Let's find final match ID now
deliveries_df.select("match_id").distinct().orderBy(func.col("match_id").desc()).show()

#Filter data only for final match id 1426312
import pyspark.sql.functions as func
ipl_final_df = deliveries_df.filter(func.col("match_id") == 1426312)

ipl_final_df.show(ipl_final_df.count(),truncate=False)

#Stepv 2: Let's build the batting scorecard for first inning.
first_inning_batting = ipl_final_df.filter( func.col("inning") == 1 )

first_inning_batting.show(first_inning_batting.count(), truncate = False)

#a. Find the runs scored by each batsman
batsman_run_df = first_inning_batting.filter(func.col("extras_type").isNull())\
		.groupBy(func.col("batter")).agg(	func.sum("batsman_runs").alias("R")	)

batsman_run_df.show()

#b. Find the balls faced by each batsman
ball_faced_df = first_inning_batting.groupBy(func.col("batter"))\
		.agg( func.count("ball").alias("balls_faced") )
ball_faced_df.show()

#c. Find out number of bounderies hitted by each batsman
first_inning_batting.groupBy(func.col("batter"))\
			.agg( func.count( func.when( func.col("batsman_runs") == 4 , 1 ) ).alias("4s")\
			).show()

#d. Let's build the final scorecard now
Scorecard_df = first_inning_batting.filter(func.col("extras_type").isNull() )/
	  .groupBy(func.col("batter")).agg(func.sum("batsman_runs").alias("R"),
          func.count("ball").alias("balls"),   
          func.count( func.when( func.col("batsman_runs") == 4 , 1 ) ).alias("4s"),
          func.count( func.when( func.col("batsman_runs") == 6 , 1 ) ).alias("6s"),
          func.round(func.sum(func.col("batsman_runs"))*100 / func.count("ball") , 2).alias("S/R")
        )

#e. Let's find batsman order
batsman_order = first_inning_batting.withColumn("over-ball", ( func.concat( func.col("over"),func.lit("."),func.col("ball") ) ).cast('Float'))
				   .groupBy("batter").agg(
							  func.min("over-ball").alias("order")
							 )\
				    .orderBy("order") 

batsman_order.show()

batting_order_df = batsman_order.withColumn("batting_order" , func.row_number().over(Window.orderBy("order"))  )
batting_order_df.show()

#.f show final scorecard
Scorecard_df.join(batting_order_df,  ["batter", "batter"] , how = "inner").\
		  select(["batter","R","B","4s","6s","S/R","batting_order"]\
		 )
		.show()

#Step 2 : Let's build bowling scorecard now
#a. Find the no. of balls bowled, runs given , maiden over and economy
first_inning_bowling_df = ipl_final_df.groupBy("bowler") \
.agg( 
    
    func.count( func.when(func.coalesce(func.col("extras_type"), func.lit("XYZ")) != "wides" ,1)).alias("balls_bowled"), # find over using this /6 & this
    # logic func.concat(func.floor(func.count("over")/6 ), func.lit('.') ,func.count("over")%6  ) .alias("O") ,
    func.sum(func.when(func.col("extras_type").isNull(), func.col("batsman_runs"))  # add wide to this
                 .when(func.col("extras_type") == "wides", func.col("extra_runs"))
           ).alias("R"),
    func.sum(func.when(func.col("is_wicket") == 1, 1)).alias("W")
    )
first_inning_bowling_df.show()

#b.
first_inning_scorecard_df = \
first_inning_bowling_df.select( "bowler",
    				func.concat(func.floor(func.col("balls_bowled")/6 ), func.lit('.') ,func.col("balls_bowled")%6  ).alias("O"), \
                           	func.col("R"),\
                           	"W",
                            	func.round( (func.col("R") / func.col("O")) , 2). alias("Econ")
                    	   )
first_inning_scorecard_df.show()

#c.Find the maiden over
maiden_over_df = first_inning_batting.groupBy("bowler","over")\	
		.agg(func.sum(func.col("total_runs")).alias("runs"),\
                     func.count(func.col("over")).alias("balls")
		    )

maiden_over_df = maiden_over_df.filter((func.col("runs") == 0) & (func.col("balls") ==6 ))\
                                .groupBy("bowler").agg(func.count("bowler").alias("M"))

maiden_over_df.show()

# show the final bowler's scorecard
first_inning_final_df = first_inning_scorecard_df.join(maiden_over_df, on = ["bowler","bowler"], how="left").fillna(value=0)

first_inning_final_df.show()

