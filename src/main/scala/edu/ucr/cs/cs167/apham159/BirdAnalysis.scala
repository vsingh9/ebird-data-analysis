package edu.ucr.cs.cs167.apham159

import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._

// ML-related imports
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.tuning.TrainValidationSplitModel

/**
 * Combined eBird Data Analysis Project
 * This program integrates all tasks from Project B into a single executable
 */
object BirdAnalysis {
  def main(args: Array[String]): Unit = {
    // Initialize Spark context
    val conf = new SparkConf().setAppName("Bird Data Analysis")

    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")

    val spark: SparkSession.Builder = SparkSession.builder().config(conf)
    val sparkSession: SparkSession = spark.getOrCreate()
    val sparkContext = sparkSession.sparkContext
    SparkSQLRegistration.registerUDT
    SparkSQLRegistration.registerUDF(sparkSession)

    if (args.length < 1) {
      println("Usage: BirdAnalysis <operation> [args...]")
      println("Operations:")
      println("  prepare-data <csv_input>")
      println("  zipcode-ratio <parquet_input> <species_name>")
      println("  time-analysis <parquet_input> <start_date:MM/DD/YYYY> <end_date:MM/DD/YYYY>")
      println("  range-report <parquet_input> <start_date:MM/DD/YYYY> <end_date:MM/DD/YYYY> <x_min> <y_min> <x_max> <y_max>")
      println("  predict-category <parquet_input>")
      System.exit(1)
    }

    val operation: String = args(0)

    try {
      // Import Beast features
      import edu.ucr.cs.bdlab.beast._

      val t1 = System.nanoTime()
      var validOperation = true

      operation match {
        case "prepare-data" =>
          if (args.length < 2) {
            println("Usage: prepare-data <csv_input>")
            System.exit(1)
          }

          val inputFile = args(1)
          prepareData(sparkSession, inputFile)

        case "zipcode-ratio" =>
          if (args.length < 3) {
            println("Usage: zipcode-ratio <parquet_input> <species_name>")
            System.exit(1)
          }

          val parquetFile = args(1)
          val speciesName = args(2)
          zipcodeRatio(sparkSession, parquetFile, speciesName)

        case "time-analysis" =>
          if (args.length < 4) {
            println("Usage: time-analysis <parquet_input> <start_date:MM/DD/YYYY> <end_date:MM/DD/YYYY>")
            System.exit(1)
          }

          val inputPath = args(1)
          val startDate = args(2)
          val endDate = args(3)
          timeAnalysis(sparkSession, inputPath, startDate, endDate)

        case "range-report" =>
          if (args.length < 8) {
            println("Usage: range-report <parquet_input> <start_date:MM/DD/YYYY> <end_date:MM/DD/YYYY> <x_min> <y_min> <x_max> <y_max>")
            System.exit(1)
          }

          val filePath = args(1)
          val startDate = args(2)
          val endDate = args(3)
          val xMin = args(4).toDouble
          val yMin = args(5).toDouble
          val xMax = args(6).toDouble
          val yMax = args(7).toDouble
          rangeReport(sparkSession, filePath, startDate, endDate, xMin, yMin, xMax, yMax)

        case "predict-category" =>
          if (args.length < 2) {
            println("Usage: predict-category <parquet_input>")
            System.exit(1)
          }

          val inputFile = args(1)
          predictCategory(sparkSession, inputFile)

        case _ =>
          validOperation = false
          println(s"Invalid operation: $operation")
          println("Operations: prepare-data, zipcode-ratio, time-analysis, range-report, predict-category")
      }

      val t2 = System.nanoTime()
      if (validOperation)
        println(s"Operation '$operation' took ${(t2 - t1) * 1E-9} seconds")
      else
        Console.err.println(s"Invalid operation '$operation'")

    } finally {
      sparkSession.stop()
    }
  }

  /**
   * Task 1: Data preparation
   * Prepare the data for processing by introducing a ZIPCode attribute and converting to Parquet
   */
  def prepareData(sparkSession: SparkSession, inputFile: String): Unit = {
    // Import Beast features
    import edu.ucr.cs.bdlab.beast._

    println(s"Loading and preparing data from $inputFile")

    // 1) Parse and load the CSV file using the Dataframe API
    val birdDF = sparkSession.read
      .option("sep", ",")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputFile)

    println("CSV data loaded successfully")

    // 2) Introduce a geometry attribute that represents the location of each observation
    val birdDF1 = birdDF.selectExpr("*", "ST_CreatePoint(x, y) AS geometry")

    // 3) Keep only the selected columns
    val filterDF = birdDF1.selectExpr(
      "x",
      "y",
      "`GLOBAL UNIQUE IDENTIFIER`",
      "CATEGORY",
      "`COMMON NAME`",
      "`SCIENTIFIC NAME`",
      "`OBSERVATION COUNT`",
      "`OBSERVATION DATE`",
      "geometry"
    )

    // 4) Rename all attributes that include a space
    val filterDF1 = filterDF
      .withColumnRenamed("GLOBAL UNIQUE IDENTIFIER", "GLOBAL_UNIQUE_IDENTIFIER")
      .withColumnRenamed("COMMON NAME", "COMMON_NAME")
      .withColumnRenamed("SCIENTIFIC NAME", "SCIENTIFIC_NAME")
      .withColumnRenamed("OBSERVATION COUNT", "OBSERVATION_COUNT")
      .withColumnRenamed("OBSERVATION DATE", "OBSERVATION_DATE")

    println("Data transformed with renamed columns")

    // 5) Convert the resulting Dataframe to a SpatialRDD
    val birdRDD: SpatialRDD = filterDF1.toSpatialRDD

    // 6) Load the ZIP Code dataset using Beast
    println("Loading ZIP code dataset")
    val zipDF = sparkSession.read.format("shapefile").load("tl_2018_us_zcta510.zip")
    val zipRDD: SpatialRDD = zipDF.toSpatialRDD

    // 7) Run a spatial join query to find the ZIP code of each observation
    println("Running spatial join to find ZIP codes")
    val zipBird: RDD[(IFeature, IFeature)] = birdRDD.spatialJoin(zipRDD)

    // 8 & 9) Use ZCTA5CE10 attribute to introduce ZIPCode and convert to Dataframe
    val zipCodeBird: DataFrame = zipBird.map({
      case (bird, zip) => Feature.append(bird, zip.getAs[String]("ZCTA5CE10"), "ZIPCode")
    }).toDataFrame(sparkSession)

    // 10) Drop the geometry column
    val finalDF = zipCodeBird.drop("geometry")

    // 11) Write the output as a Parquet file
    println("Writing output as Parquet file to eBird_ZIP")
    finalDF.write.mode(SaveMode.Overwrite).parquet("eBird_ZIP")

    println("Data preparation completed successfully")
  }

  /**
   * Task 2: Spatial analysis
   * Count the percentage of observations for a specific bird species among all observations per ZIP Code
   */
  def zipcodeRatio(sparkSession: SparkSession, parquetFile: String, speciesName: String): Unit = {
    // Import Beast features
    import edu.ucr.cs.bdlab.beast._

    println(s"Loading data from $parquetFile")
    // 1. Load the Parquet dataset
    val birdDF = sparkSession.read.parquet(parquetFile)
    // Register the DataFrame as a temporary view to run SQL queries
    birdDF.createOrReplaceTempView("birds")
    println("Bird data loaded successfully")

    // 2. Run a grouped-aggregate SQL query that computes the total number of observations per ZIP code
    println("Computing total observations per ZIP code")
    val totalObservationsQuery = """
      SELECT ZIPCode, SUM(
        CASE
          WHEN OBSERVATION_COUNT = 'X' THEN 1
          WHEN OBSERVATION_COUNT IS NULL THEN 0
          ELSE CAST(OBSERVATION_COUNT AS INT)
        END
      ) as total_observations
      FROM birds
      WHERE ZIPCode IS NOT NULL
      GROUP BY ZIPCode
    """
    val totalObservationsDF = sparkSession.sql(totalObservationsQuery)
    totalObservationsDF.createOrReplaceTempView("total_observations")
    println("Total observations calculated")

    // 3. Run a second grouped-aggregate SQL query for the given species
    println(s"Computing observations for species: $speciesName")
    val speciesObservationsQuery = s"""
      SELECT ZIPCode, SUM(
        CASE
          WHEN OBSERVATION_COUNT = 'X' THEN 1
          WHEN OBSERVATION_COUNT IS NULL THEN 0
          ELSE CAST(OBSERVATION_COUNT AS INT)
        END
      ) as species_observations
      FROM birds
      WHERE COMMON_NAME = '$speciesName' AND ZIPCode IS NOT NULL
      GROUP BY ZIPCode
    """
    val speciesObservationsDF = sparkSession.sql(speciesObservationsQuery)
    speciesObservationsDF.createOrReplaceTempView("species_observations")
    println("Species observations calculated")

    // 4. Join the results of the two queries by ZIP Code and compute the ratio
    println("Computing ratio of species to total observations")
    val ratioQuery = """
      SELECT t.ZIPCode,
             CASE WHEN t.total_observations > 0
                  THEN COALESCE(s.species_observations, 0) / t.total_observations
                  ELSE 0
             END as ratio
      FROM total_observations t
      LEFT JOIN species_observations s ON t.ZIPCode = s.ZIPCode
    """
    val ratioDF = sparkSession.sql(ratioQuery)
    ratioDF.createOrReplaceTempView("zip_ratios")
    println("Ratio calculation complete")

    // 5. Load the ZIP Code dataset for geometry
    println("Loading ZIP code boundaries from tl_2018_us_zcta510.zip")
    val zipDF = sparkSession.read.format("shapefile").load("tl_2018_us_zcta510.zip")
    zipDF.createOrReplaceTempView("zip_geom")
    println("ZIP code data loaded")

    // 6. Join with the ZIP Code dataset to add geometry
    println("Joining ZIP ratios with geometries")
    val finalQuery = """
      SELECT r.ZIPCode, z.geometry, r.ratio
      FROM zip_ratios r
      JOIN zip_geom z ON r.ZIPCode = z.ZCTA5CE10
    """
    val resultDF = sparkSession.sql(finalQuery)
    println("Join complete")

    // 7. Ensure a single file is written to the output
    println("Preparing output file")
    val singleFileDF = resultDF.coalesce(1)

    // 8. Store the output as a Shapefile
    println("Writing output shapefile")
    singleFileDF.write
      .format("shapefile")
      .mode(SaveMode.Overwrite)
      .save("eBirdZIPCodeRatio")

    println(s"Analysis completed for species: $speciesName")
    println("Output saved as 'eBirdZIPCodeRatio'")
  }

  /**
   * Task 3: Temporal analysis
   * Find the number of observations of each species in a given date range
   */
  def timeAnalysis(sparkSession: SparkSession, inputPath: String, startDate: String, endDate: String): Unit = {
    println(s"Running temporal analysis on $inputPath from $startDate to $endDate")

    // Load Parquet file
    val df = sparkSession.read.parquet(inputPath)

    // Register DataFrame as a temp view for SQL
    df.createOrReplaceTempView("eBird")

    // Run SQL query to filter by date range and group by species
    val filteredDF = sparkSession.sql(
      s"""
         |SELECT
         |   COMMON_NAME,
         |   SUM(
         |     CASE
         |       WHEN OBSERVATION_COUNT = 'X' THEN 1
         |       WHEN OBSERVATION_COUNT IS NULL THEN 0
         |       ELSE CAST(OBSERVATION_COUNT AS INT)
         |     END
         |   ) AS num_observations
         |FROM eBird
         |WHERE to_date(OBSERVATION_DATE, 'yyyy-MM-dd')
         |      BETWEEN to_date('$startDate', 'MM/dd/yyyy')
         |          AND to_date('$endDate', 'MM/dd/yyyy')
         |GROUP BY COMMON_NAME
         |ORDER BY num_observations DESC
         """.stripMargin)

    // Write to a single CSV file
    println("Writing results to eBirdObservationsTime")
    filteredDF.coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("eBirdObservationsTime")

    println(s"Temporal analysis complete. Results saved to eBirdObservationsTime")
  }

  /**
   * Task 4: Spatio-Temporal Analysis
   * Show all locations of observed birds within a given time and region
   */
  def rangeReport(sparkSession: SparkSession, filePath: String, startDate: String, endDate: String,
                  xMin: Double, yMin: Double, xMax: Double, yMax: Double): Unit = {
    println(s"Running spatio-temporal analysis on $filePath")
    println(s"Date range: $startDate to $endDate")
    println(s"Spatial bounds: ($xMin, $yMin) to ($xMax, $yMax)")

    // Load Parquet file
    val df = sparkSession.read.parquet(filePath)

    // Print schema for reference
    println("=== Parquet Schema ===")
    df.printSchema()

    // Convert observation date to proper date type
    val dfWithDate = df.withColumn("ObservationDate", to_date(col("OBSERVATION_DATE"), "yyyy-MM-dd"))

    // Show overall stats
    println("=== Overall Stats ===")
    dfWithDate.select(
      min("x").alias("min_x"),
      max("x").alias("max_x"),
      min("y").alias("min_y"),
      max("y").alias("max_y"),
      min("ObservationDate").alias("min_date"),
      max("ObservationDate").alias("max_date")
    ).show(false)

    // Total row count before filtering
    val totalCount = dfWithDate.count()
    println(s"Total rows in the dataset: $totalCount")

    // Parse user-provided dates
    val startDateCol = to_date(lit(startDate), "MM/dd/yyyy")
    val endDateCol = to_date(lit(endDate), "MM/dd/yyyy")

    println(s"Parsed start date: $startDate")
    println(s"Parsed end date: $endDate")

    // Apply filters
    val filteredDF = dfWithDate
      .filter(col("ObservationDate").between(startDateCol, endDateCol))
      .filter(col("x").between(xMin, xMax))
      .filter(col("y").between(yMin, yMax))

    // Count after filtering
    val filteredCount = filteredDF.count()
    println(s"Number of rows after filtering: $filteredCount")

    if (filteredCount == 0) {
      println("No rows matched your filters. Possible reasons:")
      println("1) The date format in the Parquet file might not match 'yyyy-MM-dd'.")
      println("2) The bounding box for x, y doesn't include any points.")
      println("3) The user-supplied date range (MM/dd/yyyy) doesn't match the actual data range.")
    } else {
      // Select relevant columns
      val resultDF = filteredDF.select("x", "y", "GLOBAL_UNIQUE_IDENTIFIER", "ObservationDate")

      // Save results to CSV
      println("Saving filtered results")
      val outputFilePath = "BirdRangeReport.csv"
      resultDF.coalesce(1)
        .write
        .mode(SaveMode.Overwrite)
        .option("header", "true")
        .csv(outputFilePath)

      println(s"Filtered results saved to: $outputFilePath")
    }
  }

  /**
   * Task 5: Bird's Category Prediction
   * Predict the CATEGORY of a bird sighting using features from its names
   */
  def predictCategory(sparkSession: SparkSession, inputFile: String): Unit = {
    println(s"Running category prediction on $inputFile")

    // Load Parquet file
    val birdData = sparkSession.read.parquet(inputFile)

    // Filter dataset for specified categories
    birdData.createOrReplaceTempView("birds_table")

    // Run Spark SQL query to filter CATEGORY values
    val filteredData = sparkSession.sql(
      """SELECT COMMON_NAME, SCIENTIFIC_NAME, CATEGORY
        |FROM birds_table
        |WHERE CATEGORY IN ('species', 'form', 'issf', 'slash')
      """.stripMargin)

    println("Sample data:")
    filteredData.show(5, truncate = false)

    // Tokenizer: Split names into words
    val tokenizerCommon = new Tokenizer().setInputCol("COMMON_NAME").setOutputCol("common_words")
    val tokenizerScientific = new Tokenizer().setInputCol("SCIENTIFIC_NAME").setOutputCol("scientific_words")

    // HashingTF: Convert words into numerical features
    val hashingTFCommon = new HashingTF()
      .setInputCol("common_words")
      .setOutputCol("common_features")
      .setNumFeatures(1024)

    val hashingTFScientific = new HashingTF()
      .setInputCol("scientific_words")
      .setOutputCol("scientific_features")
      .setNumFeatures(1024)

    // Assemble features
    val assembler = new VectorAssembler()
      .setInputCols(Array("common_features", "scientific_features"))
      .setOutputCol("features")

    // StringIndexer: Convert CATEGORY to numerical labels
    val indexer = new StringIndexer()
      .setInputCol("CATEGORY")
      .setOutputCol("label")
      .setHandleInvalid("skip")

    // Logistic Regression Classifier
    val classifier = new LogisticRegression()

    // ML Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizerCommon, tokenizerScientific,
        hashingTFCommon, hashingTFScientific,
        assembler, indexer, classifier))

    // Hyperparameter tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.regParam, Array(0.01, 0.001))
      .addGrid(classifier.maxIter, Array(10, 20))
      .build()

    // Train-Validation Split
    val cv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
      .setParallelism(2)

    // Train-test split
    val Array(trainingData, testData) = filteredData.randomSplit(Array(0.8, 0.2))

    println("Training model...")
    val t1 = System.nanoTime()

    // Train model
    val model: TrainValidationSplitModel = cv.fit(trainingData)

    val t2 = System.nanoTime()
    println(s"Training time: ${(t2 - t1) * 1E-9} seconds")

    // Get best model parameters
    val bestModel = model.bestModel.asInstanceOf[PipelineModel]
    val regParam: Double = bestModel.stages.last.asInstanceOf[LogisticRegressionModel].getRegParam
    val maxIter: Int = bestModel.stages.last.asInstanceOf[LogisticRegressionModel].getMaxIter

    println(s"Best Model Parameters:")
    println(s"regParam: $regParam")
    println(s"maxIter: $maxIter")

    // Predictions
    val predictions: DataFrame = model.transform(testData)

    println("Sample predictions:")
    predictions.select("COMMON_NAME", "SCIENTIFIC_NAME", "CATEGORY", "label", "prediction")
      .show(10, truncate = false)

    // Evaluate Model
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)

    println("Model Evaluation:")
    println(s"Test Accuracy: $accuracy")
    println(s"Test Precision: $precision")
    println(s"Test Recall: $recall")
  }
}