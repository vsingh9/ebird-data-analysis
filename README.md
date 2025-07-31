# eBird Data Analysis
The goal of this project was to develop a comprehensive Apache Spark application in Scala to analyze a dataset of 100k+ bird observations within a subset of California from the Cornell Lab of Ornithology. A big-data processing pipeline was assembled to achieve this.

* Built end-to-end ETL pipeline converting CSV data to optimized Parquet format, reducing file size by 96% and enabling faster query performance for geospatial analysis

* Implemented spatial and temporal analytics using Beast geospatial library to calculate species distribution ratios over specific ZIP code boundaries and/or time ranges, generating choropleth visualizations using QGIS for ecological pattern analysis

* Created ML pipeline using Spark MLlib with feature engineering (tokenization, hashing, assembler, indexer), hyperparameter tuning, and logistic regression to predict bird observation categories with 94% accuracy

* Designed modular command-line interface supporting 5 distinct operations (data preparation, spatial analysis, temporal analysis, spatio-temporal filtering, ML prediction) with comprehensive error handling and performance monitoring
