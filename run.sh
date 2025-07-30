#!/usr/bin/env sh
mvn clean package

JAVA_OPTS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

spark-submit --driver-java-options "$JAVA_OPTS" --class edu.ucr.cs.cs167.apham159.BirdAnalysis target/apham159_projectB-1.0-SNAPSHOT.jar prepare-data eBird_1k.csv

spark-submit --driver-java-options "$JAVA_OPTS" --class edu.ucr.cs.cs167.apham159.BirdAnalysis target/apham159_projectB-1.0-SNAPSHOT.jar zipcode-ratio eBird_ZIP "Mallard"

spark-submit --driver-java-options "$JAVA_OPTS" --class edu.ucr.cs.cs167.apham159.BirdAnalysis target/apham159_projectB-1.0-SNAPSHOT.jar time-analysis eBird_ZIP 02/21/2015 05/22/2015

spark-submit --driver-java-options "$JAVA_OPTS" --class edu.ucr.cs.cs167.apham159.BirdAnalysis target/apham159_projectB-1.0-SNAPSHOT.jar range-report eBird_ZIP 02/21/2015 05/22/2015 -120 34 -118 36

spark-submit --driver-java-options "$JAVA_OPTS" --class edu.ucr.cs.cs167.apham159.BirdAnalysis target/apham159_projectB-1.0-SNAPSHOT.jar predict-category eBird_ZIP
