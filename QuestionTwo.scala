/**
  * Created by monkeyzxr on 2017/4/6.
  */
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}

import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object QuestionTwo {
  def main(args: Array[String]): Unit = {
    val glassDataFile = "file:///Users/monkeyzxr/Desktop/CS 6350.001 - Big Data Management and Analytics" +
                        "/Assignment/Homework-3/hw3datasetnew/glass.data"
    val conf = new SparkConf().setAppName("QuestionTwo").setMaster("local")
    val sc = new SparkContext(conf)

    //load glass data RDD
    val glassRDD = sc.textFile(glassDataFile)

    // convert each line rdd into labeled point.
    // A labeled point is a local vector, either dense or sparse, associated with a label/response
    // new LabeledPoint(label: Double, features: Vector)
    // the first column is id_number, useless; the last column is label, not used in feature vector
    //The substring begins at the specified beginIndex and extends to the character at index endIndex - 1.
    val labeledData = glassRDD
      .map(line => LabeledPoint(line.split(",")(10).toDouble, Vectors.dense(line.split(",",2)(1).substring(0, line.split(",",2)(1).length-2).split(",").map(_.toDouble))))

    // labeledData.collect().foreach(println)
    //(1.0,[1.52101,13.64,4.49,1.1,71.78,0.06,8.75,0.0,0.0])
    // (1.0,[1.51761,13.89,3.6,1.36,72.73,0.48,7.83,0.0,0.0])

    //split the labeled data into training and test data
    val splitedData = labeledData.randomSplit(Array(0.6, 0.4))
    val (trainingData, testData) = (splitedData(0), splitedData(1))


    ///////////////////////////////////////////////////////////////////////////////////
    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 8 //labels should be within the range [0,numClasses-1], so, numClass = labels + 1
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model_DT = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model_DT.predict(point.features)
      (point.label, prediction)
    }
    //labelAndPreds.collect().foreach(println) // get the label value and the predicted labeled value
    //(1.0,1.0)
    //(1.0,2.0)

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
   // println("Test Error of Decision Tree model = " + testErr)
  //  println("Learned classification tree model:\n" + model_DT.toDebugString) //A description of this RDD

    ///////////////////////////////////////////////////////////////////////////////////
    // Train a Naive Bayes model.
    val model_NB = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = testData.map(p => (model_NB.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testData.count()
    ////////////////////////////////////////////////////////////////////////////////////////
    println("Accuracy of the Decision Tree model = " + (1.0 - testErr))
    println("Accuracy of the Naive Bayes model = " + accuracy)

  }
}
