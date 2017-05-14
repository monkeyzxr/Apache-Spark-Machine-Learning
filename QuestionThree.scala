/**
  * Created by monkeyzxr on 2017/4/7.
  */
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.{SparkConf, SparkContext}

object QuestionThree {
  def main(args: Array[String]): Unit = {
    val ratingDataInput = "file:///Users/monkeyzxr/Desktop/CS 6350.001 - Big Data Management and Analytics" +
                          "/Assignment/Homework-3/hw3datasetnew/ratings.dat"

    val conf = new SparkConf().setAppName("QuestionThree").setMaster("local")
    val sc = new SparkContext(conf)

    //load data
    val data = sc.textFile(ratingDataInput)

    //parse data
    val ratings = data.map(line => Array(line.split("::")(0), line.split("::")(1), line.split("::")(2))
                           match {case Array(user, item, rate) => Rating(user.toInt, item.toInt,rate.toDouble)})

    val splitData = ratings.randomSplit(Array(0.6, 0.4))
    val (trainingData, testData) = (splitData(0), splitData(1))

    // Build the recommendation model using ALS on training data
    val rank = 10
    val numIterations = 10
    val model = ALS.train(trainingData, rank, numIterations, 0.01)

    // Evaluate the model on test data
    val usersProducts = testData.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }
    val ratesAndPreds = testData.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println("Mean Squared Error = " + MSE)


  }

}
