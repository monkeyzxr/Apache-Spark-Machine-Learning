/**
  * Created by monkeyzxr on 2017/4/5.
  */
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession

object QuestionOne_New {
  //case class must be outside the main!!!
  case class MovieClusterClass(cluster_number: Int, movie_id: String)
  case class MovieClass(movie_id: String, movie_title: String, genre: String)

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("QuestionOne_New")
      .master("local")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    val itemUserMatFilePath = "file:///Users/monkeyzxr/Desktop/CS 6350.001 - Big Data Management and Analytics" +
      "/Assignment/Homework-3/hw3datasetnew/itemusermat"
    val moviesDatFilePath = "file:///Users/monkeyzxr/Desktop/CS 6350.001 - Big Data Management and Analytics" +
      "/Assignment/Homework-3/hw3datasetnew/movies.dat"

    //get item-user RDD
    val item_user_line = spark.sparkContext.textFile(itemUserMatFilePath)

    // training data only use the rating, not use the first column
    val trainingData = item_user_line.map(line => Vectors.dense(line.split(" ", 2)(1).split(" ").map(_.toDouble))).cache()

    // build the kMeans model
    val trainModel = KMeans.train(trainingData, 10, 25) // trainingData, numCluster, numIterations

    // predict the cluster, get the key-value pairs
    val movie_cluster = item_user_line.map(line => (trainModel.predict(Vectors.dense(line.split(" ", 2)(1).split(" ").map(_.toDouble))), line.split(" ", 2)(0)))
    // movie_cluster.collect().foreach(println)
    //(9, 342)   tuple

    //make the cluster result into data-frame
    val movie_cluster_DF = movie_cluster.map(tuple => MovieClusterClass(tuple._1.toInt, tuple._2)).toDF()

    //create movie rdd from text file and convert it to data-frame
    val movie_DF = spark.sparkContext
      .textFile(moviesDatFilePath)
      .map(_.split("::"))
      .map(attribute => MovieClass(attribute(0), attribute(1), attribute(2)))
      .toDF()


    /*Following is: join 2 data-frame table, with duplicated column(movie_id),and cluster = 0*/
   //   val joinData = movie_cluster_DF.join(movie_DF, movie_cluster_DF("movie_id") === movie_DF("movie_id"))
   //                                  .orderBy(movie_cluster_DF("cluster_number")) //orderBy is useless!!
   //                                  .filter("cluster_number = 0")
   //                                  .limit(5)
   //                                  .show(false)

/*
//Following is: union 2 result tables, with 5 rows each
    val unionData = joinData.filter(movie_cluster_DF("cluster_number") === 0).limit(5)
      .union(joinData.filter(movie_cluster_DF("cluster_number") === 1).limit(5))
      .union(joinData.filter(movie_cluster_DF("cluster_number") === 2).limit(5))
      .show(false)

*/
    // This a how to join without duplicated column of movie_id
    // specify the columns as an array type or string to avoid having this problem
    val joinData = movie_cluster_DF.join(movie_DF, "movie_id")
    //joinData.show(false)
    /*
    Results looks like:
    +--------+--------------+-----------------------------------------------------------+-----------------------+
    |movie_id|cluster_number|movie_title                                                |genre                  |
    +--------+--------------+-----------------------------------------------------------+-----------------------+
    |1090    |8             |Platoon (1986)                                             |Drama|War              |
    |1436    |0             |Falling in Love Again (1980)                               |Comedy                 |
    |296     |4             |Pulp Fiction (1994)                                        |Crime|Drama            |
    */


    // union the 10 cluster results together, with 5 rows each cluster
    var unionData = joinData.filter(movie_cluster_DF("cluster_number") === 0).limit(5)
    var i = 1
    while(i < 10){
      unionData = unionData.union(joinData.filter(movie_cluster_DF("cluster_number") === i).limit(5))
      i = i + 1
    }

    val num = unionData.count().toInt
    unionData.select("cluster_number","movie_id","movie_title","genre").show(num, false)

  }

}
