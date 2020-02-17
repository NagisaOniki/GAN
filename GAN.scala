//report5
//GAN
package report5
import breeze.linalg._
import CLASS._
object GAN{
  val ln = 1 // 学習回数 ★
  val dn = 50 // 学習データ数 ★
  val rand = new scala.util.Random(0)
  //////////////////save///////////////////////////////
  def save(fn:String,input:Array[Double]){
    val save = new java.io.PrintWriter(fn)
    for(i<-0 until ln){
      save.print(input(i) + ",")
    }
    save.close
  }
  ////////////////////main//////////////////////////////
  def main(args:Array[String]){
    //-------データの読み込み----------
    val (dtrain,dtest) = Image.load_mnist("/home/share/fashion-mnist")
    //----------画像データをまとめる--------
    var ds = (0 until 100).map(i=>dtrain(i)._1).toArray
    //Image.write(f"GAN-original.png" , Image.make_image(ds,10,10))
    //---------ネットワーク構成-----------------
    val N = new network()
    val LayerG = List( //original
      new Affine(100,256),
      new LeakyReLU(0.2),
      new BN(256),
      new Affine(256,512),
      new LeakyReLU(0.2),
      new BN(512),
      new Affine(512,1024),
      new LeakyReLU(0.2),
      new BN(1024),
      new Affine(1024,784),
      new Tanh()
    )

    /**
    val LayerG = List( //loseBN(pattern3)
      new Affine(100,256),
      new LeakyReLU(0.2),
      new Affine(256,512),
      new LeakyReLU(0.2),
      new Affine(512,1024),
      new LeakyReLU(0.2),
      new Affine(1024,784),
      new Tanh()
    )

    */

    val LayerD = List( //original
      new Affine(784,512),
      new ReLU(),
      new Affine(512,256),
      new ReLU(),
      new Affine(256,1),
      new Sigmoid()
    )
/*
    val LayerD = List( //addLN(pattrn2)
      new Affine(784,512),
      new ReLU(),
      new LN(512),
      new Affine(512,256),
      new ReLU(),
      new LN(256),
      new Affine(256,1),
      new Sigmoid()
    )*/
    var correctDx = new Array[Double](ln)
    var correctDz = new Array[Double](ln)
    var lossG = new Array[Double](ln)
    var lossDx = new Array[Double](ln)
    var lossDz = new Array[Double](ln)
    for(i<-0 until ln){ //学習
      var cDx = 0d
      var cDz = 0d
      val x = (rand.shuffle(dtrain.toList).take(dn)).map(_._1).toArray.map(y=>y.map(a=>a*2d-1d)) //本物シャッフル
      //val x = dtrain.take(dn).map(_._1).toArray //本物
      val z = Array.ofDim[Double](100,100).map(y => y.map(_ => rand.nextGaussian)) //偽物 //データ数,次元数(Affineと対応)
      //----------G----------------
      //----forward----
      val yG = N.forwards(LayerG,z)
      val yD = N.forwards(LayerD,yG)
      lossG(i) = yD.map(y=>y.map(a=>math.log(1d-a.toDouble+2e-8)).sum).sum
      //----backward----
      val dLG = yD.map(y => y.map(a => -1d/a ))
      val dD = N.backwards(LayerD.reverse,dLG)
      N.backwards(LayerG.reverse,dD)
      N.updates(LayerG)
      N.resets(LayerD)
      //----------D1---------------
      //----forward----
      val yDx = N.forwards(LayerD,x)
      lossDx(i) = yDx.map(y=>y.map(a=>math.log(a.toDouble+2e-8)).sum).sum
      yDx.map{y =>
        if(y(0) >= 0.5){
          cDx += 1d
        }
      }
      //----backward----
      val dLDx = yDx.map(y => y.map(a => -1d/a ))
      N.backwards(LayerD.reverse,dLDx)
      N.updates(LayerD)
      //----------D2---------------
      //----forward----
      val yDz = N.forwards(LayerD,yG.take(dn))
      lossDz(i) = yDz.map(y=>y.map(a=>math.log(1d-a.toDouble+2e-8)*(-1d)).sum).sum
      yDz.map{y =>
        if(y(0) < 0.5){
          cDz += 1d
        }
      }
      //----backward----
      val dLDz = yDz.map(y => y.map(a => 1d/(1d-a) ))
      N.backwards(LayerD.reverse,dLDz)
      N.updates(LayerD)
      N.resets(LayerG)
      N.resets(LayerD)
      //-----入力画像更新--------
      ds = yG
      //------------画像作成--------------------
      if(i==50||i==100||i==200||i==300||i==400||i%500==0||i==ln-1){
        //Image.write(f"GAN1-" + i + ".png" , Image.make_image(ds.map(y=>y.map(a=>(a+1d)/2)),10,10))
      }
      println("1 : " + i + " : 本物正解率:" +  cDx/dn*100 + "  偽物正解率:" + cDz/dn*100)
      correctDx(i) = cDx/dn*100
      correctDz(i) = cDz/dn*100
    }//ln

    for(i<-0 until LayerG.size){
      LayerG(i).save("exGANsaveG"+i)
    }
    for(i<-0 until LayerD.size){
      LayerD(i).save("exGANsaveD"+i)
    }

    //save("saveCorrectDx3",correctDx)
    //save("saveCorrectDz3",correctDz)
    //save("saveLossG3",lossG)
    //save("saveLossDx3",lossDx)
    //save("saveLossDz3",lossDz)
  }//main
}//GAN
