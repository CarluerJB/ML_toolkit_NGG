<snippet>
  <content>
  
# ML_toolkit_NGG

ML_toolkit_NGG is a set of tool based on scikit and keras.
	  
This code has been designed by CARLUER Jean-Baptiste for research purpose : "Link to the paper"

## Installation

You will need at least conda installed on your computer. 
1. Go in main directory and type 
  
    conda create --name myenv --file spec-file.txt
    
2. Once all package are installed activate your environment
  
    conda activate myenv
    
3. You are all set to go !

## Usage

This algorithm is design to predict a phenotype Y as itself or as class using a X (a genotype or a phenotype or a combinaison)

To run the test study : 

    ./run.sh

To run your own study : 

    python main.py [args]
    
List of mains args : 

	-y_data_path : the path to the phenotype dataset, it should contain at least two column (One for the ID)
	-x_data_path: the path to the genotype dataset
  -ID_data_path : the path to the genotype ID list (the intersection with y_data_path will be kept)
  -nth_elem_1D_path : the path to the nth_element file to select which 1D SNP to keep
	-nth_elem_2D_path: the path to the nth_element file to select which 2D SNP to keep
  -save_path : the path to save the results
  -json_model : the path to the DNN models (see DNN model format for more informations)
	-data_mapper_path : the path to the data_mapper, the data mapper is important to specify X, Y file delimiter/header and to indicate combination methods
  -p : the phenotype to predict (must be one of the phenotype in the phenotype dataset)
  -1D : the number of Ktop to keep from the nth_elem_1D file
	-2D : the number of Ktop to keep from the nth_elem_2D file
  -m : the DNN model id (must be one of the model in the json_model file)
  -random_2D : to specify if you want to randomly generate nth_elem_2D ID's
  -dataset : the dataset to use default is ionome
  -method : the method to use (see the method list)
  -categorize : the method to use to phenotype to class transformation (by_quantile or by_kmeans)
  -nb_cluster : the number of class for phenotype to class transformation
  
  ## DNN model format
  
  To keep a track of your DNN models, we use a json format. In it are specify the layers (with ID) and the inputs to each layers. 
  The name of the model is also important because it will be the key to the -m option if your json DNN model file contain many models.
  The first layer wich will of the size of your data, need to be set to null, the algorithm will set it automatically according to your datas.
  
  The main format is as follow for a four layer models : 
  
    {
      "Name of the model" : {
        "inputs" : {
          "0" : null
        },
        "layers" : {
          "1" : [
            {"inputs" : {"layer" : 0, "node" : 0}, "type" : "BatchNorm"}
          ],
          "2" : [
            {"inputs" : {"layer" : 1, "node" : 0}, "type" : "Dense", "size" : 17, "activation" : "relu"},
            {"inputs" : {"layer" : 1, "node" : 0}, "type" : "Dense", "size" : 17, "activation" : "relu"},
            {"inputs" : {"layer" : 1, "node" : 0}, "type" : "Dense", "size" : 17, "activation" : "relu"}
          ],
          "3" : [
            {"inputs" : {"layer" : 2, "node" : [0,1,2]}, "type" : "Concat"}
          ],
          "4" : [
            {"inputs" : {"layer" : 3, "node" : 0}, "type" : "Dense", "size" : 1, "activation" : "relu"}
          ]
        }
      }
    }
  
  In this example, the first layer is a Batchnorm layer, followed by 3 Dense node for the second layer, a Concat layer and finaly the output layer,
  for the prediction.
  
  ## Method list, info and ID
  
  - Random forest, to call this method for prediction use "RF" and for class prediction use "RF_Classifier"
  - Deep Neural Network : DNN for prediction and DNN_Classifier for class prediction, by default this method will transform your phenotype class using one hot encoding.
  - Support Vector Machine : SVM for prediction and SVM_Classifier for class prediction
  - Gaussian Process : Gaussian_Process for prediction and Gaussian_Process_Classifier for class prediction
  - Gradient boosting : Gradient_boosting for prediction and Gradient_boosting_Classifier for class prediction
  - Linear regression : Linear_regression for prediction and Linear_Classifier_SGD for class prediction
  - Linear regression with L1 penalisation (Lasso) : Lasso for prediction and Lasso_Classifier_SGD for class prediction
  - Linear regression with L2 penalisation (Ridge) : Ridge  for prediction
  - Linear regression with L1 and L2 penalisation (ElasticNet) : ElasticNet for prediction and Elastic_Classifier_SGD for class prediction
  - A special model based on DNN, to introduce combined plot function : MULTI-DNN_demo
  
  
