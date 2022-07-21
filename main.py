# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from library_python_script.file_dealer.file_manager import FileManager
from library_python_script.tools.classifier import DNN_Classifier, Gaussian_Process_Classifier, Gradient_boosting_Classifier, Linear_Classifier_SGD, RF_Classifier, SVM_Classifier
from library_python_script.tools.predictor import DNN_Predictor, Linear_Regression, RF_Predictor, DNN_auto_Classifier, TSNE_Visualisator, UMAP_Visualisator, SVM_Predictor, Gaussian_Process, Gradient_boosting
from library_python_script.tools.dataAnalyser import DataAnalyser
import json
import sys

# PARAMETERS
parameters = sys.argv[1:]
args = {key : value for key, value in zip(parameters[0::2], parameters[1::2])}
print("[INFO] " + str(args))
phenotype = args['-p'] if '-p' in args.keys() else 'P31'
ktop_1D = int(args['-1D']) if '-1D' in args.keys() else 250
ktop_2D = int(args['-2D']) if '-2D' in args.keys() else 500
model_ID = args['-m'] if '-m' in args.keys() else "PCR_bigger_like_model"
gridSearch = args['-gs']=="True" if '-gs' in args.keys() else False
live = args['-live']=="True" if '-live' in args.keys() else False
early_stopping = [True, int(args['-es'])] if '-es' in args.keys() else [False, 0]
epoch = int(args['-epoch']) if '-epoch' in args.keys() else 1000
iteration = int(args['-it']) if '-it' in args.keys() else 5
batchsize = int(args['-bs']) if '-bs' in args.keys() else 10
verbose = args['-vb']=="True" if '-vb' in args.keys() else False
method = args['-method'] if '-method' in args.keys() else "DNN"
newXdim = int(args['-redim']) if '-redim' in args.keys() else None
dataset = args['-dataset'] if '-dataset' in args.keys() else "ionome"
component = int(args['-component']) if '-component' in args.keys() else 3
categorize_data = args['-categorize'] if '-categorize' in args.keys() else None
nb_cluster = int(args['-nb_cluster']) if '-nb_cluster' in args.keys() else None
normalize = args['-norm']=="True" if '-norm' in args.keys() else False
post_categorize = args['-post_categorize'] if '-post_categorize' in args.keys() else None
random_2D = args['-random_2D']=="True" if '-random_2D' in args.keys() else False
filename_json_model = args['-json_model'] if '-json_model' in args.keys() else None
save_path = args['-save_path'] if '-save_path' in args.keys() else None
x_data_path = args['-x_data_path'] if '-x_data_path' in args.keys() else None
y_data_path = args['-y_data_path'] if '-y_data_path' in args.keys() else None
ID_data_path = args['-ID_data_path'] if '-ID_data_path' in args.keys() else None
nth_elem_1D_path = args['-nth_elem_1D_path'] if '-nth_elem_1D_path' in args.keys() else None
nth_elem_2D_path = args['-nth_elem_2D_path'] if '-nth_elem_2D_path' in args.keys() else None
data_mapper_path = args['-data_mapper'] if '-data_mapper' in args.keys() else ""

with open(data_mapper_path, "r") as f:
    data_mapper = json.load(f)
    for i_elem in range(len(data_mapper["X"]["build"])):
        if data_mapper["X"]["build"][i_elem]['sub_id'] == "ktop1D":
            data_mapper["X"]["build"][i_elem]['sub_id'] = ktop_1D
            data_mapper["X"]["build"][i_elem]['file'] = nth_elem_1D_path
        elif data_mapper["X"]["build"][i_elem]['sub_id'] == "ktop2D":
            data_mapper["X"]["build"][i_elem]['sub_id'] = ktop_2D
            data_mapper["X"]["build"][i_elem]['file'] = nth_elem_2D_path
        else:
            data_mapper["X"]["build"][i_elem]['sub_id'] = {'line' : data_mapper["X"]["build"][i_elem]['sub_id'], 'col':phenotype}
    for i_elem in range(len(data_mapper["Y"]["build"])):
        data_mapper["Y"]["build"][i_elem]["sub_id"] = {'line' : data_mapper["Y"]["build"][i_elem]["sub_id"], 'col':[phenotype]}


if dataset == "ionome":
    # FILES LOADING    
    datas = FileManager(x_data_path = x_data_path, 
                        y_data_path = y_data_path, 
                        ID_data_path = ID_data_path)
    
    datas.loadX(header=data_mapper["X"]["load"]["header"], sep=data_mapper["X"]["load"]["sep"])
    datas.loadY(header=data_mapper["Y"]["load"]["header"], sep=data_mapper["Y"]["load"]["sep"])
    datas.loadID()
    
    
    print("[INFO] Building dataset on IONOME")
    # BUILD SUB Y  
    datas.buildSubYData(method=data_mapper["Y"]["build"][0]["method"], sub_id=data_mapper["Y"]["build"][0]["sub_id"], normalize=normalize, categorize=categorize_data, nb_cluster=nb_cluster, show_cluster=True, save_path=save_path)
    for i_elem in data_mapper["Y"]["build"][1:]:
        datas.buildSubYData(method=data_mapper["Y"]["build"][i_elem]["method"], sub_id=data_mapper["Y"]["build"][i_elem]["sub_id"], normalize=normalize, categorize=categorize_data, nb_cluster=nb_cluster, show_cluster=True, save_path=save_path, append=True)
    
    # BUILD SUB X 
    datas.buildSubXData(method=data_mapper["X"]["build"][0]["method"], sub_id=data_mapper["X"]["build"][0]["sub_id"], file=data_mapper["X"]["build"][0]["file"], normalize=True, random=random_2D, reduce_dim=newXdim) # Add Y part in X
    for i_elem in range(1,len(data_mapper["X"]["build"][1:])):
        datas.buildSubXData(method=data_mapper["X"]["build"][i_elem]["method"], sub_id=data_mapper["X"]["build"][i_elem]["sub_id"], file=data_mapper["X"]["build"][i_elem]["file"], normalize=True, random=random_2D, append=True, reduce_dim=newXdim) # Add Y part in X
    
    # SPLIT TRAIN TEST
    datas.splitTrainVal(categorize_done=categorize_data, normalize = normalize, nb_cluster=nb_cluster, categorize=post_categorize)
elif dataset=="boston":
    print("[INFO] Running dataset on BOSTON")
    datas = FileManager(x_data_path = x_data_path, 
                        y_data_path = y_data_path, 
                        ID_data_path = ID_data_path)
    datas.loadX(sep=",", header=0)
    datas.loadY(sep=",", header=0)
    datas.loadID()
    var = ["crim","zn","indus","chas","nox","rm", "age", "dis",	"rad", "tax", "ptratio", "black", "lstat", "medv"]
    var.remove(phenotype)
    datas.buildSubXData(method="std", sub_id={'col' : var})
    datas.buildSubYData(method="std", sub_id={'col' : [phenotype]})
    datas.splitTrainVal()
elif dataset=="ALE_WSN":
    print("[INFO] Running dataset on ALE_WSN")
    datas = FileManager(x_data_path = x_data_path,
                        y_data_path = y_data_path, 
                        ID_data_path = ID_data_path)
    datas.loadX(sep=",", header=0)
    datas.loadY(sep=",", header=0)
    datas.loadID()
    var = ["anchor_ratio","trans_range","node_density","iterations","ale","sd_ale"]
    var.remove(phenotype)
    datas.buildSubXData(method="std", sub_id={'col' : var})
    datas.buildSubYData(method="std", sub_id={'col' : [phenotype]})
    datas.splitTrainVal()
else:
    print("[ERROR] Dataset is unkonwn")
    raise NotImplementedError

with open(filename_json_model, 'r') as f:
        json_models = json.load(f)

# FIRST GRID SEARCH TO DETERMINE BEST CONFIGURATION  
if gridSearch:
    print("[INFO] Running GridSearch")
    if method == "DNN":
        print("[INFO] Running Deep Neural Network GridSearch")
        dnnGrid_search = DNN_Predictor(data=datas, json_models=json_models[model_ID], live=False, batch_size=batchsize)
        dnnGrid_search.runGridSearch()
    elif method == "DNN_Classifier":
        print("[INFO] Running Deep Neural Network Classifier GridSearch")
        datas.encodeYonehotEncoding()
        dnn1D = DNN_Classifier(data=datas, json_models=json_models[model_ID], live=False, batch_size=batchsize)
        dnn1D.runGridSearch()
    elif method == "Gaussian_Process_Classifier":
        gp_2 = Gaussian_Process_Classifier(data=datas)
        gp_2.runGridSearch()
    else:
        print("[ERROR] Method is unkonwn or is not compatible with gridSearch parameter")
    exit(0)

if method == "HDBSCANNER":
    dA = DataAnalyser(savepath = save_path)
    dA.Hdbscan(np.append(datas.get_y_train(), datas.get_y_test()))
    exit(0)
if method == "DensityPlot":
    dA = DataAnalyser(savepath = save_path)
    dA.DensityPlot(np.append(datas.get_y_train(), datas.get_y_test()))
    exit(0)

if method == "RF":
    print("[INFO] Running Random Forest")
    rf1D = RF_Predictor(data=datas)
    rf1D.run()
    rf1D.plot(save_only=True, save_path=save_path+"RF/")
elif method == "DNN":
    print("[INFO] Running Deep Neural Network")
    dnn1D = DNN_Predictor(data=datas, json_models=json_models[model_ID], live=live, early_stopping=early_stopping, epoch=epoch,iteration=iteration,batch_size=batchsize, verbose=verbose)
    dnn1D.run(save_path=save_path+"DNN/" + model_ID + "/")
    dnn1D.plot(save_only=True, save_path=save_path+"DNN/" + model_ID + "/")
    with open(save_path+"DNN/" + model_ID + "/" + "info.txt", 'w') as f:
        f.write(str(args))
elif method == "Auto_Classifier":
    print("[INFO] Running Deep Neural Network Auto Classifier")
    classifer = DNN_auto_Classifier(data=datas, json_models=json_models[model_ID], live=live, early_stopping=early_stopping, epoch=epoch,iteration=iteration,batch_size=batchsize, verbose=verbose)
    classifer.run(save_path=save_path+"DNN_auto_classifier/" + model_ID + "/")
    classifer.plot(save_only=True, save_path=save_path+"DNN_auto_classifier/" + model_ID + "/")
elif method == 'TSNE':
    print("[INFO] Running TSNE Visualisator")
    visualisator = TSNE_Visualisator(data = datas)
    visualisator.build_model_from_dict({"component" : component, "learning_rate" : 200.0, "init_state" : "random"})
    visualisator.run()
    visualisator.plot(save_path+"TSNE/", live=live)
elif method == 'UMAP':
    print("[INFO] Running UMAP Visualisator")
    visualisator = UMAP_Visualisator(data = datas)
    visualisator.build_model_from_dict({"component" : component, "random_state" : 123, "init_state" : "random"})
    visualisator.run()
    visualisator.plot(save_path+"UMAP/", live=live)
elif method == "SVM":
    print("[INFO] Running SVM Visualisator")
    predictor = SVM_Predictor(data = datas)
    predictor.build_model_from_dict({"tol" : 1e-5, "random_state" : 123})
    predictor.run()
    predictor.plot(save_path = save_path+"SVM/")
elif method == "Gaussian_Process":
    print("[INFO] Running Gaussian Process")
    gp = Gaussian_Process(data=datas)
    gp.build_model_from_dict(model_dict={"0" : {'type' : 'RBF', 'length' : 16384}, "1" : {'type' : 'MATERN', 'length' : 16384, 'nu' : 0.5}, 'OP' : '*'})
    gp.run()
    gp.plot(save_only=True, save_path=save_path+"GP/")
elif method == "Gradient_boosting":
    print("[INFO] Running Gradient boosting")
    gb = Gradient_boosting(data = datas)
    gb.build_model_from_dict({"random_state" : 123})
    gb.run()
    gb.plot(save_only=True, save_path=save_path+"GB/")
elif method == "Linear_regression":
    print("[INFO] Running Linear Regression")
    lm = Linear_Regression(data = datas)
    lm.build_model_from_dict(model_dict={"method" : "Linear"})
    lm.run()
    lm.plot(save_only=True, save_path=save_path+"Linear_Reg/")
elif method == "Lasso":
    print("[INFO] Running Lasso Regression")
    lm = Linear_Regression(data = datas)
    lm.build_model_from_dict(model_dict={"method" : "Lasso"})
    lm.run()
    lm.plot(save_only=True, save_path=save_path+"Lasso/")
elif method == "Ridge":
    print("[INFO] Running Ridge Regression")
    lm = Linear_Regression(data = datas)
    lm.build_model_from_dict(model_dict={"method" : "Ridge"})
    lm.run()
    lm.plot(save_only=True, save_path=save_path+"Ridge/")
elif method == "ElasticNet":
    print("[INFO] Running ElasticNet Regression")
    lm = Linear_Regression(data = datas)
    lm.build_model_from_dict(model_dict={"method" : "ElasticNet"})
    lm.run()
    lm.plot(save_only=True, save_path=save_path+"ElasticNet/")
elif method == "RF_Classifier":
    print("[INFO] Running Random Forest_Classifier")
    rf1D = RF_Classifier(data=datas)
    rf1D.run()
    rf1D.plot(save_only=True, save_path=save_path+"RF_Classifier/")
elif method == "DNN_Classifier":
    print("[INFO] Running Deep Neural Network Classifier")
    datas.encodeYonehotEncoding()
    dnn1D = DNN_Classifier(data=datas, json_models=json_models[model_ID], live=live, early_stopping=early_stopping, epoch=epoch,iteration=iteration,batch_size=batchsize, verbose=verbose)
    dnn1D.run(save_path=save_path+"DNN_Classifier/" + model_ID + "/")
    dnn1D.plot(save_only=True, save_path=save_path+"DNN_Classifier/" + model_ID + "/")
    with open(save_path+"DNN_Classifier/" + model_ID + "/" + "info.txt", 'w') as f:
        f.write(str(args))
elif method == 'TSNE_Classifier':
    print("[INFO] Running TSNE Visualisator Classifier")
elif method == 'UMAP_Classifier':
    print("[INFO] Running UMAP Visualisator Classifier")
elif method == "SVM_Classifier":
    print("[INFO] Running SVM Visualisator Classifier")
    predictor = SVM_Classifier(data = datas)
    predictor.build_model_from_dict()
    predictor.run()
    predictor.plot(save_path = save_path+"SVM_CLassifier/")
elif method == "Gaussian_Process_Classifier":
    print("[INFO] Running Gaussian Process Classifier")
    gp_2 = Gaussian_Process_Classifier(data=datas)
    gp_2.build_model_from_dict(model_dict={"0" : {'type' : 'FLOAT', 'value' : 1.0}, "1" : {'type' : 'RBF', 'length' : 1, 'length_scale_bound' : 1.0}, 'OP' : '*'})
    gp_2.run()
    gp_2.plot(save_only=True, save_path=save_path+"GP_Classifier_MATERN/")
elif method == "Gradient_boosting_Classifier":
    print("[INFO] Running Gradient boosting Classifier")
    gb = Gradient_boosting_Classifier(data = datas)
    gb.build_model_from_dict({"random_state" : 123})
    gb.run()
    gb.plot(save_only=True, save_path=save_path+"GB_Classifier/")
elif method == "Linear_Classifier_SGD":
    print("[INFO] Running Linear Classifier")
    lc = Linear_Classifier_SGD(data = datas)
    lc.build_model_from_dict({"random_state" : 123, "method" : "Standard"})
    lc.run()
    lc.plot(save_only=True, save_path=save_path+"Linear_SGD_Classifier/")
elif method == "Lasso_Classifier_SGD":
    print("[INFO] Running Linear Classifier")
    lc = Linear_Classifier_SGD(data = datas)
    lc.build_model_from_dict({"random_state" : 123, "method" : "Lasso"})
    lc.run()
    lc.plot(save_only=True, save_path=save_path+"Lasso_Classifier_SGD/")
elif method == "Elastic_Classifier_SGD":
    print("[INFO] Running Linear Classifier")
    lc = Linear_Classifier_SGD(data = datas)
    lc.build_model_from_dict({"random_state" : 123, "method" : "elasticnet"})
    lc.run()
    lc.plot(save_only=True, save_path=save_path+"Elastic_Classifier_SGD/")
elif method == "MULTI-DNN_demo":
    assert dataset == "ionome"
    # 1D
    print("Running 1D DNN")
    datas.buildSubXData(method="by_top", sub_id=ktop_1D, file=nth_elem_1D_path)
    datas.splitTrainVal()
    with open(filename_json_model, 'r') as f:
        json_models = json.load(f)
    dnn1D = DNN_Predictor(data=datas, json_models=json_models[model_ID])
    dnn1D.run(save_path=save_path+"1D/")
    dnn1D.plot(save_only=True, save_path=save_path+"1D/")

    # 1D + 2D
    print("Running 1D+2D DNN")
    datas.buildSubXData(method="product", sub_id=ktop_2D, file=nth_elem_2D_path, append=True)
    datas.splitTrainVal()
    with open(filename_json_model, 'r') as f:
        json_models = json.load(f)
    dnn1D2D = DNN_Predictor(data=datas, json_models=json_models[model_ID])
    dnn1D2D.run(save_path=save_path+"1D_2D/")
    dnn1D2D.plot(save_only=True, save_path=save_path+"1D_2D/")

    # 1D + 2D_random
    print("Running 1D+2D DNN Random")
    datas.buildSubXData(method="by_top", sub_id=ktop_1D, file=nth_elem_1D_path)
    datas.buildSubXData(method="product", sub_id=ktop_2D, file=nth_elem_2D_path, append=True, random=True)
    datas.splitTrainVal()
    with open(filename_json_model, 'r') as f:
        json_models = json.load(f)
    dnn1D2D_random = DNN_Predictor(data=datas, json_models=json_models[model_ID])
    dnn1D2D_random.run(save_path=save_path+"1D_2D_random/")
    dnn1D2D_random.plot(save_only=True, save_path=save_path+"1D_2D_random/")

    # plot results
    dnn1D.combine_plot([dnn1D2D, dnn1D2D_random], ["1D", "1D+2D", "1D+2D_random"], save_only=True, save_path=save_path)


else:
    print("[ERROR] Method is unkonwn")
    exit(1)


# # 1D + 2D
# print("Running 1D+2D DNN")
# datas.buildSubXData(method="product", sub_id=ktop_2D, file="/data/home/carluerjb/Documents/data/data_for_NEO/nth_element_out/"+phenotype+"_Ynorm/161152.ktop", append=True)
# datas.splitTrainVal()
# with open(filename_json_model, 'r') as f:
#     json_models = json.load(f)
# dnn1D2D = DNN_Predictor(data=datas, json_models=json_models[model_ID])
# dnn1D2D.run(save_path=save_path+"1D_2D/")
# dnn1D2D.plot(save_only=True, save_path=save_path+"1D_2D/")

# # 1D + 2D_random
# print("Running 1D+2D DNN Random")
# datas.buildSubXData(method="by_top", sub_id=ktop_1D, file="/data/home/carluerjb/Documents/data/data_for_NEO/nth_element_out/"+phenotype+"_Ynorm/diag.ktop")
# datas.buildSubXData(method="product", sub_id=ktop_2D, file="/data/home/carluerjb/Documents/data/data_for_NEO/nth_element_out/"+phenotype+"_Ynorm/161152.ktop", append=True, random=True)
# datas.splitTrainVal()
# with open(filename_json_model, 'r') as f:
#     json_models = json.load(f)
# dnn1D2D_random = DNN_Predictor(data=datas, json_models=json_models[model_ID])
# dnn1D2D_random.run(save_path=save_path+"1D_2D_random/")
# dnn1D2D_random.plot(save_only=True, save_path=save_path+"1D_2D_random/")

# # plot results
# dnn1D.combine_plot([dnn1D2D, dnn1D2D_random], ["1D", "1D+2D", "1D+2D_random"], save_only=True, save_path=save_path)

