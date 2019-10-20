## Author: Md. Tarikul Islam
## United International University


# loading necessery packages #

using CSV
using DataFrames
using Clustering
using ElasticArrays
using DataStructures
using DecisionTree
using MLDataUtils 
using FreqTables
using PyPlot
using StatsPlots


# reading dataset 
dt = CSV.read("fraudData.csv")

# here df is in DataFrame type and columnName is string type, call like the following
# newdt = OneHotEncoding(dt,"type")
function OneHotEncoding(df,columnName)
    newdf = copy(df)
    for c in unique(df[Symbol(columnName)])
        newdf[Symbol(c)] = ifelse.(df[Symbol(columnName)] .== c, 1, 0)
    end
    deletecols!(newdf, Symbol(columnName))
    return newdf
end

#Encoding the type column
dt = OneHotEncoding(dt,"type")

# size function returns the number of row and columns of the dataset
row, col = size(dt)
println("Number of Instances: ", row)
println("Number of Features: ", col)

names(dt)

# printing first 10 example

first(dt,10)

# Printing last 10 examples

last(dt, 10)

println(describe(dt))

h = PyPlot.plt.hist(dt[:isFraud], 2) 
plt.title("Histogram of isFraud")
plt.xlabel("Nonfraud, fraud")
plt.ylabel("Frequency")


# Deleting String Type Feature 'nameOrig'
# Deleting String Type Feature 'nameDest'

dt = deletecols!(dt, :nameOrig)
dt = deletecols!(dt, :nameDest)

frequency_isFraud = freqtable(dt[:isFraud])

# Ration of non-fraud and fraud 
ratio = prop(frequency_isFraud)

first(dt, 5)

# Setting Random Seed 
using Random
Random.seed!(1234)

#= 
Split dataset to minorityData where isFraud = 1
Split dataset to minorityData where isFraud = 0
=#
minorityData = dt[dt[:, 7] .== 1, :]
majorityData = dt[dt[:, 7] .== 0, :]

# printing size of minorityData

println("Instances of Minority Data: ", nrow(minorityData))
println("Features of Minority Data: ", ncol(minorityData))

# printing size of majorityData
println("Instances of Majority Data: ", nrow(majorityData))
println("Features of Majority Data: ", ncol(majorityData))

# This function splits minority data in given ratio
function split_MinorityData(min_data, ratio)
    minority_mat = convert(Array, min_data[1:13])
    min_trainShuf = shuffleobs(transpose(minority_mat))
    (min_train, min_test) = splitobs(min_trainShuf, at =ratio)
    min_train = Array(transpose(min_train))
    min_test = Array(transpose(min_test))
    return min_train, min_test
end 

# Split minorityData for train an test 
min_train, min_test = split_MinorityData(minorityData, .70)
println(size(min_train))
println(size(min_test))


min_train

maj_mat = convert(Matrix{Float64}, majorityData[1:13])

# converting DataFrame to Matrix and fliping dimension 
# because Clustering.jl package takes in this form

majorityData_mat = permutedims(maj_mat, [2, 1])

# running kmeans clustering using k = 5
k = 5

result = kmeans(majorityData_mat, k; maxiter = 1000, display = :iter)

size(majorityData_mat)

M = result.centers  

#dmat = pairwise( SqEuclidean(), majorityData_mat) ## Equivalent to pairwise(distance, data_matrix, data_matrix)
#dmat = convert(Array{T} where T <: AbstractFloat, dmat) 
#size(dmat)


# size of each cluster out of k

c = counts(result) 

# we need clusters that only contines more than or equal 8213 instances

global j = 1
global cluster_list = []
global clusterNumber = 0
for i in c
    if i >= 8213
        println(j, "th Cluster Contains: ", i)
        clusterNumber = clusterNumber + 1 
        push!(cluster_list, j)
    end
    j = j + 1
end
println("Usable Cluster: ", clusterNumber)
println(cluster_list)

# obtain the resultant assignments
# a[i] indicates which cluster the i-th sample is assigned to

cluster_no = assignments(result)

# Number of row and column in features Matrix
global majority_sz = 6354407
row, col = size(majorityData_mat)
println(col)

# This function returns minoity instances in elastic array [ just a helper function ]
function collectMinority(min)
    
    row, col = size(min)
    
    temp = ElasticArray{Float64}(undef, 13, 0)
    
    for i = 1 : row
        r = min[i, :]
        append!(temp, r)    
    end
    return temp
    #return reshape(temp, row, col)
end

# It's a helper function that takes cluster number as parameter 
# and returns all the instances belongs to this cluster

function collectClusteredData(cl_no)
    
    temp = ElasticArray{Float64}(undef, 13, 0)
    
    for i = 1 : majority_sz
        if cluster_no[i] == cl_no
            append!(temp, majorityData_mat[:, i])
        end
    end
    
    return temp
end

# It's a helper function that returns cluster member of a perticular cluster

function getElem(cluster, i)
    st_idx = (i - 1) * 13 + 1
    en_idx = st_idx + 13 - 1
    return cluster[st_idx : en_idx]
end

# this function takes cluster number and sample size as parameter 
# and returns n-random sample from that cluster 

function randSample(cluster, sample_sz)
    row, col = size(cluster)
    idxs = rand(1:col, sample_sz)
    
    rnd_sample = ElasticArray{Float64}(undef, 13, 0)
    
    for i = 1 : sample_sz
        sample = getElem(cluster, idxs[i])
        append!(rnd_sample, sample)
    end
        
    return  rnd_sample#reshape(rnd_sample,sample_sz, row)
end

# This function merge both minoirty and majority sub data in given sample size
function mergeMajorityMinority(majority, minority, sample_sz)
    return hcat(majority, minority)
end

# cluster_list array has the usable cluster indexes 
println(cluster_list)

# Usable clusters, decided in the previousStep

c1 = collectClusteredData(cluster_list[1])
c2 = collectClusteredData(cluster_list[2])
c3 = collectClusteredData(cluster_list[3])
c4 = collectClusteredData(cluster_list[4])


# min_train and min_test are converted to elastic array. 
min_trains =  collectMinority(min_train)
min_tests =  collectMinority(min_test)

print("Size of minority_train: ", size(min_train))
print("Size of minority_test: ", size(min_test))


# This function rename given dataframe
function renameDf(df)
 
col_name = [:step,          
 :amount,        
 :oldbalanceOrg, 
 :newbalanceOrig,
 :oldbalanceDest,
 :newbalanceDest,
 :isFraud,       
 :isFlaggedFraud,
 :CASH_IN,       
 :CASH_OUT,      
 :DEBIT,         
 :PAYMENT,       
 :TRANSFER]
 return names!(df, col_name)
    
end

function PrepareSampledDataset(cluster, minority_sample, sample_sz)

    picked_sample = randSample(cluster, sample_sz)
    temp_dataset  = mergeMajorityMinority(picked_sample, minority_sample, sample_sz)
    
    features_sz, msamples_sz = size(temp_dataset)
    
    to_dataframe   = convert(DataFrame, temp_dataset)
    transposed_mat = permutedims(convert(Matrix{Float64}, to_dataframe[1:msamples_sz]), [2, 1])

    
    to_dataframe   = convert(DataFrame, transposed_mat)
    #df = to_dataframe.sample(frac=1).reset_index(drop=True)
    dataframe_shuffle = to_dataframe[StatsBase.sample(1:size(to_dataframe,1), size(to_dataframe,1), replace=false),:]
    
    return renameDf(dataframe_shuffle)
    
end

majority_sample_train = 8623
majority_sample_test = 5749


# 7 sample for training

using StatsBase

sample1_c1 = PrepareSampledDataset(c1, min_trains, majority_sample_train)
sample2_c1 = PrepareSampledDataset(c1, min_trains, majority_sample_train)
sample3_c1 = PrepareSampledDataset(c1, min_trains, majority_sample_train)


sample1_c2 = PrepareSampledDataset(c2, min_trains, majority_sample_train)
sample2_c2 = PrepareSampledDataset(c2, min_trains, majority_sample_train)

sample1_c3 = PrepareSampledDataset(c3, min_trains, majority_sample_train)
sample2_c3 = PrepareSampledDataset(c3, min_trains, majority_sample_train)


# sample for test
sample1_c4_test = PrepareSampledDataset(c4, min_tests, majority_sample_test)

CSV.write("sample1_c1.csv", sample1_c1)
CSV.write("sample2_c1.csv", sample2_c1)
CSV.write("sample3_c1.csv", sample3_c1)

CSV.write("sample1_c2.csv", sample1_c2)
CSV.write("sample2_c2.csv", sample2_c2)

CSV.write("sample1_c3.csv", sample1_c3)
CSV.write("sample2_c3.csv", sample2_c3)

CSV.write("sample1_c4_test.csv", sample1_c4_test)

IJulia.installkernel("Julia nodeps", "--depwarn=no")


 using ScikitLearn: fit!, predict, @sk_import, fit_transform! 
 import ScikitLearn: CrossValidation 
 @sk_import metrics: accuracy_score 
 @sk_import ensemble: RandomForestClassifier 
 @sk_import metrics: (confusion_matrix, f1_score, classification_report, precision_score, recall_score)


sample1_c1 = CSV.read("sample1_c1.csv")
sample2_c1 = CSV.read("sample2_c1.csv")
sample3_c1 = CSV.read("sample3_c1.csv")

sample1_c2 = CSV.read("sample1_c2.csv")
sample2_c2 = CSV.read("sample2_c2.csv")

sample1_c3 = CSV.read("sample1_c3.csv")
sample2_c3 = CSV.read("sample2_c3.csv")

sample1_c4_test = CSV.read("sample1_c4_test.csv")

frequency_isFraud = freqtable(sample1_c1[:isFraud])

row, col = size(sample1_c1)
println("Number of Instances in training sub-sample: ", row)
println("Number of features  in training sub-sample: ", col)

frequency_isFraud = freqtable(sample1_c1[:isFraud])

h = PyPlot.plt.hist(sample1_c1[:isFraud], 3) 
plt.title("Histogram of isFraud")
plt.xlabel("Nonfraud, fraud")
plt.ylabel("Frequency")



# Ratio of non-fraud and fraud in training 
ratio = prop(frequency_isFraud)
println("Non-Fraud and Fraud ratio in training sub-sample: ", round(ratio[1]*1000)," : ", round(ratio[2]*1000))

# Size of a sample for test set 
row, col = size(sample1_c4_test)
println("Number of Instances in test sub-sample: ", row)
println("Number of features  in test sub-sample: ", col)

frequency_isFraud = freqtable(sample1_c4_test[:isFraud])

h = PyPlot.plt.hist(sample1_c4_test[:isFraud], 3) 
plt.title("Histogram of isFraud")
plt.xlabel("Nonfraud, fraud")
plt.ylabel("Frequency")


# Ratio of non-fraud and fraud in testing 

ratio = prop(frequency_isFraud)
println("Non-Fraud and Fraud ratio in training sub-sample: ", round(ratio[1]*1000)," : ", round(ratio[2]*1000))

# Features for Training
features = [:step,:amount,:oldbalanceOrg,:newbalanceOrig,:oldbalanceDest,:newbalanceDest,
:isFlaggedFraud,:CASH_IN,:CASH_OUT,:DEBIT,:PAYMENT, :TRANSFER,]

Xfeat_samp1_c1 = convert(Matrix, sample1_c1[features])
Xfeat_samp2_c1 = convert(Matrix, sample2_c1[features])
Xfeat_samp3_c1 = convert(Matrix, sample3_c1[features])
Xfeat_samp1_c2 = convert(Matrix,sample1_c2[features])
Xfeat_samp2_c2 = convert(Matrix,sample2_c2[features])
Xfeat_samp1_c3 = convert(Matrix, sample1_c3[features])
Xfeat_samp2_c3 = convert(Matrix, sample2_c3[features])

Xfeat_samp1_c4_test = convert(Matrix, sample1_c4_test[features])


Ylabel_samp1_c1 = convert(Array, sample1_c1[:isFraud])
Ylabel_samp2_c1 = convert(Array, sample2_c1[:isFraud])
Ylabel_samp3_c1 = convert(Array, sample3_c1[:isFraud])
Ylabel_samp1_c2  = convert(Array, sample1_c2[:isFraud])
Ylabel_samp2_c2  = convert(Array, sample2_c2[:isFraud])
Ylabel_samp1_c3 = convert(Array, sample1_c3[:isFraud])
Ylabel_samp2_c3 = convert(Array, sample2_c3[:isFraud])

Ylabel_samp1_c4_test = convert(Array, sample1_c4_test[:isFraud])

model1 = build_forest(Ylabel_samp1_c1, Xfeat_samp1_c1, 5, 100, 0.5, 10)
model2 = build_forest(Ylabel_samp2_c1, Xfeat_samp2_c1, 5, 100, 0.5, 10)
model3 = build_forest(Ylabel_samp3_c1, Xfeat_samp3_c1, 5, 100, 0.5, 10)
model4 = build_forest(Ylabel_samp1_c2, Xfeat_samp1_c2, 5, 100, 0.5, 10)
model5 = build_forest(Ylabel_samp2_c2, Xfeat_samp2_c2, 5, 100, 0.5, 10)
model6 = build_forest(Ylabel_samp1_c3, Xfeat_samp1_c3, 5, 100, 0.5, 10)
model7 = build_forest(Ylabel_samp2_c3, Xfeat_samp1_c3, 5, 100, 0.5, 10)

# packing all the models into a tuple 
all_models = (model1, model2, model3, model4, model5, model6, model7)

function ensemblerClassifier(Xfeatures_test, models)
    
    model1, model2, model3, model4, model5, model6, model7 = models
    predictions1 = apply_forest(model1, Xfeatures_test)
    predictions2 = apply_forest(model2, Xfeatures_test)
    predictions3 = apply_forest(model3, Xfeatures_test)
    predictions4 = apply_forest(model4, Xfeatures_test)
    predictions5 = apply_forest(model5, Xfeatures_test)
    predictions6 = apply_forest(model6, Xfeatures_test)
    predictions7 = apply_forest(model7, Xfeatures_test)


    predictions = (predictions1, predictions2, predictions3, predictions4, predictions5, predictions6, predictions7)
    
    sz = size(predictions7)
    prediction_rwo_sz = sz[1]

    majorityVoted_predictions = calcualteMajorityVote(predictions, prediction_rwo_sz)
    
    return majorityVoted_predictions
end

function calcualteMajorityVote(predictions, len)
    majority_prediction = Float64[]
    predictions1, predictions2, predictions3, predictions4, predictions5,predictions6, predictions7  = predictions
    for i = 1 : len
       temp = predictions1[i] + predictions2[i] + predictions3[i] + predictions4[i] + predictions5[i] + predictions6[i]+ predictions7[i] 
        if temp > 3
            push!(majority_prediction, 1.0)
        else
            push!(majority_prediction, 0.0)
        end
        temp = 0
    end
    return majority_prediction
end

# ensemblerClassifier function takes test data, and traning models as argument and return votted prediction set. 

votted_prediction = ensemblerClassifier(Xfeat_samp1_c4_test, all_models)

function ensembler_result(votted_prediction, Ylabel_samp1_c4_test)
    acc = accuracy_score(votted_prediction, Ylabel_samp1_c4_test)
    p_score  = precision_score( Ylabel_samp1_c4_test, votted_prediction)
    r_score = recall_score(Ylabel_samp1_c4_test, votted_prediction)
    f_score = f1_score( Ylabel_samp1_c4_test, votted_prediction)

    return acc, p_score, r_score, f_score 
end

all_result =  ensembler_result(votted_prediction, Ylabel_samp1_c4_test)

println("Accuracy: ",  all_result[1] * 100)
println("Precision: ", all_result[2] * 100)
println("Recall: ", all_result[3] * 100)
println("f1-Score: ", all_result[4] * 100)

confusion_matrix(Ylabel_samp1_c4_test, votted_prediction)
