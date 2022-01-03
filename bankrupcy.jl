# list all files in the current folder
readdir()

using CSV, DataFrames

data = CSV.read("data.csv", DataFrame)

using Statistics

## check whether the dataset is balanced
mean(data[!, "Bankrupt?"]) 
# 0.032262795

## check correlation between variables
corrmatrix = cor(Matrix(data[!, 2:end]));
# set the diagonal line as 0 for better resolution
for i in 1:size(corrmatrix,1)
    corrmatrix[i,i] = 0
end
using Plots
gr()
heatmap(1:size(corrmatrix,1),
    1:size(corrmatrix,2), corrmatrix,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel=" ", ylabel=" ",
    title="correlation map")
# select variables with high correlation
corrset = DataFrame(Name1 = [], Name2 = [], correlationcoef = []) # this is used to save all the pairs that are listed
corrset2 = [] # this is used to save all the columns that needs to be removed
for i in 1:size(corrmatrix,1)
    for j in i+1:size(corrmatrix,2)
        if abs(corrmatrix[i,j]) > 0.6 
            corrset = [corrset; DataFrame(Name1 = names(data)[i+1], Name2 = names(data)[j+1], correlationcoef = corrmatrix[i,j])]
            push!(corrset2, j+1)
        end
    end
end
# post-processing all the saved data to save it
transform!(corrset, :correlationcoef => ByRow(x -> abs(x))=>:abscorrcoef);
sort!(corrset, order(:abscorrcoef, rev=true));
CSV.write("QC.csv", corrset);
#=
# save the output file for a deeper look
using DelimitedFiles
io = open("qc1.txt", "w") do io
    for x in corrset
        println(io,x)
    end
end
=#
# remove all the correlated columns
unique!(corrset2);
#sort!(corrset2, rev=true)
newdata = data[!, Not(corrset2)];

## prepare data for machine learning
rename!(newdata, :"Bankrupt?" => :Bankrupt)
bankrupted = newdata[[x for x in 1:size(newdata, 1) if newdata[x, 1] == 1], :]
survived = newdata[[x for x in 1:size(newdata, 1) if newdata[x, 1] == 0], :]
forsurvived = rand(1:size(survived)[1], size(bankrupted)[1] * 2) # allowing a slight more survived companies

## split for training and testing datasets
bankrupted_train = sort!( unique!( rand(1:size(bankrupted)[1], floor(Int, size(bankrupted)[1] * 2/3)) ) )
bankrupted_test = deleteat!([x for x in (1:size(bankrupted)[1])],  bankrupted_train)

survived_train = sort!( unique!( rand(1:size(forsurvived)[1], floor(Int, size(forsurvived)[1] * 2/3)) ) )
survived_test = deleteat!([x for x in (1:size(forsurvived)[1])],  survived_train)

train = [bankrupted[bankrupted_train, :]; survived[survived_train, :]]
test = [bankrupted[bankrupted_test, :]; survived[survived_test, :]]

# replace all space in names with "_"
for i = 2:size(train)[2]
    tmpname = names(train)[i]
    newname = filter(x -> !isspace(x), tmpname)
    rename!(test, i => "X$i")
    rename!(train, i => "X$i")
end

## use logistic regression for machine learning
using StatsModels, GLM
fm_bkrp = @formula(Bankrupt ~  X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19 +
X20 + X21 + X22 + X23 + X24 + X25 + X26 + X27 + X28 + X29 + 
X30 + X31 + X32 + X33 + X34 + X35 + X36 + X37 + X38 + X39 +  
X40 + X41 + X42 + X43 + X44 + X45 + X46 + X47 + X48 + X49 )
lm_bkrp = glm(fm_bkrp, train, Binomial(), LogitLink())
pred_bkrp_tmp = predict(lm_bkrp, test)
pred_bkrp = BitArray(pred_bkrp_tmp[x] > 0.8 for x in 1:length(pred_bkrp_tmp)) # binarize the result
# calculate accuracy
correct = 0
for i = 1:length(pred_bkrp)
    if test[i, "Bankrupt"] == pred_bkrp[i]
        correct = correct + 1
    end
end
println( correct/length(pred_bkrp) )
## filter out coefficients that p values is significant
pval = GLM.coeftable(lm_bkrp).cols[4]
## extract those variables with significant impact
for i in 1:length(pval)
    if pval[i] < 0.05
        println(GLM.coeftable(lm_bkrp).cols[1][i], names(data)[i])
    end
end


## another way to use logistic regression
using ScikitLearn
@sk_import linear_model: LogisticRegression
# Fit the model
X_train = Matrix(train[:, 2:end]);
y_train = convert(Array, train[:,1]);
model = LogisticRegression(fit_intercept=true, panelty = 'l2')
log_reg = fit!(model, X_train, y_train)
# Predict on the test set
sklearn_pred = log_reg.predict(Matrix(test[:,2:end]))
# calculate accuracy
correct = 0
for i = 1:length(sklearn_pred)
    if test[i, "Bankrupt"] == sklearn_pred[i]
        correct = correct + 1
    end
end
println( correct/length(sklearn_pred) )

## cross-validation 
using ScikitLearn.CrossValidation: cross_val_score
cross_val_score(LogisticRegression(), X_train, y_train, cv=5)  # 5-fold


## hyperparameter tuning
using ScikitLearn.GridSearch: GridSearchCV
gridsearch = GridSearchCV(LogisticRegression(), Dict(:C => 0.1:0.1:2.0))
fit!(gridsearch, X_train, y_train)
println("Best parameters: $(gridsearch.best_params_)")


## using decision tree
using DecisionTree
model = DecisionTreeClassifier(max_depth=3);
fit!(model, X_train, y_train)
# pretty print of the tree, to a depth of 5 nodes (optional)
print_tree(model, 5)
treepred = predict(model, Matrix(test[:,2:end]));
correct = 0
for i = 1:length(treepred)
    if test[i, "Bankrupt"] == treepred[i]
        correct = correct + 1
    end
end
println( correct/length(treepred) )

## another way to do decision treet classification
model = build_tree(y_train, X_train)
# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, 0.9)
# pretty print of the tree, to a depth of 5 nodes (optional)
print_tree(model, 5)
# apply learned model
preds = apply_tree(model, Matrix(test[:,2:end]));
# generate confusion matrix, along with accuracy and kappa scores
confusion_matrix(test[:,1], preds)

## random forest classifier
# using 2 random features, 100 trees, 0.5 portion of samples per tree, and a maximum tree depth of 6
model = build_forest(y_train, X_train, 2, 100, 0.5, 6);
predforest = apply_forest(model, Matrix(test[:,2:end]));
confusion_matrix(test[:,1], predforest)
