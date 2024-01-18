from stabletrees.random_forest import RF
from sklearn.datasets import make_regression
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from stabletrees import BaseLineTree,AbuTree,NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree,SklearnTree
from adjustText import adjust_text
from pareto_efficient import is_pareto_optimal
from stabletrees.random_forest import RF
import tarfile
from sklearn.linear_model import PoissonRegressor
from stabletrees.AGTBoost import AGTBoost

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#


parameters = {'max_depth':[None, 5, 10],"min_samples_leaf": [5]} # , 
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)


SEED = 0
EPSILON = 1.1

################################################################
## data prepocessing 
################################################################
with tarfile.open("..\data\poisson\\freMTPLfreq.tar.gz", "r:*") as tar:
    csv_path = tar.getnames()[0]
    df = pd.read_csv(tar.extractfile(csv_path), header=0)


df["Frequency"] = df["ClaimNb"] / df["Exposure"]

brand_to_letter = {'Japanese (except Nissan) or Korean': "F",
                   'Fiat':"D",
                    'Opel, General Motors or Ford':"C",
                      'Mercedes, Chrysler or BMW': "E",
                      'Renault, Nissan or Citroen': "A",
                     'Volkswagen, Audi, Skoda or Seat':"B",
                      'other':"G" }


# glm binning based on book
df["Density_binned"] = pd.cut(df.Density, include_lowest=True, bins=[0,40,200,500,4500,np.inf])
df["DriverAge_binned"]  = pd.cut(df.DriverAge , bins=[17,22,26,42,74,np.inf])
df["CarAge_binned"]  = pd.cut(df.CarAge, include_lowest=True , bins=[0,15,np.inf])
df["brandF"] = np.where(df.Brand=="Japanese (except Nissan) or Korean","F","other")
df["Power_glm"] = ["DEF" if p in ["d","e","f"] else "other" if p in ["d","e","f"] else "GH" for p in df.Power ]
df.insert(len(df.columns)-1, 'Frequency', df.pop('Frequency'))
df.Brand = df.Brand.apply(lambda x: brand_to_letter[x])

tree_preprocessor = ColumnTransformer(
    [
        ("categorical",
            OrdinalEncoder(),
            ["Brand", "Power", "Gas", "Region"],
        ),
        ("numeric", "passthrough", ["CarAge","DriverAge","Density"]),
    ],
    remainder="drop",
)

glm_preprocessor = ColumnTransformer(
    [
        (
            "onehot_categorical",
            OneHotEncoder(),
            ["CarAge_binned","Gas","Power_glm", "brandF","DriverAge_binned", "Density_binned"],
        ),
    ],
    remainder="drop",
)

tree_preprocessor.fit_transform(df)
glm_preprocessor.fit_transform(df)

################################################################
## Dictionaries with plotting info 
################################################################
markers = {"basetree":"tree","NU":"NU","SL": "SL", "SL1":"SL_{0.1}", "SL2":"SL_{0.25}","SL3":"SL_{0.5}",
            "SL4": "SL_{0.75}", "SL5": "SL_{0.9}",
            "TR":"TR","TR1":"TR_{0,5}",
            "TR2":"TR_{5,5}", "TR3" :"TR_{10,5}",
            "ABU":"ABU",
            "BABU":"BABU", "BABU1": r"BABU_{1}","BABU2": r"BABU_{3}" ,"BABU3": r"BABU_{5}","BABU4": r"BABU_{7}","BABU5": r"BABU_{10}","BABU6": r"BABU_{20}",
               "baseforest":"rf",
                "NUrf": "NUrf", "TRrf":"TRrf", "TRrf2":"TRrf2", "TRrf3":"TRrf3", "ABUrf":"ABUrf", "BABUrf1":"rf_{BABU_{1}}","BABUrf2":"rf_{BABU_{3}}","BABUrf3":"rf_{BABU_{5}}",
                 "SLrf1" :"rf_{SL_{0.1}}","SLrf2" :"rf_{SL_{0.25}}"  ,"SLrf3" :"rf_{SL_{0.5}}" ,"SLrf4" :"rf_{SL_{0.75}}" ,"SLrf5" :"rf_{SL_{0.9}}" ,
            "baseGTB":"GTB" }

markers_to_method = {"basetree":"tree","NU":"tree","SL": "tree", "SL1":"tree", "SL2":"tree","SL3":"tree",
            "SL4": "tree", "SL5": "tree",
            "TR":"tree","TR1":"tree",
            "TR2":"tree", "TR3" :"tree",
            "ABU":"tree",
            "BABU":"tree", "BABU1": "tree","BABU2": "tree" ,"BABU3": "tree","BABU4": "tree","BABU5": "tree","BABU6": "tree",
               "baseforest":"rf",
                "NUrf": "rf",
                  "TRrf":"rf","TRrf2":"rf", "TRrf3":"rf",
                "ABUrf":"rf", "BABUrf1":"rf","BABUrf2":"rf","BABUrf3":"rf",
                 "SLrf1" :"rf","SLrf2" :"rf"  ,"SLrf3" :"rf" ,"SLrf4" :"rf" ,"SLrf5" :"rf" ,
            "baseGTB":"GTB" }

markers_to_m = {"basetree":"baseline","NU":"NU","SL1":"SL_{0.1}", "SL2":"SL_{0.25}","SL3":"SL_{0.5}",
             "SL4": "SL_{0.75}", "SL5": "SL_{0.9}",
             "TR":"TR","TR1":"TR_{0,5}",
            "TR2":"TR_{5,5}", "TR3" :"TR_{10,5}",
            "ABU":"ABU",
            "BABU":"BABU", "BABU1": r"BABU_{1}","BABU2": r"BABU_{3}" ,"BABU3": r"BABU_{5}","BABU4": r"BABU_{7}","BABU5": r"BABU_{10}","BABU6": r"BABU_{20}",
               "baseforest":"baseline",
                "NUrf": "NU", "TRrf":"TR_{0,5}","TRrf2":"TR_{5,5}", "TRrf3":"TR_{10,5}",
                  "ABUrf":"ABU", "BABUrf1":r"BABU_{1}","BABUrf2":r"BABU_{3}","BABUrf3":r"BABU_{5}",
                 "SLrf1" :"SL_{0.1}","SLrf2" :"SL_{0.25}" ,"SLrf3" :"SL_{0.5}" ,"SLrf4" :"SL_{0.75}" ,"SLrf5" :"SL_{0.9}" ,
            "baseGTB":"GTB" }


colors = {"basetree":"#1f77b4","NU":"#D55E00", "SL":"#CC79A7","SL1":"#CC79A7", "SL2":"#CC79A7","SL3":"#CC79A7",
            "SL4": "#CC79A7", "SL5": "#CC79A7",
            "TR":"#009E73","TR1":"#009E73", "TR2":"#009E73","TR3":"#009E73",
            "TR4":"#009E73", "TR5":"#009E73","TR6":"#009E73",
            "TR7":"#009E73", "TR8":"#009E73","TR9":"#009E73",
            "ABU":"#F0E442",
            "BABU": "#E69F00", "BABU1": "#E69F00","BABU2": "#E69F00" ,"BABU3": "#E69F00","BABU4": "#E69F00","BABU5": "#E69F00","BABU6": "#E69F00",
            "baseforest":"#1f77b4",
            "NUrf":"#D55E00",
            "TRrf" : "#009E73",
            "TRrf2" : "#009E73",
            "TRrf3" : "#009E73",
            "SLrf1" :"#CC79A7","SLrf2" :"#CC79A7","SLrf3" :"#CC79A7","SLrf4" :"#CC79A7","SLrf5" :"#CC79A7",
            "ABUrf":"#F0E442",
            "BABUrf1":"#E69F00", "BABUrf2":"#E69F00","BABUrf3":"#E69F00",
            "baseGTB":"#1f77b4"}

colors2 = {"tree":"#1f77b4", "NU":"#D55E00", "SL":"#CC79A7", 
            "TR":"#009E73", 
            "ABU":"#F0E442",
            "BABU": "#E69F00"}


plot_info  = []
import statsmodels.api as sm
import statsmodels.formula.api as smf
plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}


################################################################
## Experimental
################################################################
criterion = "poisson"
# parameters = {"ccp_alpha" : [1e-4]}
models = {  
            "basetree": BaseLineTree(criterion = criterion, max_depth=5, min_samples_leaf=5),
            "NU": NaiveUpdate(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "TR1": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0),
            "TR2": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0.05),
            "TR3": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0.1),
            "SL1": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.1),
            "SL2": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.25),
            "SL3": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.5),
            "SL4": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.75),
            "SL5": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.9),
            "ABU": AbuTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "BABU1": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=1),
            "BABU2": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=3),
            "BABU3": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=5),
            "BABU4": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=7),
            "BABU5": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=10),
            "BABU6": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=20),
            "baseforest": RF("base",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=False),
            "baseforest_adapt": RF("base",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "NUrf": RF("nu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "TRrf": RF("tr",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "TRrf2": RF("tr",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0.05),
            "TRrf3": RF("tr",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0.1),
            "SLrf1": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.1),
            "SLrf2": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.25),
            "SLrf3": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.5),
            "SLrf4": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.75),
            "SLrf5": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.9),
            "ABUrf": RF("abu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "BABUrf1": RF("babu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=1),
            "BABUrf2": RF("babu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=3),
            "BABUrf3": RF("babu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=5),
            "sklearn": GridSearchCV(DecisionTreeRegressor(criterion="poisson",random_state=0), parameters),
            "poisReg": PoissonRegressor(solver="newton-cholesky"),
            "glm":  smf.glm,
            "TR": TreeReevaluation(criterion = criterion, max_depth=5, min_samples_leaf=5),
            "SL": StabilityRegularization(criterion = criterion, max_depth=5, min_samples_leaf=5),
            "ABU": AbuTree(criterion = criterion, max_depth=5, min_samples_leaf=5),
            "BABU": BABUTree(criterion = criterion, max_depth=5, min_samples_leaf=5),
            #"baseGTB" : AGTBoost(loss_function=criterion,gamma=0),
            # "SLbaseGTB" : AGTBoost(loss_function=criterion,gamma=0.1)
            }

# dictionaries to store results
stability_all = {name:[] for name in models.keys()}
standard_stability_all= {name:[] for name in models.keys()}
mse_all= {name:[] for name in models.keys()}

stability = {name:[] for name in models.keys()}
standard_stability = {name:[] for name in models.keys()}
mse = {name:[] for name in models.keys()}

mse2 = {name:[] for name in models.keys()}

stability2 = {name:[] for name in models.keys()}

train_stability = {name:[] for name in models.keys()}
train_standard_stability = {name:[] for name in models.keys()}
train_mse = {name:[] for name in models.keys()}
orig_stability = {name:[] for name in models.keys()}
orig_standard_stability = {name:[] for name in models.keys()}
orig_mse = {name:[] for name in models.keys()}
#parameters = {"ccp_alpha" : [0,0.01,1e-3,1e-4, 1e-5]}


validation_score = {name:[] for name in models.keys()}
name = 'BABUrf2'
print(colors[name])
print(markers[name])
print(markers_to_method[name])
print(markers_to_m[name])

#clf = GridSearchCV(DecisionTreeRegressor(random_state=0, criterion="poisson"), parameters)
kf = RepeatedKFold(n_splits= 6,n_repeats=1, random_state=SEED) # 6-fold cross-validation
itesd= 1
for train_index, test_index in kf.split(df.to_numpy()):
    df_12 = df.iloc[train_index]
    
    # print(df_12.columns)
    
    # df12 = pd.DataFrame(tree_preprocessor.transform(df_12),columns=["Brand", "Power", "Gas", "Region","CarAge","DriverAge","Density"])
    # print(df_12.Frequency.isnull().any())
    # print(df_12.Exposure.isnull().any())
    # print(df_12.ClaimNb.isnull().any())
    # df12["Frequency"] = df_12["Frequency"].to_numpy()
    # df12["Exposure"] = df_12["Exposure"].to_numpy()
    # df12["ClaimNb"] = df_12["ClaimNb"].to_numpy()
    # df12.to_csv(f'freq_data_6fold_cv/D2/D2_{itesd}.csv', index=False)
    # if df12.isnull().any().any():
    #     print("df12 The DataFrame contains NaN values.")
    # else:
    #     print("df12 The DataFrame does not contain NaN values.")
    # print(itesd)
    
    df_test = df.iloc[test_index]
    
    # dftest = pd.DataFrame(tree_preprocessor.transform(df_test),columns=["Brand", "Power", "Gas", "Region","CarAge","DriverAge","Density"])
    # dftest["Frequency"] = df_test["Frequency"].to_numpy()
    # dftest["Exposure"] = df_test["Exposure"].to_numpy()
    # dftest["ClaimNb"] = df_test["ClaimNb"].to_numpy()
    # dftest.to_csv(f'freq_data_6fold_cv/D_test/D_test_{itesd}.csv', index=False)
    # if dftest.isnull().any().any():
    #     print("dftest The DataFrame contains NaN values.")
    # else:
    #     print("dftest The DataFrame does not contain NaN values.")

    df_1,df_2 =  train_test_split(df_12, test_size=0.5, random_state=SEED)
    df_1_train, df_1_val = train_test_split(df_1, test_size=0.3, random_state=SEED)
    
    # df1 = pd.DataFrame(tree_preprocessor.transform(df_1),columns=["Brand", "Power", "Gas", "Region","CarAge","DriverAge","Density"])
    # df1["Frequency"] = df_1["Frequency"].to_numpy()
    # df1["Exposure"] = df_1["Exposure"].to_numpy()
    # df1["ClaimNb"] = df_1["ClaimNb"].to_numpy()
    # if df1.isnull().any().any():
    #     print("df1 The DataFrame contains NaN values.")
    # else:
    #     print("df1 The DataFrame does not contain NaN values.")
    # df1.to_csv(f'freq_data_6fold_cv/D1/D1_{itesd}.csv', index=False)
    print(itesd)
    itesd+=1
    # clf.fit(X1,y1)
    # params = clf.best_params_
    # initial model 
    criterion = "poisson"
    models = {  
             "basetree": BaseLineTree(criterion = criterion, max_depth=5, min_samples_leaf=5, adaptive_complexity=True),
            "NU": NaiveUpdate(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
             "TR1": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0),
             "TR2": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0.05),
             "TR3": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0.1),
             "SL1": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.1),
            "SL2": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.25),
             "SL3": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.5),
             "SL4": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.75),
             "SL5": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.9),
            "ABU": AbuTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
             "BABU1": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=1),
             "BABU2": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=3),
             "BABU3": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=5),
             "BABU4": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=7),
            "BABU5": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=10),
            "BABU6": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=20),
            "baseforest": RF("base",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=100,adaptive_complexity=False),
            "baseforest": RF("base",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "NUrf": RF("nu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "TRrf": RF("tr",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "TRrf2": RF("tr",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0.05),
            "TRrf3": RF("tr",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0.1),
            "SLrf1": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.1),
            "SLrf2": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.25),
            "SLrf3": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.5),
            "SLrf4": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.75),
            "SLrf5": RF("sl",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.9),
            "ABUrf": RF("abu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            "BABUrf1": RF("babu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=False,bumping_iterations=1),
            "BABUrf2": RF("babu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=3),
            "BABUrf3": RF("babu",n_estimators= 100,max_features="third",criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=5),
            "sklearn": GridSearchCV(DecisionTreeRegressor(criterion="poisson",random_state=0), parameters),
            "poisReg": PoissonRegressor(solver="newton-cholesky"),
            # "TR": TreeReevaluation(criterion = criterion, max_depth=5, min_samples_leaf=5),
            # "SL": StabilityRegularization(criterion = criterion, max_depth=5, min_samples_leaf=5),
            # "ABU": AbuTree(criterion = criterion, max_depth=5, min_samples_leaf=5),
            # "BABU": BABUTree(criterion = criterion, max_depth=5, min_samples_leaf=5),
            #"BABU": BABUTree(criterion = criterion, max_depth=5, min_samples_leaf=5),
            #"baseGTB" : AGTBoost(loss_function=criterion,gamma=0),
            #"SLbaseGTB" : AGTBoost(loss_function=criterion,gamma=0.1)



            }
    
    for name, model in models.items():
        # model.fit(tree_preprocessor.transform(df_1_train),df_1_train.Frequency, sample_weight=df_1_train.Exposure)
        # pred1 = model.predict(tree_preprocessor.transform(df_1_val) )
        # validation_score[name] = mean_poisson_deviance(df_1_val.ClaimNb, pred1*df_1_val.Exposure)
        # print(name, validation_score[name])
        if name == "poisReg":
            preprocessor = glm_preprocessor            
        else:
            preprocessor = tree_preprocessor
        if name == "glm":
            m = smf.glm("Frequency~DriverAge_binned+Density_binned+CarAge_binned+brandF+Power_glm+Gas", df_1, family=sm.families.Poisson(), freq_weights=df_1['Exposure']).fit()
            pred1 = m.predict(df_test) 
        elif name == "baseGTB":
            model.fit(preprocessor.transform(df_1),df_1.ClaimNb,verbose = 25, offset=np.log(df_1["Exposure"]))
            pred1 = model.predict(preprocessor.transform(df_test),offset=np.log(df_test.Exposure) )
            print(pred1)
        else:
            model.fit(preprocessor.transform(df_1),df_1.Frequency, sample_weight=df_1.Exposure)
            pred1 = model.predict(preprocessor.transform(df_test) )


        #print("before")
        if name == "poisReg":
            model.fit(preprocessor.transform(df_12),df_12.Frequency, sample_weight=df_12["Exposure"])
        elif name == "sklearn":
            model.fit(preprocessor.transform(df_12),df_12.Frequency, sample_weight=df_12["Exposure"])
            params = model.best_params_
            #print(params)
        elif name == "GGTB":
            model.fit(preprocessor.transform(df_12),df_12.Frequency, sample_weight=df_12["Exposure"])
            #print(params)
        elif name == "baseGTB":
            model.fit(preprocessor.transform(df_12),df_12.ClaimNb,verbose = 25, offset =np.log(df_12.Exposure))
            #print(params)
        elif name == "glm":
            m = smf.glm("Frequency~DriverAge_binned+Density_binned+CarAge_binned+brandF+Power_glm+Gas", df_1, family=sm.families.Poisson(), freq_weights=df_1['Exposure']).fit()
            #print(params)
        else:
            model.update(preprocessor.transform(df_12),df_12.Frequency, sample_weight=df_12["Exposure"])

        if name == "glm":
            pred2 = m.predict(df_test) 
        elif name == "baseGTB":
            pred2 = model.predict(preprocessor.transform(df_test),offset=np.log(df_test["Exposure"]) )
            #pred2 = m.predict(tree_preprocessor.transform(df),offset = np.log(df.Exposure) )
            print(pred2)
        else:
            pred2 = model.predict(preprocessor.transform(df_test) )


        mse[name].append(mean_poisson_deviance(df_test.ClaimNb, pred2*df_test.Exposure))
        stability[name].append(S1(pred1*df_test["Exposure"],pred2*df_test["Exposure"]))

        # mse2[name].append(mean_poisson_deviance(df_test.Frequency, pred2, sample_weight = df_test["Exposure"]))
        # stability2[name].append(S1(pred1,pred2))

    
for name in models.keys():
    print("="*80)
    print(f"{name}")
    mse_scale = np.mean(mse["poisReg"]);
        
    mse_scale = np.mean(mse["poisReg"]); S_scale = np.mean(stability["poisReg"]);
    loss_score = np.mean(mse[name])
    loss_SE = np.std(mse[name])/np.sqrt(len(mse[name]))
    loss_SE_norm = np.std(mse[name]/mse_scale)/np.sqrt(len(mse[name]))
    stability_score = np.mean(stability[name])
    stability_SE = np.std(stability[name])/np.sqrt(len(mse[name]))
    stability_SE_norm = np.std(stability[name]/S_scale)/np.sqrt(len(mse[name]))
    print(f"test - poisson: {loss_score:.4f} ({loss_SE:.3f}), stability: {stability_score:.4f} ({stability_SE:.3f})")
    print(f"test - poisson: {loss_score/mse_scale:.4f} ({loss_SE_norm:.3f}), stability: {stability_score/S_scale:.4f} ({stability_SE_norm:.3f})")
    print("="*80)
    if name != "sklearn" and name != "poisReg" and name != "glm":
        x_abs =  np.mean((mse[name]))
        y_abs = np.mean(stability[name])
        x_abs_se = loss_SE
        y_abs_se =stability_SE
        x_se  = loss_SE_norm
        y_se  = stability_SE_norm
        x_r = x_abs/mse_scale
        y_r = y_abs/S_scale
        plot_info.append((x_r,y_r,colors[name],markers[name], x_abs,y_abs,x_se, y_se, x_abs_se, y_abs_se, markers_to_method[name], markers_to_m[name] ))
print()
print(plot_info)

# import os
# df = pd.DataFrame(plot_info, columns=['loss', 'stability', 'color', "marker", 'loss_abs','stability_abs','loss_se','stability_se','loss_abs_se','stability_abs_se',"method", "m"  ] )
# if os.path.isfile('results/claim_freq_results.csv'):
#     old_df =pd.read_csv('results/claim_freq_results.csv')
#     for i,m in enumerate(df.marker):
#         index = old_df.loc[(old_df["marker"] ==m)].index
#         values  = df.iloc[i]
#         if len(index)>0:
#             old_df.iloc[index]=values
#         else:
#             print(values)
#             old_df  = old_df.append(values, ignore_index=True)

#     old_df.to_csv('results/claim_freq_results.csv', index=False)
# else:
#      df.to_csv('results/claim_freq_results.csv', index=False)


# import os
# df = pd.DataFrame(plot_info, columns=['loss', 'stability', 'color', "marker", 'loss_abs','stability_abs','loss_se','stability_se','loss_abs_se','stability_abs_se',"method", "m"  ] )
# if os.path.isfile('results/claim_freq_results_tree_2705.csv'):
#     old_df =pd.read_csv('results/claim_freq_results_tree_2705.csv')
#     for i,m in enumerate(df.marker):
#         index = old_df.loc[(old_df["marker"] ==m)].index
#         values  = df.iloc[i]
#         if len(index)>0:
#             old_df.iloc[index]=values
#         else:
#             print(values)
#             old_df  = old_df.append(values, ignore_index=True)

#     old_df.to_csv('results/claim_freq_results_tree_2705.csv', index=False)
# else:
#      df.to_csv('results/claim_freq_results_tree_2705.csv', index=False)

