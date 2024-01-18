import pandas as pd
import tarfile
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns

SEED = 0
plt.rcParams["figure.figsize"] = (20,12)


#load data
with tarfile.open(".\\data\poisson\\freMTPLfreq.tar.gz", "r:*") as tar:
    csv_path = tar.getnames()[0]
    df = pd.read_csv(tar.extractfile(csv_path), header=0)


# data preprocessing
df["Frequency"] = df["ClaimNb"] / df["Exposure"]

df["DriverAge_binned"]  = pd.cut(df.DriverAge , bins=[17,22,26,42,74,np.inf])


glm_preprocessor = ColumnTransformer(
    [
        (
            "onehot_categorical",
            OneHotEncoder(),
            ["DriverAge_binned"],
        ),
    ],
    remainder="drop",
)
glm_preprocessor2 = ColumnTransformer(
    [
       ("passthrough_numeric", "passthrough", ["DriverAge"])
    ],
    remainder="drop",
)
glm_preprocessor.fit_transform(df)
glm_preprocessor2.fit_transform(df)


brand_to_letter = {'Japanese (except Nissan) or Korean': "F",
                   'Fiat':"D",
                    'Opel, General Motors or Ford':"C",
                      'Mercedes, Chrysler or BMW': "E",
                      'Renault, Nissan or Citroen': "A",
                     'Volkswagen, Audi, Skoda or Seat':"B",
                      'other':"G" }

df.Brand = df.Brand.apply(lambda x: brand_to_letter[x])

print(
    "Average Frequency = {}".format(np.average(df["Frequency"], weights=df["Exposure"]))
)
print(
    "Fraction of exposure with zero claims = {0:.1%}".format(
        df.loc[df["ClaimNb"] == 0, "Exposure"].sum() / df["Exposure"].sum()
    )
)

print(
    "Fraction of exposure with one claim = {0:.1%}".format(
        df.loc[df["ClaimNb"] == 1, "Exposure"].sum() / df["Exposure"].sum()
    )
)
print(
    "Fraction of exposure with more than one claim = {0:.1%}".format(
        df.loc[df["ClaimNb"] > 1, "Exposure"].sum() / df["Exposure"].sum()
    )
)
print(
    "Fraction of policyholders with exposure equal one = {0:.1%}".format(
        np.sum(df["Exposure"] == 1)/df.shape[0]
    )
)
print(
    "Fraction of policyholders with exposure below one = {0:.1%}".format(
        np.sum(df["Exposure"] < 1)/df.shape[0]
    )
)
print(
    "Fraction of policyholders with exposure above one = {0:.1%}".format(
        np.sum(df["Exposure"] > 1)/df.shape[0]
    )
)


plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]
          }


plt.rcParams.update(plot_params)
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(6.67*1.5*2/3, 4*1.5/2), dpi = 500)
ax0.set_title("Number of claims",fontsize=10)
ax0.set_ylabel("log frequency",fontsize=10)
ax0.set_xlabel(r"$\mathtt{ClaimNb}$",fontsize=10)
sns.histplot(df["ClaimNb"],ax=ax0,element="bars",
    stat="count",log_scale=(False, True),discrete = True)
#_ = df["ClaimNb"].hist(bins=30, log=True, ax=ax0)
ax1.set_title("Exposure in years",fontsize=10)
ax1.set_ylabel("log frequency",fontsize=10)
ax1.set_xlabel(r"$\mathtt{Exposure}$",fontsize=10)
sns.histplot(df["Exposure"],ax=ax1,element="bars",
    stat="count",log_scale=(False, True),bins=30)

# ax2.set_title("Frequency (number of claims per year)",fontsize=10)
# ax2.set_ylabel("log frequency",fontsize=10)
# ax2.set_xlabel("Claim Frequency",fontsize=10)
# #_ = df["Frequency"].hist(bins=30, log=True, ax=ax2)
# sns.histplot(df["Frequency"],ax=ax2,element="bars",
#     stat="count",log_scale=(False, True))

plt.tight_layout()
plt.savefig(f".\\StableTrees_examples\plots\\freMTPLfreq_hist.png", transparent=False, dpi=500, facecolor='white',
            bbox_inches='tight')
plt.close()


plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]
          }


plt.rcParams.update(plot_params)
fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(6.67*1.5, 4*1.5), dpi = 500)
axs[0,0].set_ylabel("Relative frequency",fontsize=10)
axs[0,0].set_xlabel(r"$\mathtt{Power}$",fontsize=10)
sns.histplot(df["Power"].sort_values(),bins =30,ax=axs[0,0],element="bars",
    stat="probability")
axs[0,1].set_ylabel("Relative frequency",fontsize=10)
axs[0,1].set_xlabel(r"$\mathtt{CarAge}$",fontsize=10)
_ = sns.histplot(df["CarAge"],bins =50,ax=axs[0,1],element="bars",
    stat="probability")

axs[0,2].set_ylabel("Relative frequency",fontsize=10)
axs[0,2].set_xlabel(r"$\mathtt{DriverAge}$",fontsize=10)
_ = sns.histplot(df["DriverAge"],bins =25,ax=axs[0,2],element="bars",
    stat="probability")

axs[1,0].set_ylabel("Relative frequency",fontsize=10)
axs[1,0].set_xlabel(r"$\mathtt{Brand}$",fontsize=10)
_ = sns.histplot(df["Brand"].sort_values(),bins =30,ax=axs[1,0],element="bars",
    stat="probability")

axs[1,1].set_ylabel("Relative frequency",fontsize=10)
axs[1,1].set_xlabel(r"$\mathtt{Gas}$",fontsize=10)
_ = sns.histplot(df["Gas"],bins =30,ax=axs[1,1],element="bars",
    stat="probability")

axs[1,2].set_ylabel("Relative frequency",fontsize=10)
axs[1,2].set_xlabel(r"$\mathtt{Density}$",fontsize=10)
_ = sns.histplot((df["Density"]),bins =5,ax=axs[1,2],element="bars",
    stat="probability")


plt.tight_layout()
plt.savefig(f".\\StableTrees_examples\plots\\claim_frequency_hist_features.png", transparent=False, dpi=500, facecolor='white',
            bbox_inches='tight')
plt.close()