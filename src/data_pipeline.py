from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURES=["HR","O2Sat","Temp","SBP","Resp","Lactate","WBC"]
TARGET="SepsisLabel"

def load_csv(path):
    return pd.read_csv(path)

def load_physionet(folder, max_patients=None):
    files=sorted(Path(folder).rglob("*.psv"))
    if max_patients: files=files[:max_patients]
    if not files: raise FileNotFoundError(f"No .psv files found under {folder}")
    frames=[]
    for f in files:
        d=pd.read_csv(f,sep="|"); d["PatientID"]=f.stem; frames.append(d)
    return pd.concat(frames,ignore_index=True)

def prepare(df, test_size=0.2, random_state=42):
    df=df.copy()
    df=df[FEATURES+[TARGET]].replace([np.inf,-np.inf],np.nan)
    df[TARGET]=df[TARGET].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(
        df[FEATURES],df[TARGET],test_size=test_size,stratify=df[TARGET],
        random_state=random_state)
    imp=SimpleImputer(strategy="median")
    sc=StandardScaler()
    Xtr=sc.fit_transform(imp.fit_transform(Xtr)).astype("float32")
    Xte=sc.transform(imp.transform(Xte)).astype("float32")
    return Xtr,Xte,ytr.to_numpy().astype("float32"),yte.to_numpy().astype("float32"),imp,sc
