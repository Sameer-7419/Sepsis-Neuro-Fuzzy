import argparse
from pathlib import Path
import numpy as np, torch, joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,roc_auc_score,average_precision_score
from .anfis_model import NeuroFuzzySepsis
from .data_pipeline import load_physionet,FEATURES

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--max-patients",type=int,default=5000)
    ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--model-out",default="models/physionet_sepsis_anfis.pt")
    a=ap.parse_args()
    df=load_physionet(a.data,a.max_patients)
    ids=df.PatientID.drop_duplicates().to_numpy(); np.random.default_rng(42).shuffle(ids)
    test_ids=set(ids[:max(1,int(.2*len(ids)))])
    tr=df[~df.PatientID.isin(test_ids)].copy(); te=df[df.PatientID.isin(test_ids)].copy()
    imp=SimpleImputer(strategy="median"); sc=StandardScaler()
    Xtr=sc.fit_transform(imp.fit_transform(tr[FEATURES])).astype("float32")
    Xte=sc.transform(imp.transform(te[FEATURES])).astype("float32")
    ytr=tr.SepsisLabel.astype("float32").to_numpy(); yte=te.SepsisLabel.astype("float32").to_numpy()
    model=NeuroFuzzySepsis(); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    x=torch.tensor(Xtr); y=torch.tensor(ytr); pw=(len(y)-y.sum().item())/max(y.sum().item(),1)
    for e in range(a.epochs):
        p=model(x); eps=1e-7; w=torch.where(y==1,torch.tensor(pw),torch.tensor(1.0))
        loss=-(w*(y*torch.log(p+eps)+(1-y)*torch.log(1-p+eps))).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        if (e+1)%5==0: print(f"epoch={e+1:03d} loss={loss.item():.5f}")
    model.eval()
    with torch.no_grad(): prob=model(torch.tensor(Xte)).numpy()
    pred=(prob>=.5).astype(int)
    print(classification_report(yte,pred,digits=4,zero_division=0))
    if len(np.unique(yte))==2:
        print("ROC-AUC:",roc_auc_score(yte,prob)); print("PR-AUC:",average_precision_score(yte,prob))
    out=Path(a.model_out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out); joblib.dump(imp,out.with_suffix(".imputer.joblib")); joblib.dump(sc,out.with_suffix(".scaler.joblib"))
    print("Saved",out)

if __name__=="__main__": main()
