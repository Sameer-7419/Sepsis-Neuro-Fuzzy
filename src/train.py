import argparse, json
from pathlib import Path
import numpy as np, torch, joblib
from sklearn.metrics import classification_report,roc_auc_score,average_precision_score
from .anfis_model import NeuroFuzzySepsis
from .data_pipeline import load_csv,prepare

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--model-out",default="models/sepsis_anfis.pt")
    a=ap.parse_args()
    np.random.seed(42); torch.manual_seed(42)
    Xtr,Xte,ytr,yte,imp,sc=prepare(load_csv(a.data))
    model=NeuroFuzzySepsis(); opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    x=torch.tensor(Xtr); y=torch.tensor(ytr)
    pos=y.sum().item(); neg=len(y)-pos; pw=neg/max(pos,1.0)
    hist=[]
    for e in range(a.epochs):
        p=model(x); eps=1e-7
        weights=torch.where(y==1,torch.tensor(pw),torch.tensor(1.0))
        loss=-(weights*(y*torch.log(p+eps)+(1-y)*torch.log(1-p+eps))).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        hist.append(float(loss))
        if (e+1)%10==0: print(f"epoch={e+1:03d} loss={loss.item():.5f}")
    model.eval()
    with torch.no_grad(): prob=model(torch.tensor(Xte)).numpy()
    pred=(prob>=0.5).astype(int)
    print(classification_report(yte,pred,digits=4,zero_division=0))
    if len(np.unique(yte))==2:
        print("ROC-AUC:",roc_auc_score(yte,prob)); print("PR-AUC:",average_precision_score(yte,prob))
    out=Path(a.model_out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out)
    joblib.dump(imp,out.with_suffix(".imputer.joblib")); joblib.dump(sc,out.with_suffix(".scaler.joblib"))
    (out.parent/"history.json").write_text(json.dumps(hist))
    print("Saved",out)

if __name__=="__main__": main()
