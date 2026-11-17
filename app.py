import os, numpy as np, pandas as pd, torch, streamlit as st, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.anfis_model import NeuroFuzzySepsis
from src.data_pipeline import FEATURES

st.set_page_config(page_title="Sepsis Neuro-Fuzzy Simulator",layout="wide")
st.title("Sepsis Neuro-Fuzzy Risk Simulator")
st.warning("Educational prototype only — not for diagnosis or clinical decisions.")
model_path="models/sepsis_anfis.pt"
if not os.path.exists(model_path):
    st.info("Run: python scripts/make_demo_data.py && python -m src.train --data data/demo/sepsis_demo.csv --epochs 40")
    st.stop()

model=NeuroFuzzySepsis(); model.load_state_dict(torch.load(model_path,map_location="cpu")); model.eval()
demo=pd.read_csv("data/demo/sepsis_demo.csv"); scaler=StandardScaler().fit(demo[FEATURES])
defaults={"HR":105.,"O2Sat":94.,"Temp":38.5,"SBP":95.,"Resp":24.,"Lactate":2.8,"WBC":13.}
ranges={"HR":(40.,180.),"O2Sat":(80.,100.),"Temp":(34.,42.),"SBP":(50.,200.),"Resp":(5.,50.),"Lactate":(.3,10.),"WBC":(1.,40.)}
vals={}
for f in FEATURES:
    lo,hi=ranges[f]; vals[f]=st.sidebar.slider(f,lo,hi,defaults[f],step=.1 if f=="Temp" else 1.)
x=scaler.transform(pd.DataFrame([vals])[FEATURES]).astype("float32")
with torch.no_grad(): risk,details=model(torch.tensor(x),return_details=True)
risk=float(risk.item())
a,b=st.columns(2); a.metric("Model risk score",f"{risk*100:.1f}%"); b.metric("Demo threshold","50%")
st.dataframe(pd.DataFrame([vals]),use_container_width=True)
st.subheader("Fuzzy membership degrees")
mu=details["membership"][0].numpy()
st.dataframe(pd.DataFrame(mu,index=FEATURES,columns=["LOW","HIGH"]).style.format("{:.3f}"),use_container_width=True)
st.subheader("Strongest activated rules")
f=details["normalized_firing"][0].numpy(); rules=model.rules.tolist(); top=np.argsort(f)[-10:][::-1]
rows=[]
for idx in top:
    txt=" AND ".join(f"{FEATURES[i]} IS {'LOW' if t==0 else 'HIGH'}" for i,t in enumerate(rules[idx]))
    rows.append({"Rule":txt,"Normalized firing":float(f[idx])})
st.dataframe(pd.DataFrame(rows),use_container_width=True)
st.subheader("Learned HR membership functions")
c=model.centers.detach().numpy()[0]; s=np.exp(model.log_sigmas.detach().numpy()[0]); xx=np.linspace(-3,3,300)
fig,ax=plt.subplots(figsize=(8,4))
for i,label in enumerate(["LOW","HIGH"]): ax.plot(xx,np.exp(-.5*((xx-c[i])/s[i])**2),label=label)
ax.set_xlabel("Standardized HR"); ax.set_ylabel("Membership"); ax.legend(); st.pyplot(fig)
