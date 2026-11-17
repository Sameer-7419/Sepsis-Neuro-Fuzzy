from pathlib import Path
import numpy as np,pandas as pd
rng=np.random.default_rng(42); n=4000
hr=np.clip(rng.normal(82,15,n),45,160); o2=np.clip(rng.normal(97,2.5,n),84,100)
temp=np.clip(rng.normal(37.1,.7,n),34,41); sbp=np.clip(rng.normal(115,18,n),60,190)
resp=np.clip(rng.normal(18,5,n),7,45); lac=np.clip(rng.lognormal(np.log(1.5),.45,n),.4,8)
wbc=np.clip(rng.normal(9,3.5,n),2,30)
z=.035*(hr-90)+.55*(temp-37.5)+.025*(resp-20)+.045*(100-sbp)+.7*(lac-2)+.04*(wbc-10)+.35*(94-o2)
p=1/(1+np.exp(-z)); y=(rng.random(n)<p*.35).astype(int)
df=pd.DataFrame({"HR":hr,"O2Sat":o2,"Temp":temp,"SBP":sbp,"Resp":resp,"Lactate":lac,"WBC":wbc,"SepsisLabel":y})
Path("data/demo").mkdir(parents=True,exist_ok=True); df.to_csv("data/demo/sepsis_demo.csv",index=False)
print(df["SepsisLabel"].value_counts()); print("Saved data/demo/sepsis_demo.csv")
