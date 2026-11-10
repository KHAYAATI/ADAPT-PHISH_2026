import json,sys,re
f=json.load(open('facts.json'))
pat=re.compile(sys.argv[1])
for k,v in f.items():
    if pat.search(k):
        if isinstance(v,float): v=round(v,4)
        elif isinstance(v,dict): v={a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items()}
        print(k,'=',v)
