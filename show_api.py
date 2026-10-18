import json, sys
sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open(sys.argv[1], encoding="utf-8"))
for k, v in sorted(d.items(), key=lambda x: int(x[0])):
    ins = {n: (j if not isinstance(j, list) else f"<-{j[0]}.{j[1]}>") for n, j in v["inputs"].items()}
    print(f"#{k:>3} {v['class_type']:<24} {json.dumps(ins, ensure_ascii=False)}")
