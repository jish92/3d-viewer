import json,sys
id,label,url,anim,note=sys.argv[1:6]
m=json.load(open('models.json'))
m['models']=[e for e in m['models'] if e['id']!=id]+[dict(id=id,label=label,url=url,anim=anim,note=note)]
json.dump(m,open('models.json','w'),indent=1)
