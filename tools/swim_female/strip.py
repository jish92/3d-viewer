import sys
from PIL import Image
n=sys.argv[1]; cols=int(sys.argv[2]) if len(sys.argv)>2 else 4
ims=[Image.open(f'{n}_{i}.png') for i in range(8)]
w,h=ims[0].size; rows=8//cols
s=Image.new('RGB',(w*cols,h*rows))
for i,im in enumerate(ims): s.paste(im,((i%cols)*w,(i//cols)*h))
s.resize((w*cols*2//3,h*rows*2//3)).save(f'{n}_strip.png')
