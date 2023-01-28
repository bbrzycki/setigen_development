rm /datax/scratch/bbrzycki/data/raw_files/inj_573_50*

python gmrt_injection.py

rawspec -f 8192 -t 1 -d /datax/scratch/bbrzycki/data/raw_files/ /datax/scratch/bbrzycki/data/raw_files/inj_573_50

python gmrt_injection1.py