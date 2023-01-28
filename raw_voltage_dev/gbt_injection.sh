rm /datax/scratch/bbrzycki/data/raw_files/blc00_guppi_59114_15868_TMC1_0010_injected*

python gbt_injection.py

rawspec -f 2048 -t 1 -d /datax/scratch/bbrzycki/data/raw_files/ /datax/scratch/bbrzycki/data/raw_files/blc00_guppi_59114_15868_TMC1_0010_injected

python gbt_injection1.py