#!/bin/bash

fftlength=2048
int_factor=1
for source_type in synth3 gbt synth7;
do
    for digitize in n y;
    do
        for snr in 10 100 1000 10000;
        do 
            python inject_voltage.py $source_type $digitize $snr $fftlength $int_factor
            rawspec -f $fftlength -t $int_factor -d /datax/scratch/bbrzycki/data/raw_files/inject_voltage /datax/scratch/bbrzycki/data/raw_files/inject_voltage/${source_type}_D${digitize}_SNR${snr}_FFT${fftlength}
            python plot_injection.py /datax/scratch/bbrzycki/data/raw_files/inject_voltage/${source_type}_D${digitize}_SNR${snr}_FFT${fftlength}.rawspec.0000.fil
        done;
    done;
done;