exec_mode=0

for ((i = 0 ; i < 48 ; i++)); do
    timeout 10m python calc_soltimes.py \
        -exec_mode $exec_mode
done


exec_mode=1

for ((i = 0 ; i < 48 ; i++)); do
    timeout 10m python calc_soltimes.py \
        -exec_mode $exec_mode
done