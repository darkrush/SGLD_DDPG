cat exp_list.txt | while read line 
do
    while :
    do
    {
    flock -xn -w 1 200
    if [ $? -eq 0 ]; then
        export CUDA_VISIBLE_DEVICES=0;$line &
        break
    fi
    } 200>.GPU00
    {
    flock -xn -w 1 200
    if [ $? -eq 0 ]; then
        export CUDA_VISIBLE_DEVICES=1;$line &
        break
    fi
    } 200>.GPU10
    {
    flock -xn -w 1 200
    if [ $? -eq 0 ]; then
        export CUDA_VISIBLE_DEVICES=0;$line &
        break
    fi
    } 200>.GPU01
    {
    flock -xn -w 1 200
    if [ $? -eq 0 ]; then
        export CUDA_VISIBLE_DEVICES=1;$line &
        break
    fi
    } 200>.GPU11
    done
done