# 注意 代码是在src里面，也就是有mian函数位置，手动运行

# baby的数据集的参数设置
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/babypara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/babypara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/babypara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/babypara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/babypara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &






nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e+1 --mask_weight_g 1e-2 --missing_rate 0.3 -c 0 > ./Parameters/babypara-mask_weight_f_1e+1-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-0 --mask_weight_g 1e-2 --missing_rate 0.3 -c 1 > ./Parameters/babypara-mask_weight_f_1e-0-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-1 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Parameters/babypara-mask_weight_f_1e-1-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 3 > ./Parameters/babypara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Parameters/babypara-mask_weight_f_1e-3-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-4 --mask_weight_g 1e-2 --missing_rate 0.3 -c 0 > ./Parameters/babypara-mask_weight_f_1e-4-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-5 --mask_weight_g 1e-2 --missing_rate 0.3 -c 1 > ./Parameters/babypara-mask_weight_f_1e-5-mask_weight_g-1e-2-missing3.log  2>&1 &


nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 4 > ./Parameters/babyflow-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Parameters/babyflow-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 4 > ./Parameters/babyflow-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 6 > ./Parameters/babyflow-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 6 > ./Parameters/babyflow-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &





nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e+1 --missing_rate 0.3 -c 2 > ./Parameters/babypara-mask_weight_f_1e-2-mask_weight_g-1e+1-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-0 --missing_rate 0.3 -c 4 > ./Parameters/babypara-mask_weight_f_1e-2-mask_weight_g-1e-0-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-1 --missing_rate 0.3 -c 6 > ./Parameters/babypara-mask_weight_f_1e-2-mask_weight_g-1e-1-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 0 > ./Parameters/babypara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-3 --missing_rate 0.3 -c 1 > ./Parameters/babypara-mask_weight_f_1e-2-mask_weight_g-1e-3-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-4 --missing_rate 0.3 -c 2 > ./Parameters/babypara-mask_weight_f_1e-2-mask_weight_g-1e-4-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-5 --missing_rate 0.3 -c 4 > ./Parameters/babypara-mask_weight_f_1e-2-mask_weight_g-1e-5-missing3.log  2>&1 &






nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e+1 --mask_weight_g 1e-2 --missing_rate 0.3 -c 4 > ./Parameters/clothingpara-mask_weight_f_1e+1-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-0 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Parameters/clothingpara-mask_weight_f_1e-0-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-1 --mask_weight_g 1e-2 --missing_rate 0.3 -c 4 > ./Parameters/clothingpara-mask_weight_f_1e-1-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Parameters/clothingpara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.3 -c 4 > ./Parameters/clothingpara-mask_weight_f_1e-3-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-4 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Parameters/clothingpara-mask_weight_f_1e-4-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-5 --mask_weight_g 1e-2 --missing_rate 0.3 -c 4 > ./Parameters/clothingpara-mask_weight_f_1e-5-mask_weight_g-1e-2-missing3.log  2>&1 &




nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e+1 --missing_rate 0.3 -c 6 > ./Parameters/clothingpara-mask_weight_f_1e-2-mask_weight_g-1e+1-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-0 --missing_rate 0.3 -c 4 > ./Parameters/clothingpara-mask_weight_f_1e-2-mask_weight_g-1e-0-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-1 --missing_rate 0.3 -c 6 > ./Parameters/clothingpara-mask_weight_f_1e-2-mask_weight_g-1e-1-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 4 > ./Parameters/clothingpara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &

nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-3 --missing_rate 0.3 -c 6 > ./Parameters/clothingpara-mask_weight_f_1e-2-mask_weight_g-1e-3-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-4 --missing_rate 0.3 -c 4 > ./Parameters/clothingpara-mask_weight_f_1e-2-mask_weight_g-1e-4-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-5 --missing_rate 0.3 -c 6 > ./Parameters/clothingpara-mask_weight_f_1e-2-mask_weight_g-1e-5-missing3.log  2>&1 &







nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e+1 --mask_weight_g 1e-2 --missing_rate 0.3 -c 0 > ./Parameters/tiktokpara-mask_weight_f_1e+1-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-0 --mask_weight_g 1e-2 --missing_rate 0.3 -c 1 > ./Parameters/tiktokpara-mask_weight_f_1e-0-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-1 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Parameters/tiktokpara-mask_weight_f_1e-1-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 3 > ./Parameters/tiktokpara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Parameters/tiktokpara-mask_weight_f_1e-3-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-4 --mask_weight_g 1e-2 --missing_rate 0.3 -c 0 > ./Parameters/tiktokpara-mask_weight_f_1e-4-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-5 --mask_weight_g 1e-2 --missing_rate 0.3 -c 1 > ./Parameters/tiktokpara-mask_weight_f_1e-5-mask_weight_g-1e-2-missing3.log  2>&1 &




nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-2 --mask_weight_g 1e+1 --missing_rate 0.3 -c 2 > ./Parameters/tiktokpara-mask_weight_f_1e-2-mask_weight_g-1e+1-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-2 --mask_weight_g 1e-0 --missing_rate 0.3 -c 4 > ./Parameters/tiktokpara-mask_weight_f_1e-2-mask_weight_g-1e-0-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-2 --mask_weight_g 1e-1 --missing_rate 0.3 -c 6 > ./Parameters/tiktokpara-mask_weight_f_1e-2-mask_weight_g-1e-1-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 0 > ./Parameters/tiktokpara-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-2 --mask_weight_g 1e-3 --missing_rate 0.3 -c 1 > ./Parameters/tiktokpara-mask_weight_f_1e-2-mask_weight_g-1e-3-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-2 --mask_weight_g 1e-4 --missing_rate 0.3 -c 2 > ./Parameters/tiktokpara-mask_weight_f_1e-2-mask_weight_g-1e-4-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-2 --mask_weight_g 1e-5 --missing_rate 0.3 -c 4 > ./Parameters/tiktokpara-mask_weight_f_1e-2-mask_weight_g-1e-5-missing3.log  2>&1 &


















# baby的数据集的参数设置
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/babymean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/babymean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/babymean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/babymean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/babymean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &


# baby的数据集的参数设置
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/babygaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/babygaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/babygaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/babygaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/babygaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &



nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/babymlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/babymlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/babymlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/babymlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/babymlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &



nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/babynongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/babynongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/babynongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/babynongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/babynongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &




nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/babyuncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/babyuncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/babyuncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/babyuncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/babyuncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &






nohup python main.py -m Regression -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/babyregression-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m Regression -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/babyregression-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m Regression -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/babyregression-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m Regression -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/babyregression-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m Regression -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/babyregression-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &


# clothing的数据集的参数设置
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 4 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 4 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 6 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 6 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &



# baby的数据集的参数设置
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/clothingmean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/babymean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/babymean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/babymean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/babymean-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &


# baby的数据集的参数设置
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/clothinggaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/clothinggaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/clothinggaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/clothinggaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/clothinggaissian-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &



nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/clothinggaissianmlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/clothinggaissianmlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/clothinggaissianmlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/clothinggaissianmlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/clothinggaissianmlp-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &



nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/clothingnongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/clothingnongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/clothingnongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/clothingnongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/clothingnongraph-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &




nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 > ./Result/clothinguncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 > ./Result/clothinguncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 > ./Result/clothinguncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 > ./Result/clothinguncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 > ./Result/clothinguncondition-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &










# tiktok的数据集的参数设置
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.1 -c 4 > ./Result/tiktokv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Result/tiktokv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.5 -c 4 > ./Result/tiktokv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.7 -c 6 > ./Result/tiktokv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.9 -c 4 > ./Result/tiktokv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &



nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 4 > ./Result/babyv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Result/babyv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 4 > ./Result/babyv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 6 > ./Result/babyv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 4 > ./Result/babyv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &


nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 4 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing1.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 6 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing3.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 4 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing5.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 6 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing7.log  2>&1 &
nohup python main.py -m GenerativeAlignment -d clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 4 > ./Result/clothingv2-mask_weight_f_1e-2-mask_weight_g-1e-2-missing9.log  2>&1 &

