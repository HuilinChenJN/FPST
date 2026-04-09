# Progressive Semantic Transport via Flow Matching for Missing Modality Completion in Recommender Systems


## Overview
Missing modalities are prevalent in real-world multimodal recommender systems, posing significant challenges to reliable preference modeling. Existing methods often address this issue by reconstructing or imputing missing modalities, with generative completion being a common paradigm. However, empirical evidence shows that improved reconstruction quality does not consistently lead to better ranking performance, revealing a fundamental misalignment between reconstruction objectives and recommendation goals. Specifically, reconstruction-oriented learning focuses on recovering raw features, which may overlook preference-relevant semantics critical for recommendation. To address this issue, we propose a Flow-based Progressive Semantic Transport framework (FPST), which reformulates modality completion as a preference-driven alignment process rather than feature reconstruction. Specifically, FPST models this process as a continuous transport, progressively transforming representations from observed modality latents toward targets grounded in collaborative interaction signals. To improve robustness under varying missing patterns, where available modalities differ across items, we first construct a stable source distribution via a variational encoder with product-of-experts aggregation. We then parameterize preference-conditioned velocity fields to model a continuous transport trajectory, enabling representations to gradually align with collaborative signals while preserving item-specific semantics. Extensive experiments on Amazon and TikTok datasets demonstrate that FPST consistently outperforms state-of-the-art methods, achieving up to 7% improvement in Recall@20 and maintaining robust performance across missing modality rates ranging from 10% to 90%. These results highlight the importance of aligning modality completion with recommendation goals, and suggest that preference-aware semantic transport provides a more effective alternative to reconstruction-based approaches.

## Environment
Run `bash environment.sh` To Install Conda Environment


## Dataset

Download from Google Drive: [Baby/Clothing](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing) from [MMRec](https://github.com/enoche/MMRec).
The data already contains text and image features extracted from Sentence-Transformers and CNN, which is provided by [MMRec](https://github.com/enoche/MMRec).
Please move your downloaded data into the 'Data' folder for model training.



## Training / Test for Missing Modality Setting on Baby
Note that: `--missing_rate` is defined as missing modality rate, modifying it to change the missing rate of modaliy feature for training data.


``` bash
python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 

python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 

python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 

python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 

python main.py -m GenerativeAlignment -d baby --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 
```


## Training / Test for Missing Modality Setting on Clothing

``` bash
python main.py -m GenerativeAlignment -d Clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 

python main.py -m GenerativeAlignment -d Clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 

python main.py -m GenerativeAlignment -d Clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 

python main.py -m GenerativeAlignment -d Clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 

python main.py -m GenerativeAlignment -d Clothing --mask_weight_f 1e-2 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 
```


## Training / Test for Missing Modality Setting on Tiktok

``` bash
python main.py -m GenerativeAlignment -d Tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.1 -c 0 

python main.py -m GenerativeAlignment -d Tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.3 -c 2 

python main.py -m GenerativeAlignment -d Tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.5 -c 3 

python main.py -m GenerativeAlignment -d Tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.7 -c 4 

python main.py -m GenerativeAlignment -d Tiktok --mask_weight_f 1e-3 --mask_weight_g 1e-2 --missing_rate 0.9 -c 2 
```

# Performance Result

| Datasets | Baby Recall@20 | Baby NDCG@20 | Baby Recall@50 | Baby NDCG@50 | Clothing Recall@20 | Clothing NDCG@20 | Clothing Recall@50 | Clothing NDCG@50 | Tiktok Recall@20 | Tiktok NDCG@20 | Tiktok Recall@50 | Tiktok NDCG@50 |
|----------|----------------|---------------|----------------|---------------|-------------------|------------------|-------------------|-----------------|-----------------|----------------|-----------------|----------------|
| MFBPR    | 0.0602         | 0.0267        | 0.1087         | 0.0372        | 0.0346            | 0.0162           | 0.0533            | 0.0201          | 0.0558          | 0.0220         | 0.0909          | 0.0289         |
| LightGCN | 0.0733         | 0.0321        | 0.1324         | 0.0437        | 0.0514            | 0.0214           | 0.0802            | 0.0267          | 0.0916          | 0.0393         | 0.1572          | 0.0536         |
| SimGCL   | 0.0795         | 0.0331        | 0.1395         | 0.0469        | 0.0546            | 0.0255           | 0.0832            | 0.0302          | 0.0954          | 0.0403         | 0.1453          | 0.0509         |
| VBPR     | 0.0487         | 0.0205        | 0.0936         | 0.0300        | 0.0462            | 0.0207           | 0.0737            | 0.0226          | 0.0406          | 0.0170         | 0.0699          | 0.0229         |
| MMGCN    | 0.0527         | 0.0218        | 0.0992         | 0.0310        | 0.0289            | 0.0120           | 0.0530            | 0.0168          | 0.0874          | 0.0368         | 0.1431          | 0.0484         |
| GRCN     | 0.0632         | 0.0275        | 0.1147         | 0.0376        | 0.0381            | 0.0161           | 0.0644            | 0.0214          | 0.0709          | 0.0280         | 0.1257          | 0.0389         |
| BM3      | 0.0683         | 0.0296        | 0.1235         | 0.0408        | 0.0592            | 0.0252           | 0.0920            | 0.0334          | 0.0760          | 0.0319         | 0.1215          | 0.0409         |
| LATTICE  | 0.0732         | 0.0312        | 0.1296         | 0.0432        | 0.0581            | 0.0215           | 0.0929            | 0.0332          | 0.0824          | 0.0372         | 0.1353          | 0.0477         |
| MGCN     | 0.0826         | 0.0363        | 0.1389         | 0.0481        | 0.0665            | 0.0268           | 0.1052            | 0.0377          | 0.0867          | 0.0352         | 0.1400          | 0.0460         |
| SMORE    | 0.0821         | 0.0361        | 0.1399         | 0.0477        | 0.0610            | 0.0278           | 0.0978            | 0.0356          | 0.0903          | 0.0372         | 0.1526          | 0.0479         |
| GUME     | 0.0832         | 0.0401        | 0.1427         | 0.0487        | 0.0639            | 0.0291           | 0.1016            | 0.0366          | 0.0968          | 0.0389         | 0.1645          | 0.0524         |
| MILK     | 0.0415         | 0.0184        | 0.0763         | 0.0250        | 0.0226            | 0.0090           | 0.0376            | 0.0124          | 0.0404          | 0.0184         | 0.0640          | 0.0230         |
| SIBRAR   | 0.0472         | 0.0217        | 0.0889         | 0.0289        | 0.0264            | 0.0110           | 0.0453            | 0.0148          | 0.0548          | 0.0218         | 0.0854          | 0.0280         |
| MoDiCF   | 0.0798         | 0.0348        | 0.1297         | 0.0475        | 0.0602            | 0.0273           | 0.1062            | 0.0326          | 0.0902          | 0.0385         | 0.1472          | 0.0493         |
| I³-MRec  | 0.0815         | 0.0361        | 0.1475         | 0.0496        | 0.0635            | 0.0285           | 0.1074            | 0.0375          | 0.0929          | 0.0398         | 0.1484          | 0.0479         |
| DGMRec   | 0.0892         | 0.0400        | 0.1527         | 0.0531        | 0.0738            | 0.0327           | 0.1133            | 0.0405          | 0.1012          | 0.0453         | 0.1683          | 0.0583         |
| **Ours** | **0.0932**     | **0.0411**    | **0.1610**     | **0.0548**    | **0.0803**        | **0.0348**       | **0.1203**        | **0.0431**      | **0.1081**      | **0.0475**     | **0.1768**      | **0.0610**     |
| Improv.  | **4.48%**      | **2.24%**     | **5.44%**      | **3.20%**     | **8.81%**         | **6.42%**        | **6.17%**         | **6.16%**       | **6.82%**       | **4.85%**      | **5.05%**       | **4.63%**      |