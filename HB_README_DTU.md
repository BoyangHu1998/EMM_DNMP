
# Use the code with DTU

## 1. Convert images to point cloud

``` bash
python /home/hb/Desktop/dtu_dataset_samples/dtu_dataset.py
```

<!-- 
preprocess_kitti.py:
````bash 
python preprocess/preprocess_dtu.py --root_dir "/home/hb/Desktop/dtu_dataset_samples/dtu" --save_dir "/home/hb/Documents/DNMP-main/dtu_data"
```` -->

## 2. 
``` bash
python preprocess/preprocess_dtu.py --root_dir "/home/hb/Desktop/dtu_dataset_samples/dtu" --save_dir "/home/hb/Documents/DNMP-main/dtu_data"
```

## 3. preprocess/run_metashape_kitti.sh:  
inputs:
- SEQUENCE='seq_1'
- PROJECT_ROOT='/home/hb/Documents/DNMP-main'
- ROOT_DIR='/home/hb/Documents/DNMP-main/data'