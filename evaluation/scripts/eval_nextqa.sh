CUDA_VISIBLE_DEVICES=1 python eval_nextqa.py --model_path your_path/AoTD-7B \
                                              --data_dir your_path/val.csv \
                                              --video_dir your_path/NExTVideo \
                                              --save_dir your_path/result.json