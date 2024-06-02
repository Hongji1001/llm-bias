export OMP_NUM_THREADS=120
###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-06-01 17:43:15
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-06-01 17:43:24
 # @FilePath: /llm-bias/running_scripts/bookcorpus.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 
# python generation_dataset --dataset=cnn_dailymail
python generation_dataset.py --dataset=bookcorpus