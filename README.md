# mmpretrain_Jetson
fineturing the EfficientNet model and deploy on Jetson

## 0x00 整体总览
本项目选用mmpretrain的EfficientNet进行微调，并部署在Jetson Orin开发板上

本次项目选用个人私人云服务器进行开发
## 0x01 环境安装
- clone mmpretrain && install
源码安装：
```bash
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -U openmim && mim install -e .
```
- pip安装
```bash
pip install -U openmim && mim install "mmpretrain>=1.0.0rc8"
```
- 测试环境是否安装成功(如果发现上述方法安装不成功，建议pip再安装一次)
```python
from mmpretrain import get_model, inference_model

model = get_model('resnet18_8xb32_in1k', device='cpu')  # 或者 device='cuda:0'
inference_model(model, 'demo/demo.JPEG')
```
## 0x02 数据集分析
选取公开交通牌数据集进行微调。 8：1：1安排训练，测试，验证集。 这个数据集共有58类
- 数据集分析

    dataset/
    
    ├── images
    
    │   ├── xxx.png
    
    │   ├── xxy.png
    
    │   └── ...
    
    └── annotations.csv
    
    这是下载后的数据集样式，但是我们需要转换成mmpretrain支持训练的两种基本格式之一(子文件夹方式/标注文件方式)。
    
    具体细节请看mmpretrain官方文档准备数据集:  [链接](https://mmpretrain.readthedocs.io/zh_CN/latest/user_guides/dataset_prepare.html)
    
    我们这里采用标注文件的方式，具体需要转换成如下格式所示

    dataset
    
    ├── meta
    
    │   ├── test.txt     # 测试数据集的标注文件
    
    │   ├── train.txt    # 训练数据集的标注文件
    
    │   └── val.txt      # 验证数据集的标注文件  
    
    └── images

    转换脚本请参见 [split_images.ipynb](https://github.com/yinfan98/mmpretrain_Jetson/blob/main/split_images.ipynb)
- 数据集中图片展示

![数据集图片](https://github.com/yinfan98/mmpretrain_Jetson/blob/main/000_1_0030.png)

- config文件解析
当按照我们这样排布数据集时，我们需要修改config文件。我这里简单介绍config需要修改的位置，若有训练需求可以自取。
```python
# 首先我们需要加载一个预训练好的EfficientNet-b0
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpt/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=58),
)
# 这里修改的是训练验证函数
train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/hy-tmp', # 根目录位置 也就是上面树状图中dataset位置
        ann_file='/hy-tmp/datasets/meta/train.txt', #训练标注文件位置
        data_prefix='',
        classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58'],
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/hy-tmp',
        ann_file='/hy-tmp/datasets/meta/val.txt',
        data_prefix='',
        classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58'],
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))
# 训练器设置，这里是微调，就把学习率降低了一个数量级。这里也可以试试Adam等优化策略
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
```
## 0x03 fine-tune效果展示
**todo 插入训练后结果**
能从上图看出，在fine-tune之后，准确率有较高提升
## 0x04 转换
在[deploee](https://platform.openmmlab.com/deploee/task-convert-list)网站上进行模型转换
![模型转换](https://github.com/yinfan98/mmpretrain_Jetson/blob/main/model_convert.PNG)

- 算法中选择mmpretrain算法
- 训练配置文件路径选择自己的配置文件。配置文件有需要可以自取
    需要注意的是，这里训练配置文件必须是全部展开后的配置文件，也就是说不能出现类似以下代码
    ```python
    _base_ = [
        'efficientnet_b0.py',
        'imagenet_bs32.py',
        'imagenet_bs256.py',
        'default_runtime.py',
    ]
    ```
- 目标runtime选择Jetson Orin
- 测试数据上传一张数据集中图片即可。

转换后即可下载模型，然后开始模型测速。
## 0x04 测速
![模型测速](https://github.com/yinfan98/mmpretrain_Jetson/blob/main/speed_test.PNG)

- 任务类型选择mmpretrain
- 模型上传模型的zip文件
- 测速数据上传一张测试图片
- 测速设备选择Jetson Orin
    得到以下测速结果

```text
========== stdout ==========
[2023-08-23 22:03:57.359] [mmdeploy] [info] [model.cpp:35] [DirectoryModel] Load model: "/tmp/datadir/jetson_orin_profiler/4591aa"
label: 0, label_id: 0, score: 0.9996
label: 1, label_id: 4, score: 0.0002
label: 2, label_id: 1, score: 0.0001
label: 3, label_id: 11, score: 0.0000
label: 4, label_id: 24, score: 0.0000
========== stderr ==========
None
========== analyze ==========
+---------------------------+--------+-------+--------+--------+-------+-------+
|           name            | occupy | usage | n_call | t_mean | t_50% | t_90% |
+===========================+========+=======+========+========+=======+=======+
| ./Pipeline                | -      | -     | 21     | 44.378 | 3.042 | 3.061 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|     Preprocess/Compose    | -      | -     | 21     | 0.246  | 0.204 | 0.242 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         LoadImageFromFile | 0.001  | 0.001 | 21     | 0.039  | 0.025 | 0.048 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         Resize            | 0.001  | 0.001 | 21     | 0.048  | 0.044 | 0.047 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         CenterCrop        | 0.001  | 0.001 | 21     | 0.039  | 0.040 | 0.042 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         Normalize         | 0.001  | 0.001 | 21     | 0.042  | 0.041 | 0.045 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         ImageToTensor     | 0.001  | 0.001 | 21     | 0.063  | 0.034 | 0.038 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|         Collect           | 0.000  | 0.000 | 21     | 0.011  | 0.010 | 0.011 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|     efficientnet          | 0.993  | 0.993 | 21     | 44.063 | 2.771 | 2.780 |
+---------------------------+--------+-------+--------+--------+-------+-------+
|     postprocess           | 0.001  | 0.001 | 21     | 0.050  | 0.049 | 0.054 |
+---------------------------+--------+-------+--------+--------+-------+-------+
========== megpeak CPU profiling ==========
there are 12 cores, currently use core id :0
Vendor is: ARM, uArch: unknown, frequency: 0Hz

bandwidth: 18.958020 Gbps
nop throughput: 0.046433 ns 21.536610 GFlops latency: 0.060893 ns :
ldd throughput: 0.152348 ns 13.127861 GFlops latency: 0.152118 ns :
ldq throughput: 0.152234 ns 26.275372 GFlops latency: 0.152112 ns :
stq throughput: 0.229189 ns 17.452873 GFlops latency: 0.229173 ns :
ldpq throughput: 0.304489 ns 26.273569 GFlops latency: 0.304341 ns :
lddx2 throughput: 0.228379 ns 17.514774 GFlops latency: 0.230601 ns :
ld1q throughput: 0.152236 ns 26.275038 GFlops latency: 0.152116 ns :
eor throughput: 0.228299 ns 17.520906 GFlops latency: 0.913028 ns :
fmla throughput: 0.228263 ns 35.047352 GFlops latency: 1.830289 ns :
fmlad throughput: 0.228303 ns 17.520605 GFlops latency: 1.827763 ns :
fmla_x2 throughput: 0.467541 ns 34.221565 GFlops latency: 3.675978 ns :
mla throughput: 0.456499 ns 17.524673 GFlops latency: 1.829411 ns :
fmul throughput: 0.228291 ns 17.521524 GFlops latency: 1.371082 ns :
mul throughput: 0.456715 ns 8.758192 GFlops latency: 1.841269 ns :
addp throughput: 0.228363 ns 17.516001 GFlops latency: 0.915103 ns :
sadalp throughput: 0.456589 ns 8.760610 GFlops latency: 1.828035 ns :
add throughput: 0.228125 ns 17.534275 GFlops latency: 0.914881 ns :
fadd throughput: 0.228245 ns 17.525057 GFlops latency: 0.912938 ns :
smull throughput: 0.456355 ns 8.765101 GFlops latency: 1.833639 ns :
smlal_4b throughput: 0.458533 ns 17.446936 GFlops latency: 1.840173 ns :
smlal_8b throughput: 0.480294 ns 33.312958 GFlops latency: 1.850761 ns :
dupd_lane_s8 throughput: 0.228325 ns 35.037834 GFlops latency: 0.915127 ns :
mlaq_lane_s16 throughput: 0.456507 ns 35.048782 GFlops latency: 1.830033 ns :
sshll throughput: 0.456441 ns 17.526899 GFlops latency: 0.912775 ns :
tbl throughput: 0.228233 ns 70.103912 GFlops latency: 0.915910 ns :
ins throughput: 0.456479 ns 4.381360 GFlops latency: 1.115577 ns :
sqrdmulh throughput: 0.472065 ns 8.473401 GFlops latency: 1.866842 ns :
usubl throughput: 0.230361 ns 17.364079 GFlops latency: 0.915229 ns :
abs throughput: 0.228223 ns 17.526747 GFlops latency: 0.912941 ns :
fcvtzs throughput: 0.914873 ns 4.372194 GFlops latency: 1.827603 ns :
scvtf throughput: 0.915296 ns 4.370169 GFlops latency: 1.828205 ns :
fcvtns throughput: 0.947023 ns 4.223763 GFlops latency: 1.830869 ns :
fcvtms throughput: 0.913081 ns 4.380774 GFlops latency: 1.827703 ns :
fcvtps throughput: 0.914997 ns 4.371601 GFlops latency: 1.827285 ns :
fcvtas throughput: 0.913522 ns 4.378655 GFlops latency: 1.867723 ns :
fcvtn throughput: 0.916999 ns 4.362057 GFlops latency: 1.827859 ns :
fcvtl throughput: 0.912731 ns 4.382454 GFlops latency: 1.827561 ns :
prefetch_very_long throughput: 13.660245 ns 0.292821 GFlops latency: 0.152268 ns :
ins_ldd throughput: 0.464063 ns 4.309762 GFlops latency: 0.456615 ns :Test ldd ins dual issue
ldd_ldx_ins throughput: 1.104095 ns 3.622878 GFlops latency: 0.456455 ns :
ldqstq throughput: 2.922742 ns 1.368578 GFlops latency: 2.899029 ns :Test ldq stq dual issue
ldq_fmlaq throughput: 0.228327 ns 35.037525 GFlops latency: 0.228165 ns :
stq_fmlaq_lane throughput: 0.304502 ns 26.272392 GFlops latency: 2.284634 ns :Test stq fmlaq_lane dual issue
ldd_fmlad throughput: 0.228167 ns 17.531048 GFlops latency: 0.228341 ns :Test ldd fmlad dual issue
ldq_fmlaq_sep throughput: 0.228325 ns 35.037762 GFlops latency: 1.827905 ns :Test throughput ldq + 2 x fmlaq
ldq_fmlaq_lane_sep throughput: 0.237275 ns 33.716118 GFlops latency: 2.300703 ns :Test compute throughput ldq + 2 x fmlaq_lane
ldd_fmlaq_sep throughput: 0.228333 ns 35.036613 GFlops latency: 1.827823 ns :Test compute throughput ldq + fmlaq
lds_fmlaq_lane_sep throughput: 0.228303 ns 35.041210 GFlops latency: 2.287084 ns :
ldd_fmlaq_lane_sep throughput: 0.228359 ns 35.032616 GFlops latency: 2.318689 ns :Test compute throughput ldd + fmlaq_lane
ldx_fmlaq_lane_sep throughput: 0.228353 ns 35.033539 GFlops latency: 2.332591 ns :
ldd_ldx_ins_fmlaq_lane_sep throughput: 0.379272 ns 21.093048 GFlops latency: 2.287856 ns :Test compute throughput ldd+fmlaq+ldx+fmlaq+ins+fmlaq
ldd_nop_ldx_ins_fmlaq_lane_sep throughput: 0.343434 ns 23.294142 GFlops latency: 2.286804 ns :
ins_fmlaq_lane_1_4_sep throughput: 0.405852 ns 19.711609 GFlops latency: 3.760201 ns :Test compute throughput ins + 4 x fmlaq_lane
ldd_fmlaq_lane_1_4_sep throughput: 0.243628 ns 32.836971 GFlops latency: 2.287016 ns :Test compute throughput ldd + 4 x fmlaq_lane
ldq_fmlaq_lane_1_4_sep throughput: 0.228340 ns 35.035461 GFlops latency: 0.228173 ns :Test compute throughput ldq + 4 x fmlaq_lane
ins_fmlaq_lane_1_3_sep throughput: 0.404522 ns 19.776428 GFlops latency: 3.767939 ns :Test compute throughput ins + 3 x fmlaq_lane
ldd_fmlaq_lane_1_3_sep throughput: 0.384988 ns 20.779846 GFlops latency: 3.738999 ns :
ldq_fmlaq_lane_1_3_sep throughput: 0.228464 ns 35.016472 GFlops latency: 0.230189 ns :Test compute throughput ldq + 3 x fmlaq_lane
ldq_fmlaq_lane_1_2_sep throughput: 0.228360 ns 35.032379 GFlops latency: 0.228303 ns :Test compute throughput ldq + 2 x fmlaq_lane
ins_fmlaq_lane_sep throughput: 1.163445 ns 6.876129 GFlops latency: 2.284408 ns :
dupd_fmlaq_lane_sep throughput: 0.686848 ns 11.647411 GFlops latency: 2.297210 ns :
smlal_8b_addp throughput: 0.458385 ns 34.905136 GFlops latency: 3.232407 ns :
smlal_8b_dupd throughput: 0.456357 ns 35.060253 GFlops latency: 1.828027 ns :
ldd_smlalq_sep_8b throughput: 0.458723 ns 34.879414 GFlops latency: 0.462053 ns :Test ldd smlalq dual issue
ldq_smlalq_sep throughput: 0.456663 ns 35.036758 GFlops latency: 0.458535 ns :Test ldq smlalq dual issue
lddx2_smlalq_sep throughput: 0.456581 ns 35.043045 GFlops latency: 0.456503 ns :
smlal_sadalp throughput: 0.458595 ns 34.889153 GFlops latency: 3.700649 ns :
smull_smlal_sadalp throughput: 0.915211 ns 34.964634 GFlops latency: 5.481531 ns :Test smull smlal dual issue
smull_smlal_sadalp_sep throughput: 0.456479 ns 35.050926 GFlops latency: 5.529644 ns :
ins_smlalq_sep_1_2 throughput: 0.588847 ns 27.171757 GFlops latency: 3.324252 ns :
ldx_ins_smlalq_sep throughput: 0.456678 ns 35.035648 GFlops latency: 3.439042 ns :
dupd_lane_smlal_s8 throughput: 0.459077 ns 34.852520 GFlops latency: 3.199949 ns :
ldd_mla_s16_lane_1_4_sep throughput: 0.456810 ns 35.025478 GFlops latency: 0.456497 ns :
ldrd_sshll throughput: 0.456533 ns 17.523367 GFlops latency: 0.458355 ns :
```
## 0x05 补充说明
mmdeploy不支持EfficientRandomCrop和EfficientCenterCrop，需要修改成CenterCrop和RandomResizedCrop进行测速操作。
个人正在实现EfficientRandomCrop和EfficientCenterCrop算子，即将PR
