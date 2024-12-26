# 基于 OCL 的物体类别分类推理实验

- 使用 Clip api 能够跑通， SigLIP api 的 accuracy 还很低，还没 debug。
- 主要代码就是在 `dataset.py` 和 `model.py` 中，evaluation 使用 `test_model.py`。
    - 对于 `dataset.py` 的测试在 `test_dataset.py` 中。
    - 一些小数据集的测试和可视化在 `test_mAP` 中。
    - 检验环境是否配置正确在 `test_env_Clip.py` 中。

## 两个主要模块和测试的介绍

### OCLDataset Class

#### Class Initialization
```python
def __init__(self, root_dir, split='test', top_k_categories=10):
```
- 初始化数据集类，接收三个参数:
  - `root_dir`: OCL数据集的根目录路径
  - `split`: 数据集划分，默认为'test'
  - `top_k_categories`: 保留样本数最多的前k个类别，默认为10
- 在初始化过程中:
  1. 配置日志记录器用于信息输出
  2. 加载CLIP模型和预处理器
  3. 从pickle文件加载数据集标注信息
  4. 加载类别信息并进行过滤

#### 文本嵌入预计算
```python
def precompute_text_embeddings(self):
```
使用 Clip 模型对 objects, attributes, affordance 进行 tokenize 和 embedding


#### 类别信息加载
```python
def load_class_info(self, pkl_dir):
```
- 从JSON文件中加载三类信息:
  1. 属性(attributes)列表
  2. 对象(objects)类别列表 
  3. 效能(affordances)列表
- 创建对象到索引的映射字典
- 加载类别-效能关系矩阵

#### 类别过滤
```python
def filter_top_categories(self):
```
- 统计每个类别的样本数量
- 选择样本数最多的前K个类别
- 过滤数据集标注，只保留这K个类别的样本
- 为过滤后的类别创建新的索引映射

#### 数据样本获取
```python
def __getitem__(self, idx):
```
- 加载并预处理图像:
  1. 读取图像文件并转换为RGB
  2. 对大尺寸图像进行缩放
  3. 应用CLIP的预处理
- 提取标签信息:
  1. 对象类别
  2. 属性标签(多热编码)
  3. 效能标签(多热编码)
- 返回包含处理后图像和标签的字典


### OCLModel Class

#### 模型初始化
```python
def __init__(self, model_name="ViT-B/32"):
```
- 类初始化主要完成以下工作:
  1. 配置日志记录器用于记录运行信息
  2. 自动选择设备(GPU/CPU)
  3. 加载指定的CLIP模型
  4. 将模型设置为评估模式(eval mode)
  5. 冻结所有模型参数
- 异常处理:
  - 捕获并记录模型初始化过程中的任何错误

#### 参数冻结和特征提取
```python
def _freeze_parameters(self):
    for param in self.model.parameters():
        param.requires_grad = False
        
def get_embeddings(self, batch):
```
- 参数冻结:
  - 遍历模型所有参数并设置requires_grad=False
  - 目的是防止在推理过程中更新模型参数
- 特征提取:
  1. 将输入图像批次移至指定设备
  2. 使用no_grad上下文管理器进行推理
  3. 提取图像特征并进行L2归一化
  4. 返回归一化后的特征向量

#### 相似度和准确率计算
```python
def compute_similarity(self, image_features, text_features):
    similarity = torch.mm(image_features, text_features.t()) / 0.06
    return similarity
def compute_accuracy(self, similarity, targets, k=1):
```
- 相似度计算:
  - 使用矩阵乘法计算图像和文本特征的相似度
  - 应用温度系数(0.06)进行缩放
- 准确率计算:
  1. 获取top-k预测结果
  2. 将预测结果与目标标签比较
  3. 计算正确预测的比例
  4. 记录每个batch的准确率

#### 模型评估
```python
@torch.no_grad()
def evaluate(self, dataloader):
```
- 整体评估流程:
  1. 初始化各项指标的累积变量
  2. 加载数据集预计算的文本特征
  3. 遍历数据加载器的每个batch:
     - 提取图像特征
     - 计算类别分类的准确率(Top-1,5,10)
     - 计算属性和效能预测
     - 存储预测结果和标签
  4. 定期输出进度和临时结果
- 指标计算:
  - 合并所有batch的预测结果和标签
  - 计算属性和效能的mAP分数
  - 返回包含所有指标的字典:
    * top1_accuracy: 类别分类的Top-1准确率
    * top5_accuracy: Top-5准确率
    * top10_accuracy: Top-10准确率
    * attribute_map: 属性预测的mAP
    * affordance_map: 效能预测的mAP
    
#### 平均精度计算
```python
def compute_map(self, predictions, labels):
    aps = []
    for i in range(predictions.shape[1]):
        if labels[:, i].sum() > 0:
            ap = average_precision_score(labels[:, i], predictions[:, i])
            if not np.isnan(ap):
                aps.append(ap)
    mean_ap = np.mean(aps)
    return mean_ap
```
- mAP计算流程:
  1. 将预测和标签转换为NumPy数组
  2. 验证预测和标签的形状匹配
  3. 对每个类别:
     - 仅对有正样本的类别计算AP
     - 使用scikit-learn的average_precision_score
     - 过滤掉NaN结果
  4. 计算所有有效AP的平均值

- 错误处理:
  - 记录每个类别AP计算中的错误
  - 在没有有效AP时返回0.0
  - 输出debug级别的计算细节

### test_mAP 介绍

会计算 100 个样本，输出以下内容

#### 类别预测分析

- 展示Top-5预测的类别
- 显示每个类别的相似度得分
- 对比真实类别和预测结果

#### 属性预测分析

- 展示Top-10预测的属性
- 显示每个属性的相似度得分
- 列出真实属性标签及其预测相似度


#### affordance 预测分析

- 展示Top-10预测的 affordance
- 显示 affordace 的相似度得分和定义
- 对比真实 affordace 标签和预测结果


#### 综合性能指标

- 计算类别分类准确率
- 计算属性预测的mAP
- 计算效能预测的mAP

## 结果

### test_model

```
(OCL) /home/fanghaotian/miniconda3/envs/OCL/bin/python /home/fanghaotian/OCL_lab_2412/Clip/test_model.py
2024-12-26 22:39:49,863 - TestProgram - INFO - Starting test program...
2024-12-26 22:39:49,863 - TestProgram - INFO - Initial GPU memory: 0.00MB
2024-12-26 22:39:49,863 - TestProgram - INFO - Initializing model...
2024-12-26 22:39:49,863 - OCLModel - INFO - Initializing CLIP model ViT-B/32 on cuda
2024-12-26 22:39:52,779 - TestProgram - INFO - Model initialized successfully
2024-12-26 22:39:52,779 - TestProgram - INFO - Creating data loader...
2024-12-26 22:39:55,468 - OCLDataset - INFO - Loading annotations from /data/DATA/OCL_DATA/OCL_data/data/resources/OCL_annot_test.pkl
2024-12-26 22:39:56,172 - OCLDataset - INFO - Selected top 10 categories
2024-12-26 22:39:56,182 - OCLDataset - INFO - Filtered dataset contains 4151 images
2024-12-26 22:39:56,182 - OCLDataset - INFO - Precomputing text embeddings...
2024-12-26 22:39:56,386 - OCLDataset - INFO - Precomputed feature shapes:
2024-12-26 22:39:56,386 - OCLDataset - INFO -   Categories: torch.Size([10, 512])
2024-12-26 22:39:56,386 - OCLDataset - INFO -   Attributes: torch.Size([114, 512])
2024-12-26 22:39:56,386 - OCLDataset - INFO -   Affordances: torch.Size([170, 512])
2024-12-26 22:39:56,386 - TestProgram - INFO - Created loader with 346 batches
2024-12-26 22:39:56,386 - TestProgram - INFO - Starting evaluation...
2024-12-26 22:39:59,986 - OCLModel - INFO - Processed 50/346 batches. Running top-1: 0.8150, top-5: 0.9917, top-10: 1.0000
2024-12-26 22:40:03,134 - OCLModel - INFO - Processed 100/346 batches. Running top-1: 0.7850, top-5: 0.9933, top-10: 1.0000
2024-12-26 22:40:05,884 - OCLModel - INFO - Processed 150/346 batches. Running top-1: 0.7894, top-5: 0.9911, top-10: 1.0000
2024-12-26 22:40:07,873 - OCLModel - INFO - Processed 200/346 batches. Running top-1: 0.8033, top-5: 0.9917, top-10: 1.0000
2024-12-26 22:40:10,653 - OCLModel - INFO - Processed 250/346 batches. Running top-1: 0.7923, top-5: 0.9897, top-10: 1.0000
2024-12-26 22:40:13,447 - OCLModel - INFO - Processed 300/346 batches. Running top-1: 0.7808, top-5: 0.9878, top-10: 1.0000
/home/fanghaotian/miniconda3/envs/OCL/lib/python3.8/site-packages/PIL/Image.py:1056: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
2024-12-26 22:40:18,611 - OCLModel - INFO - all_attr_labels: torch.Size([4151, 114]), all_attr_preds: torch.Size([4151, 114])
2024-12-26 22:40:18,611 - OCLModel - INFO - all_aff_labels: torch.Size([4151, 170]), all_aff_preds: torch.Size([4151, 170])
2024-12-26 22:40:19,043 - TestProgram - INFO - Test Results:
2024-12-26 22:40:19,044 - TestProgram - INFO - --------------------------------------------------
2024-12-26 22:40:19,044 - TestProgram - INFO - Top-1 Accuracy: 0.8020
2024-12-26 22:40:19,044 - TestProgram - INFO - Top-5 Accuracy: 0.9894
2024-12-26 22:40:19,044 - TestProgram - INFO - Top-10 Accuracy: 1.0000
2024-12-26 22:40:19,044 - TestProgram - INFO - Attribute mAP: 0.1577
2024-12-26 22:40:19,044 - TestProgram - INFO - Affordance mAP: 0.3287
2024-12-26 22:40:19,044 - TestProgram - INFO - --------------------------------------------------
2024-12-26 22:40:19,177 - TestProgram - INFO - Final GPU memory: 695.68MB
```

### test_mAP

```
(OCL) /home/fanghaotian/miniconda3/envs/OCL/bin/python /home/fanghaotian/OCL_lab_2412/Clip/test_mAP.py
2024-12-26 22:39:08,701 - DemoProgram - INFO - Starting demo evaluation...
2024-12-26 22:39:08,701 - DemoProgram - INFO - Using device: cuda
2024-12-26 22:39:11,577 - OCLDataset - INFO - Loading annotations from /data/DATA/OCL_DATA/OCL_data/data/resources/OCL_annot_test.pkl
2024-12-26 22:39:12,154 - OCLDataset - INFO - Selected top 10 categories
2024-12-26 22:39:12,163 - OCLDataset - INFO - Filtered dataset contains 4151 images
2024-12-26 22:39:12,163 - OCLDataset - INFO - Precomputing text embeddings...
2024-12-26 22:39:12,391 - OCLDataset - INFO - Precomputed feature shapes:
2024-12-26 22:39:12,391 - OCLDataset - INFO -   Categories: torch.Size([10, 512])
2024-12-26 22:39:12,391 - OCLDataset - INFO -   Attributes: torch.Size([114, 512])
2024-12-26 22:39:12,391 - OCLDataset - INFO -   Affordances: torch.Size([170, 512])
2024-12-26 22:39:12,391 - DemoProgram - INFO - Created small loader with 25 batches
2024-12-26 22:39:12,391 - OCLModel - INFO - Initializing CLIP model ViT-B/32 on cuda
2024-12-26 22:39:15,333 - DemoProgram - INFO - 
=== Sample 1 ===
2024-12-26 22:39:15,333 - DemoProgram - INFO - 
Top 5 predicted categories with similarities:
2024-12-26 22:39:15,343 - DemoProgram - INFO - Ground truth category: boat
2024-12-26 22:39:15,343 - DemoProgram - INFO - Category: boat            Similarity:   4.0625
2024-12-26 22:39:15,344 - DemoProgram - INFO - Category: horse           Similarity:   3.2266
2024-12-26 22:39:15,344 - DemoProgram - INFO - Category: dog             Similarity:   3.1328
2024-12-26 22:39:15,344 - DemoProgram - INFO - Category: car             Similarity:   3.1094
2024-12-26 22:39:15,344 - DemoProgram - INFO - Category: motorcycle      Similarity:   2.9766
2024-12-26 22:39:15,344 - DemoProgram - INFO - 
Top 10 predicted attributes with similarities:
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: narrow          Similarity:   3.8945
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: long            Similarity:   3.7539
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: arched          Similarity:   3.6992
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: framed          Similarity:   3.6719
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: reflective      Similarity:   3.6660
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: standing        Similarity:   3.6602
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: hanging         Similarity:   3.6582
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: in the picture  Similarity:   3.6445
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: vertical        Similarity:   3.6426
2024-12-26 22:39:15,344 - DemoProgram - INFO - Attribute: horn            Similarity:   3.6289
2024-12-26 22:39:15,345 - DemoProgram - INFO - 
Ground truth attributes with similarities:
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: wet             Similarity:   3.2637
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: cool            Similarity:   3.4043
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: gray            Similarity:   3.3301
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: wooden          Similarity:   3.5039
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: small           Similarity:   3.4941
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: old             Similarity:   3.6133
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: hard            Similarity:   3.2852
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: heavy           Similarity:   2.3242
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: new             Similarity:   3.4688
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: dry             Similarity:   3.3750
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: whole           Similarity:   3.3418
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: white           Similarity:   3.1699
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: empty           Similarity:   3.5273
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: flat            Similarity:   3.3750
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: solid           Similarity:   3.3496
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: smooth          Similarity:   3.3105
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: curved          Similarity:   3.4609
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: horizontal      Similarity:   3.4590
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: short           Similarity:   3.4102
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: narrow          Similarity:   3.8945
2024-12-26 22:39:15,345 - DemoProgram - INFO - Attribute: parked          Similarity:   3.3125
2024-12-26 22:39:15,346 - DemoProgram - INFO - 
Top 10 predicted affordances with similarities:
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: row             Similarity:   4.5586 Definition: propel with oars
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: launch          Similarity:   4.3242 Definition: to put (a boat or ship) on the water
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: sweep           Similarity:   3.9434 Definition: move with sweeping, effortless, gliding motions
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: carry           Similarity:   3.9004 Definition: have with oneself; have on one's person
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: launch          Similarity:   3.8906 Definition: propel with force
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: walk            Similarity:   3.8828 Definition: use one's feet to advance; advance by steps
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: set up          Similarity:   3.8613 Definition: get ready for a particular purpose or event
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: shoot           Similarity:   3.8574 Definition: send forth suddenly, intensely, swiftly
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: smell           Similarity:   3.8496 Definition: inhale the odor of; perceive by the olfactory sense
2024-12-26 22:39:15,346 - DemoProgram - INFO - Affordance: straddle        Similarity:   3.7637 Definition: sit or stand astride of
2024-12-26 22:39:15,346 - DemoProgram - INFO - 
Ground truth affordances with similarities:
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: assemble        Similarity:   3.0078
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: board           Similarity:   3.5039
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: break           Similarity:   3.0254
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: brush           Similarity:   3.4785
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: check           Similarity:   3.0527
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: clean           Similarity:   3.1426
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: climb           Similarity:   3.3379
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: drag            Similarity:   3.5566
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: draw            Similarity:   2.9805
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: drive           Similarity:   3.1602
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: edit            Similarity:   3.2090
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: exit            Similarity:   3.0137
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: function        Similarity:   3.4062
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: hit             Similarity:   2.7461
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: hit             Similarity:   3.4395
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: hop on          Similarity:   2.9980
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: inspect         Similarity:   3.4082
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: install         Similarity:   3.3691
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: jump            Similarity:   3.2773
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: kick            Similarity:   3.2363
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: lie             Similarity:   3.7109
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: lie             Similarity:   3.5664
2024-12-26 22:39:15,347 - DemoProgram - INFO - Affordance: lie down        Similarity:   3.2656
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: lift            Similarity:   3.5195
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: load            Similarity:   3.6289
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: move            Similarity:   3.4883
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: pack            Similarity:   3.3594
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: paint           Similarity:   3.4004
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: photograph      Similarity:   3.6992
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: produce         Similarity:   3.4824
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: pull            Similarity:   3.5547
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: push            Similarity:   3.3848
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: push            Similarity:   3.5156
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: put             Similarity:   3.2695
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: raise           Similarity:   3.2715
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: repair          Similarity:   2.9434
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: shoot           Similarity:   2.9668
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: sign            Similarity:   3.3027
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: sit             Similarity:   3.7285
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: sit             Similarity:   3.3613
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: stand           Similarity:   3.5176
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: stand           Similarity:   3.7598
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: stand           Similarity:   3.4785
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: steer           Similarity:   3.5371
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: sweep           Similarity:   3.9434
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: tie             Similarity:   3.5547
2024-12-26 22:39:15,348 - DemoProgram - INFO - Affordance: touch           Similarity:   3.4141
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: transport       Similarity:   3.2988
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: walk            Similarity:   3.8828
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: write           Similarity:   3.4727
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: jump            Similarity:   3.3574
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: park            Similarity:   3.4023
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: row             Similarity:   4.5586
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: set up          Similarity:   3.8613
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: stop            Similarity:   3.1445
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: watch           Similarity:   3.1270
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: watch           Similarity:   3.4727
2024-12-26 22:39:15,349 - DemoProgram - INFO - Affordance: tag             Similarity:   2.6875
2024-12-26 22:39:15,349 - DemoProgram - INFO - 
=== Sample 2 ===
2024-12-26 22:39:15,349 - DemoProgram - INFO - 
Top 5 predicted categories with similarities:
2024-12-26 22:39:15,349 - DemoProgram - INFO - Ground truth category: boat
2024-12-26 22:39:15,349 - DemoProgram - INFO - Category: boat            Similarity:   4.2617
2024-12-26 22:39:15,349 - DemoProgram - INFO - Category: car             Similarity:   3.5723
2024-12-26 22:39:15,349 - DemoProgram - INFO - Category: motorcycle      Similarity:   3.3770
2024-12-26 22:39:15,349 - DemoProgram - INFO - Category: horse           Similarity:   3.2773
2024-12-26 22:39:15,349 - DemoProgram - INFO - Category: dog             Similarity:   3.2695
2024-12-26 22:39:15,349 - DemoProgram - INFO - 
Top 10 predicted attributes with similarities:
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: arched          Similarity:   3.8184
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: narrow          Similarity:   3.7949
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: water           Similarity:   3.6641
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: hanging         Similarity:   3.6465
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: curved          Similarity:   3.6445
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: in the picture  Similarity:   3.5605
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: rectangular     Similarity:   3.5566
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: reflective      Similarity:   3.5527
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: large           Similarity:   3.5508
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: folded          Similarity:   3.5371
2024-12-26 22:39:15,350 - DemoProgram - INFO - 
Ground truth attributes with similarities:
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: wet             Similarity:   3.3750
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: cool            Similarity:   3.3945
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: gray            Similarity:   3.1289
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: small           Similarity:   3.3613
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: old             Similarity:   3.3516
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: hard            Similarity:   3.1230
2024-12-26 22:39:15,350 - DemoProgram - INFO - Attribute: heavy           Similarity:   2.2578
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: new             Similarity:   3.1816
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: full            Similarity:   3.5352
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: dry             Similarity:   3.2852
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: whole           Similarity:   3.3848
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: white           Similarity:   3.2168
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: flat            Similarity:   3.3848
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: solid           Similarity:   3.4238
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: smooth          Similarity:   3.3340
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: curved          Similarity:   3.6445
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: horizontal      Similarity:   3.4648
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: short           Similarity:   3.3848
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: narrow          Similarity:   3.7949
2024-12-26 22:39:15,351 - DemoProgram - INFO - Attribute: moving          Similarity:   3.3066
2024-12-26 22:39:15,351 - DemoProgram - INFO - 
Top 10 predicted affordances with similarities:
2024-12-26 22:39:15,351 - DemoProgram - INFO - Affordance: row             Similarity:   4.0312 Definition: propel with oars
2024-12-26 22:39:15,351 - DemoProgram - INFO - Affordance: launch          Similarity:   4.0234 Definition: to put (a boat or ship) on the water
2024-12-26 22:39:15,351 - DemoProgram - INFO - Affordance: ride            Similarity:   3.7988 Definition: be carried or travel on or in a vehicle
2024-12-26 22:39:15,351 - DemoProgram - INFO - Affordance: buy             Similarity:   3.6953 Definition: obtain by purchase; acquire by means of a financial transaction
2024-12-26 22:39:15,351 - DemoProgram - INFO - Affordance: walk            Similarity:   3.6621 Definition: use one's feet to advance; advance by steps
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: swim            Similarity:   3.6348 Definition: 
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: transport       Similarity:   3.6328 Definition: move while supporting, either in a vehicle or in one's hands or on one's body
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: shoot           Similarity:   3.5996 Definition: send forth suddenly, intensely, swiftly
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: ride            Similarity:   3.5781 Definition: sit and travel on the back of animal, usually while controlling its motions
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: walk            Similarity:   3.5781 Definition: accompany or escort
2024-12-26 22:39:15,352 - DemoProgram - INFO - 
Ground truth affordances with similarities:
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: assemble        Similarity:   3.1387
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: break           Similarity:   2.9121
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: brush           Similarity:   3.2344
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: check           Similarity:   3.2168
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: clean           Similarity:   3.1348
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: drag            Similarity:   3.3750
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: draw            Similarity:   3.1699
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: drive           Similarity:   3.4570
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: edit            Similarity:   3.0625
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: function        Similarity:   3.2891
2024-12-26 22:39:15,352 - DemoProgram - INFO - Affordance: hit             Similarity:   2.9707
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: hit             Similarity:   3.4590
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: inspect         Similarity:   3.2227
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: install         Similarity:   3.0527
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: jump            Similarity:   3.3125
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: kick            Similarity:   3.5430
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: kick            Similarity:   3.0859
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: lie             Similarity:   3.5410
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: lie             Similarity:   3.2520
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: lie down        Similarity:   3.4434
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: lift            Similarity:   3.3379
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: move            Similarity:   3.5430
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: pack            Similarity:   3.3789
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: paint           Similarity:   3.4648
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: photograph      Similarity:   3.3594
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: pull            Similarity:   3.3594
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: push            Similarity:   3.1602
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: push            Similarity:   3.2871
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: shoot           Similarity:   2.6523
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: sign            Similarity:   3.1855
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: sit             Similarity:   3.5371
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: sit             Similarity:   3.2539
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: stand           Similarity:   3.1016
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: stand           Similarity:   3.3379
2024-12-26 22:39:15,353 - DemoProgram - INFO - Affordance: stand           Similarity:   3.1895
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: steer           Similarity:   3.5469
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: sweep           Similarity:   3.4551
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: tie             Similarity:   3.2500
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: touch           Similarity:   3.3477
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: transport       Similarity:   3.6328
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: walk            Similarity:   3.6621
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: write           Similarity:   3.3652
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: jump            Similarity:   3.4180
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: park            Similarity:   3.1172
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: row             Similarity:   4.0312
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: set up          Similarity:   3.4473
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: stop            Similarity:   3.3809
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: watch           Similarity:   3.1875
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: watch           Similarity:   3.4258
2024-12-26 22:39:15,354 - DemoProgram - INFO - Affordance: tag             Similarity:   2.6914
2024-12-26 22:39:15,354 - DemoProgram - INFO - 
=== Sample 3 ===
2024-12-26 22:39:15,354 - DemoProgram - INFO - 
Top 5 predicted categories with similarities:
2024-12-26 22:39:15,354 - DemoProgram - INFO - Ground truth category: boat
2024-12-26 22:39:15,354 - DemoProgram - INFO - Category: boat            Similarity:   4.4922
2024-12-26 22:39:15,354 - DemoProgram - INFO - Category: car             Similarity:   3.6250
2024-12-26 22:39:15,354 - DemoProgram - INFO - Category: truck           Similarity:   3.4766
2024-12-26 22:39:15,355 - DemoProgram - INFO - Category: motorcycle      Similarity:   3.3008
2024-12-26 22:39:15,355 - DemoProgram - INFO - Category: dog             Similarity:   3.2031
2024-12-26 22:39:15,355 - DemoProgram - INFO - 
Top 10 predicted attributes with similarities:
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: parked          Similarity:   4.2812
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: portable        Similarity:   4.1328
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: grassy          Similarity:   4.1094
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: dry             Similarity:   4.0898
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: closed          Similarity:   4.0586
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: spotted         Similarity:   4.0273
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: green           Similarity:   4.0156
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: digital         Similarity:   3.9902
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: large           Similarity:   3.9883
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: open            Similarity:   3.9629
2024-12-26 22:39:15,355 - DemoProgram - INFO - 
Ground truth attributes with similarities:
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: brown           Similarity:   3.6172
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: cool            Similarity:   3.7852
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: small           Similarity:   3.7910
2024-12-26 22:39:15,355 - DemoProgram - INFO - Attribute: old             Similarity:   3.6934
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: hard            Similarity:   3.5664
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: heavy           Similarity:   3.0977
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: new             Similarity:   3.6699
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: dry             Similarity:   4.0898
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: whole           Similarity:   3.7422
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: empty           Similarity:   3.7500
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: flat            Similarity:   3.7031
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: solid           Similarity:   3.7090
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: smooth          Similarity:   3.7305
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: rectangular     Similarity:   3.7910
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: curved          Similarity:   3.7656
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: horizontal      Similarity:   3.8926
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: short           Similarity:   3.7363
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: narrow          Similarity:   3.6680
2024-12-26 22:39:15,356 - DemoProgram - INFO - Attribute: parked          Similarity:   4.2812
2024-12-26 22:39:15,356 - DemoProgram - INFO - 
Top 10 predicted affordances with similarities:
2024-12-26 22:39:15,356 - DemoProgram - INFO - Affordance: launch          Similarity:   4.2109 Definition: to put (a boat or ship) on the water
2024-12-26 22:39:15,356 - DemoProgram - INFO - Affordance: turn            Similarity:   3.8711 Definition: let (something) fall or spill from a container
2024-12-26 22:39:15,356 - DemoProgram - INFO - Affordance: row             Similarity:   3.8457 Definition: propel with oars
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: swim            Similarity:   3.7793 Definition: 
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: capture         Similarity:   3.7793 Definition: capture as if by hunting, snaring, or trapping
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: surf            Similarity:   3.7773 Definition: 
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: skateboard      Similarity:   3.7715 Definition: 
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: park            Similarity:   3.7461 Definition: place temporarily
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: occupy          Similarity:   3.7422 Definition: occupy the whole of
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: carry           Similarity:   3.7402 Definition: have with oneself; have on one's person
2024-12-26 22:39:15,357 - DemoProgram - INFO - 
Ground truth affordances with similarities:
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: assemble        Similarity:   3.1230
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: break           Similarity:   3.2422
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: brush           Similarity:   3.2695
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: check           Similarity:   3.2051
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: clean           Similarity:   3.6250
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: climb           Similarity:   2.9746
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: drag            Similarity:   3.2305
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: draw            Similarity:   3.4473
2024-12-26 22:39:15,357 - DemoProgram - INFO - Affordance: edit            Similarity:   3.1055
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: hit             Similarity:   2.9883
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: hit             Similarity:   3.3477
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: hop on          Similarity:   3.1230
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: inspect         Similarity:   3.2422
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: jump            Similarity:   3.1113
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: kick            Similarity:   3.5098
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: kick            Similarity:   3.3027
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: lie             Similarity:   3.7402
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: lie             Similarity:   3.7227
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: lie down        Similarity:   3.7266
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: lift            Similarity:   3.3730
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: load            Similarity:   3.6113
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: pack            Similarity:   3.4785
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: paint           Similarity:   3.4453
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: photograph      Similarity:   3.5488
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: pull            Similarity:   3.6895
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: push            Similarity:   3.4355
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: push            Similarity:   3.4668
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: put             Similarity:   3.0801
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: raise           Similarity:   2.8906
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: repair          Similarity:   3.1270
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: shoot           Similarity:   3.1270
2024-12-26 22:39:15,358 - DemoProgram - INFO - Affordance: sign            Similarity:   2.9590
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: sit             Similarity:   3.6465
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: sit             Similarity:   3.3633
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: stand           Similarity:   3.4004
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: stand           Similarity:   3.5078
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: stand           Similarity:   3.2109
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: steer           Similarity:   3.0527
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: sweep           Similarity:   3.5527
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: tie             Similarity:   3.1621
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: touch           Similarity:   3.3086
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: walk            Similarity:   3.5508
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: write           Similarity:   3.3691
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: jump            Similarity:   3.1797
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: watch           Similarity:   3.5098
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: watch           Similarity:   3.4277
2024-12-26 22:39:15,359 - DemoProgram - INFO - Affordance: tag             Similarity:   2.7168
2024-12-26 22:39:15,413 - DemoProgram - INFO - 
=== Overall Metrics ===
2024-12-26 22:39:15,413 - DemoProgram - INFO - Category Accuracy: 1.0000
2024-12-26 22:39:15,413 - DemoProgram - INFO - Attribute mAP: 0.8374
2024-12-26 22:39:15,413 - DemoProgram - INFO - Affordance mAP: 0.9336
```