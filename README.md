## Vision-based Robotic Grasping: Papers and Codes
According to the kinds of grasp, the methods of vision-based robotic grasping can be roughly divided into two kinds, **2D planar grasp** and **6DoF Grasp**. This repository summaries these methods in recent years, which utilize __deep learning__ mostly. Before this summary, previous review papers are also reviewed.

## 0. Review Papers

**[arXiv]** 2019-A Review of Robot Learning for Manipulation- Challenges, Representations, and Algorithms, [[paper](https://arxiv.org/abs/1907.03146)]

**[arXiv]** 2019-Vision-based Robotic Grasping from Object Localization, Pose Estimation, Grasp Detection to Motion Planning: A Review, [[paper](https://arxiv.org/abs/1905.06658)]

**[MTI]** 2018-Review of Deep Learning Methods in Robotic Grasp Detection, [[paper](https://www.mdpi.com/2414-4088/2/3/57)]

**[ToR]** 2016-Data-Driven Grasp Synthesis - A Survey, [[paper](https://arxiv.org/abs/1309.2660)]

**[RAS]** 2012-An overview of 3D object grasp synthesis algorithms - A Survey, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0921889011001485)]



## 1. 2D Planar Grasp

**Grasp Representation:**
The grasp is represented as an oriented 2D box, and the grasp is constrained from one direction.

### 1.1 RGB or RGB-D based methods

This kind of methods directly regress the oriented 2D box from RGB or RGB-D images. When using RGB-D images, the depth image is regarded as an another channel, which is similar with RGB-based methods.

***2019:***

**[arXiv]** Form2Fit: Learning Shape Priors for Generalizable Assembly from Disassembly, [[paper](https://arxiv.org/abs/1910.13675)] [[code](https://form2fit.github.io/)]

**[IROS]** GRIP: Generative Robust Inference and Perception for Semantic Robot Manipulation in Adversarial Environments, [[paper](https://arxiv.org/abs/1903.08352)]

**[arXiv]** Efficient Fully Convolution Neural Network for Generating Pixel Wise Robotic Grasps With High Resolution Images, [[paper](https://arxiv.org/abs/1902.08950)]

**[arXiv]** A Single Multi-Task Deep Neural Network with Post-Processing for Object Detection with Reasoning and Robotic Grasp Detection, [[paper](https://arxiv.org/abs/1909.07050)]

**[IROS]** ROI-based Robotic Grasp Detection for Object Overlapping Scenes, [[paper](https://arxiv.org/abs/1808.10313)]

**[IROS]** SilhoNet: An RGB Method for 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1809.06893)]

**[ICRA]** Multi-View Picking: Next-best-view Reaching for Improved Grasping in Clutter, [[paper](https://arxiv.org/abs/1809.08564)] [[code](https://github.com/dougsm/mvp_grasp)]

***2018:***

**[arXiv]** Real-Time, Highly Accurate Robotic Grasp Detection using Fully Convolutional Neural Networks with High-Resolution Images, [[paper](https://arxiv.org/abs/1809.05828)]

**[arXiv]** Real-world Multi-object, Multi-grasp Detection, [[paper](https://arxiv.org/abs/1802.00520)]

**[ICRA]** Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching, [[paper](https://arxiv.org/abs/1710.01330)] [[code](https://github.com/andyzeng/arc-robot-vision)]

***2017:***

**[IROS]** Robotic Grasp Detection using Deep Convolutional Neural Networks, [[paper](https://arxiv.org/abs/1611.08036)]

***2016:***

**[ICRA]** Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours, [[paper](https://arxiv.org/abs/1509.06825)]

***2015:***

**[ICRA]** Real-time grasp detection using convolutional neural networks, [[paper](https://arxiv.org/abs/1412.3128)] [[code](https://github.com/tnikolla/robot-grasp-detection)]

***2014:***

**[IJRR]** Deep Learning for Detecting Robotic Grasps, [[paper](https://arxiv.org/abs/1301.3592)]

---

***Datasets:***

[Cornell dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php), the dataset consists of 1035 images of 280 different objects.



### 1.2 Depth-based methods

This kind of methods utilized an indirectly way to obtain the grasp pose, which contains grasp candidate generation and grasp quality evaluation. The candidate grasp with the highly score will be selected as the final grasp.

***2019:***

**[IROS]** GQ-STN: Optimizing One-Shot Grasp Detection based on Robustness Classifier, [[paper](https://arxiv.org/abs/1903.02489)]

**[ICRA]** Mechanical Search: Multi-Step Retrieval of a Target Object Occluded by Clutter, [[paper](https://arxiv.org/abs/1903.01588)]

**[ICRA]** MetaGrasp: Data Efficient Grasping by Affordance Interpreter Network, [[paper](https://arxiv.org/abs/1902.06554)]

**[IROS]** GlassLoc: Plenoptic Grasp Pose Detection in Transparent Clutter, [[paper](https://arxiv.org/abs/1909.04269)]

***2018:***

**[RSS]** Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach, [[paper](https://arxiv.org/pdf/1804.05172.pdf)]

**[BMVC]** EnsembleNet Improving Grasp Detection using an Ensemble of Convolutional Neural Networks, [[paper](http://bmvc2018.org/contents/papers/0322.pdf)]

***2017:***

**[RSS]** Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics, [[paper](https://github.com/BerkeleyAutomation/dex-net/raw/gh-pages/docs)] [[code](https://github.com/BerkeleyAutomation/gqcnn)]

---

***Dataset:***

[Dex-Net](https://berkeleyautomation.github.io/dex-net/#dexnet_2), a synthetic dataset of 6.7 million point clouds, grasps, and robust analytic grasp metrics generated from thousands of 3D models.

[Jacquard Dataset](https://jacquard.liris.cnrs.fr), Jacquard: A Large Scale Dataset for Robotic Grasp Detection” in *IEEE International Conference on Intelligent Robots and Systems*, 2018, [[paper](https://arxiv.org/abs/1803.11469)]



### 1.3 Target object localization in 2D

In order to provide a better input to compute the oriented 2D box, or generate the candidates, the targe object's mask should be computed. The current deep learning-based 2D detection or 2D segmentation methods could assist.

#### 1.3.1 2D detection:

***2019:***

**[arXiv]** OpenLORIS-Object: A Dataset and Benchmark towards Lifelong Object Recognition, [[paper](https://arxiv.org/abs/1911.06487)] [[project](https://lifelong-robotic-vision.github.io/dataset/Data_Object-Recognition.html)]

**[IROS]** Look Further to Recognize Better: Learning Shared Topics and Category-Specific Dictionaries for Open-Ended 3D Object Recognition, [[paper](https://arxiv.org/abs/1907.12924)]

**[IROS]** Recurrent Convolutional Fusion for RGB-D Object Recognition, [[paper](https://arxiv.org/pdf/1806.01673.pdf)] [[code](https://github.com/MRLoghmani/rcfusion)]

**[arXiv]** A Review of methods for Textureless Object Recognition, [[paper](https://arxiv.org/abs/1910.14255)]

**[ICCVW]** An Annotation Saved is an Annotation Earned: Using Fully Synthetic Training for Object Detection, [[paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/R6D/Hinterstoisser_An_Annotation_Saved_is_an_Annotation_Earned_Using_Fully_Synthetic_ICCVW_2019_paper.pdf)]

**[arXiv]** Object Detection in 20 Years A Survey, [[paper](https://arxiv.org/abs/1905.05055)]

***2018:***

**[arXiv]** YOLOv3: An Incremental Improvement, [[paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)] [[code](https://github.com/eriklindernoren/PyTorch-YOLOv3)]

***2016:***

**[CVPR]** You only look once: Unified, real-time object detection, [[paper](https://arxiv.org/abs/1506.02640)] [[code](https://github.com/gliese581gg/YOLO_tensorflow)]

**[TPAMI]** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, [[paper](https://arxiv.org/abs/1506.01497)] [[code](https://github.com/rbgirshick/py-faster-rcnn)]

**[ECCV]** SSD: Single Shot MultiBox Detector, [[paper](https://arxiv.org/abs/1512.02325)] [[code](https://github.com/balancap/SSD-Tensorflow)]

#### 1.3.2 2D segmentation:

***2019:***

**[CASE]** Deep Workpiece Region Segmentation for Bin Picking, [[paper](https://arxiv.org/abs/1909.03462)]

***2017:***

**[ICCV]** Mask r-cnn, [[paper](https://arxiv.org/abs/1703.06870)] [[code](https://github.com/matterport/Mask_RCNN)]

**[IROS]** SegICP: Integrated Deep Semantic Segmentation and Pose Estimation, [[paper](https://arxiv.org/abs/1703.01661)]



## 2. 6DoF Grasp

**Grasp Representation:**
The grasp is represented as 6DoF pose in 3D domain, and the gripper can grasp the object from various angles. The input to this task is 3D point cloud from RGB-D sensors, and this task contains two stages. In the first stage, the targe object should be extracted from the scene. In the second stage, if there exist an existing 3D model, the 6D pose of the object could be computed. If there exists no 3D models, the 6DoF grasp pose will be computed from some other methods.

### 2.1 Target object extraction in 3D

The staightforward way is to conduct 2D dection or segmentation, and utilize the point cloud from the corresponding depth area. This part is already related in section 1.3. In the following, only 3D detection and 3D instance segmentation will be summarized.

#### 2.1.1 3D detection

This kind of methods can be divided into three kinds: RGB-based methods, point cloud-based methods, and fusion methods which consume images and point cloud. Most of these works are focus on autonomous driving.

##### a. RGB-based methods

Most of this kind of methods estimate depth images from RGB images, and then conduct 3D detection.

***2019:***

**[IROS]** Look Further to Recognize Better: Learning Shared Topics and Category-Specific Dictionaries for Open-Ended 3D Object Recognition, [[paper](https://arxiv.org/abs/1907.12924)]

**[arXiv]** Task-Aware Monocular Depth Estimation for 3D Object Detection, [[paper](https://arxiv.org/abs/1909.07701)]

**[CVPR]** Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving, [[paper](https://arxiv.org/abs/1812.07179)] [[code](https://github.com/mileyan/pseudo_lidar)] 

**[AAAI]** MonoGRNet: A Geometric Reasoning Network for 3D Object Localization, [[paper](https://arxiv.org/abs/1811.10247)] [[code](https://github.com/Zengyi-Qin/MonoGRNet)]

**[ICCV]** Accurate Monocular 3D Object Detection via Color-Embedded 3D Reconstruction for Autonomous Driving, [[paper](https://arxiv.org/abs/1903.11444)]

**[ICCV]** M3D-RPN: Monocular 3D Region Proposal Network for Object Detection, [[paper](https://arxiv.org/abs/1907.06038)]

**[ICCVW]** Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud, [[paper](https://arxiv.org/abs/1903.09847)]

**[arXiv]** Monocular 3D Object Detection and Box Fitting Trained End-to-End Using Intersection-over-Union Loss, [[paper](https://arxiv.org/abs/1906.08070)]

**[arXiv]** Monocular 3D Object Detection via Geometric Reasoning on Keypoints, [[paper](https://arxiv.org/abs/1905.05618)]

##### b. Point cloud-based methods

This kind of methods purely utilize the 3D point cloud data.

***2019:***

**[NeurIPSW]** Patch Refinement -- Localized 3D Object Detection, [[paper](https://arxiv.org/abs/1910.04093)]

**[CoRL]** End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds, [[paper](https://arxiv.org/abs/1910.06528)]

**[ICCV]** Deep Hough Voting for 3D Object Detection in Point Clouds, [[paper](https://arxiv.org/abs/1904.09664)] [[code](https://github.com/facebookresearch/votenet)]

**[arXiv]** Part-A2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud, [[paper](https://arxiv.org/abs/1907.03670)]

**[ICCV]** STD: Sparse-to-Dense 3D Object Detector for Point Cloud, [[paper](https://arxiv.org/abs/1907.10471)]

**[CVPR]** PointPillars: Fast Encoders for Object Detection from Point Clouds, [[paper](https://arxiv.org/abs/1812.05784)]

**[arXiv]** StarNet: Targeted Computation for Object Detection in Point Clouds, [[paper](https://arxiv.org/abs/1908.11069)]

***2018:***

**[CVPR]** PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud, [[paper](https://arxiv.org/abs/1812.04244)] [[code](https://github.com/sshaoshuai/PointRCNN)]

**[CVPR]** PIXOR: Real-time 3D Object Detection from Point Clouds, [[paper](https://arxiv.org/abs/1902.06326)] [[code](https://github.com/philip-huang/PIXOR)]

**[ECCVW]** Complex-YOLO: Real-time 3D Object Detection on Point Clouds, [[paper](https://arxiv.org/abs/1803.06199)] [[code](https://github.com/AI-liu/Complex-YOLO)]

**[ECCVW]** YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud, [[paper](https://arxiv.org/abs/1808.02350)]

##### c. Fusion methods

This kind of methods utilize both rgb images and depth images/point clouds. There exist early fusion methods, late fusion methods, and dense fusion methods.

***2020:***

**[AAAI]*** PI-RCNN: An Efficient Multi-sensor 3D Object Detector with Point-based Attentive Cont-conv Fusion Module, [[paper](https://arxiv.org/abs/1911.06084)]

***2019:***

**[ICCV]** Transferable Semi-Supervised 3D Object Detection From RGB-D Data, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Tang_Transferable_Semi-Supervised_3D_Object_Detection_From_RGB-D_Data_ICCV_2019_paper.pdf)]

**[arXiv]** Adaptive and Azimuth-Aware Fusion Network of Multimodal Local Features for 3D Object Detection, [[paper](https://arxiv.org/abs/1910.04392)]

**[arXiv]** Frustum VoxNet for 3D object detection from RGB-D or Depth images, [[paper](https://arxiv.org/abs/1910.05483)]

**[IROS]** Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection, [[paper](https://arxiv.org/abs/1903.01864)]

**[CVPR]** Multi-Task Multi-Sensor Fusion for 3D Object Detection, [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Liang_Multi-Task_Multi-Sensor_Fusion_for_3D_Object_Detection_CVPR_2019_paper.html)]

***2018:***

**[CVPR]** Frustum PointNets for 3D Object Detection from RGB-D Data, [[paper](https://arxiv.org/abs/1711.08488)] [[code](https://github.com/charlesq34/frustum-pointnets)]

**[ECCV]** Deep Continuous Fusion for Multi-Sensor 3D Object Detection, [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf)]

**[IROS]** Joint 3D Proposal Generation and Object Detection from View Aggregation, [[paper](https://arxiv.org/abs/1712.02294)] [[code](https://github.com/kujason/avod)]

**[CVPR]** PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation, [[paper](https://arxiv.org/abs/1711.10871)]

**[ICRA]** A General Pipeline for 3D Detection of Vehicles, [[paper](https://arxiv.org/abs/1803.00387)]

***2017:***
**[CVPR]** Multi-View 3D Object Detection Network for Autonomous Driving, [[paper](https://arxiv.org/abs/1611.07759)] [[code](https://github.com/bostondiditeam/MV3D)]



#### 2.1.2 3D segmentation

***2019:***

**[NeurIPS]** Exploiting Local and Global Structure for Point Cloud Semantic Segmentation with Contextual Point Representations, [[paper](https://arxiv.org/abs/1911.05277)]

**[arXiv]** Addressing the Sim2Real Gap in Robotic 3D Object Classification, [[paper](https://arxiv.org/abs/1910.12585)]

**[arXiv]** Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds, [[paper](https://arxiv.org/pdf/1906.01140.pdf)]

**[IROS]** LDLS: 3-D Object Segmentation Through Label Diffusion From 2-D Images, [[paper](https://arxiv.org/abs/1910.13955)]

**[arXiv]** GSPN: Generative Shape Proposal Network for 3D Instance Segmentation in Point Cloud, [[paper](https://arxiv.org/abs/1812.03320)]

**[CoRL]** The Best of Both Modes: Separately Leveraging RGB and Depth for Unseen Object Instance Segmentation, [[paper](https://arxiv.org/abs/1907.13236)] [[code](https://arxiv.org/abs/1907.13236)]

***2018:***

**[arXiv]** PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation, [[paper](https://arxiv.org/abs/1807.00652)]



#### 2.1.3 3D deep learning networks

Some of these works are cited from [awesome-point-cloud-analysis](https://github.com/Yochengliu/awesome-point-cloud-analysis) by [Yongcheng Liu](https://yochengliu.github.io/), thank him.

***2019:***

**[arXiv]** Kaolin: A PyTorch Library for Accelerating 3D Deep Learning Research, [[paper](https://arxiv.org/abs/1911.05063)] [[code](https://github.com/NVIDIAGameWorks/kaolin/)]

**[ICCV]** DensePoint: Learning Densely Contextual Representation for Efficient Point Cloud Processing, [[paper](https://arxiv.org/pdf/1909.03669.pdf)] [[code](https://github.com/Yochengliu/DensePoint)]

**[TOG]** Dynamic Graph CNN for Learning on Point Clouds, [[paper](https://arxiv.org/pdf/1801.07829.pdf)] [[code](https://github.com/WangYueFt/dgcnn)]

**[ICCV]** DeepGCNs: Can GCNs Go as Deep as CNNs?, [[paper](https://arxiv.org/pdf/1904.03751.pdf)] [[code](https://github.com/lightaime/deep_gcns)]

**[ICCV]** KPConv: Flexible and Deformable Convolution for Point Clouds, [[paper](https://arxiv.org/abs/1904.08889)] [[code](https://github.com/HuguesTHOMAS/KPConv)]

**[MM]** SRINet: Learning Strictly Rotation-Invariant Representations for Point Cloud Classification and Segmentation, [[paper](https://arxiv.org/abs/1911.02163)]

**[CVPR]** PointConv: Deep Convolutional Networks on 3D Point Clouds, [[paper](https://arxiv.org/pdf/1811.07246.pdf)] [[code](https://github.com/DylanWusee/pointconv)]

**[CVPR]** PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing, [[paper](http://jiaya.me/papers/pointweb_cvpr19.pdf)] [[code](https://github.com/hszhao/PointWeb)]

**[CVPR]** Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN, [[paper](https://arxiv.org/pdf/1811.07782.pdf)] [[code](https://github.com/voidrank/Geo-CNN)]

**[arXiv]** SAWNet: A Spatially Aware Deep Neural Network for 3D Point Cloud Processing, [[paper](https://arxiv.org/pdf/1905.07650v1.pdf)]

**[arXiv]** PyramNet: Point Cloud Pyramid Attention Network and Graph Embedding Module for Classification and Segmentation, [[paper](https://arxiv.org/pdf/1906.03299.pdf)]

**[ICCV]** Interpolated Convolutional Networks for 3D Point Cloud Understanding, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Mao_Interpolated_Convolutional_Networks_for_3D_Point_Cloud_Understanding_ICCV_2019_paper.pdf)]

**[arXiv]** A survey on Deep Learning Advances on Different 3D Data Representations, [[paper](https://arxiv.org/abs/1808.01462)]

***2018:***

**[TOG]** MCCNN: Monte Carlo Convolution for Learning on Non-Uniformly Sampled Point Clouds, [[paper](https://arxiv.org/abs/1806.01759)] [[code](https://github.com/viscom-ulm/MCCNN)]

**[NeurIPS]** PointCNN: Convolution On X-Transformed Points, [[paper](https://arxiv.org/abs/1801.07791)] [[code](https://github.com/yangyanli/PointCNN)]

**[CVPR]** Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling, [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Mining_Point_Cloud_CVPR_2018_paper.pdf)] [[code](http://www.merl.com/research/license#KCNet)]

**[CVPR]** SO-Net: Self-Organizing Network for Point Cloud Analysis, [[paper](https://arxiv.org/abs/1803.04249)] [[code](https://github.com/lijx10/SO-Net)]

**[CVPR]** SPLATNet: Sparse Lattice Networks for Point Cloud Processing, [[paper](https://arxiv.org/abs/1802.08275)] [[code](https://github.com/NVlabs/splatnet)]

**[arXiv]** Point Convolutional Neural Networks by Extension Operators, [[paper](https://arxiv.org/abs/1803.10091)]

***2017:***

**[ICCV]** Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models, [[paper](https://arxiv.org/abs/1704.01222)] [[code](https://github.com/fxia22/kdnet.pytorch)]

**[CVPR]** PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, [[paper](https://arxiv.org/abs/1612.00593)] [[code](https://github.com/charlesq34/pointnet)]

**[NeurIPS]** PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space, [[paper](https://github.com/charlesq34/pointnet2)] [[code](https://github.com/charlesq34/pointnet2)]

**[CVPR]** SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation, [[paper](https://arxiv.org/abs/1612.00606)]



### 2.2 6D object pose estimation (Exist 3D models)

#### 2.2.1 RGB-D based methods

This kind of methods can be divided into four kinds, which are corresponding-based methods, template-based methods, voting-based methods and regression-based methods.

**[ECCVW]** A Summary of the 4th International Workshop on Recovering 6D Object Pose, [[paper](https://arxiv.org/abs/1810.03758)]

##### a. Corresponding-based methods

***2019:***

**[CVPR]** Segmentation-driven 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1812.02541)]

***2018:***

**[arXiv]** Estimating 6D Pose From Localizing Designated Surface Keypoints, [[paper](https://arxiv.org/abs/1812.01387)]

***2017:***

**[ICRA]** 6-DoF Object Pose from Semantic Keypoints, [[paper](https://arxiv.org/abs/1703.04670)]

***2012:***

**[3DIMPVT]** 3D Object Detection and Localization using Multimodal Point Pair Features, [[paper](http://far.in.tum.de/pub/drost20123dimpvt/drost20123dimpvt.pdf)]

##### b. Template-based methods

***2019:***

**[arXiv]** Real-time Background-aware 3D Textureless Object Pose Estimation, [[paper](https://arxiv.org/abs/1907.09128)]

***2012:***

**[ACCV]** Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.250.6547&rep=rep1&type=pdf)]

##### c. Voting-based methods

***2018:***

**[TPAMI]** Robust 3D Object Tracking from Monocular Images Using Stable Parts, [[paper](https://ieeexplore.ieee.org/document/7934426)]

***2014:***

**[ECCV]** Learning 6d object pose estimation using 3d object coordinate, [[paper](http://wwwpub.zih.tu-dresden.de/~cvweb/publications/papers/2014/PoseEstimationECCV2014.pdf)]

**[ECCV]** Latent-class hough forests for 3d object detection and pose estimation, [[paper](https://labicvl.github.io/docs/pubs/Aly_ECCV_2014.pdf)]

##### d. Regression-based methods

###### 1) Directly way

***2019:***

**[CoRL]** Scene-level Pose Estimation for Multiple Instances of Densely Packed Objects, [[paper](https://arxiv.org/abs/1910.04953)]

**[IROS]** Learning to Estimate Pose and Shape of Hand-Held Objects from RGB Images, [[paper](https://arxiv.org/abs/1903.03340)]

**[IROSW]** Motion-Nets: 6D Tracking of Unknown Objects in Unseen Environments using RGB, [[paper](https://arxiv.org/abs/1910.13942)]

**[ICCV]** DPOD: 6D Pose Object Detector and Refiner, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Zakharov_DPOD_6D_Pose_Object_Detector_and_Refiner_ICCV_2019_paper.html)]

**[ICCV]** Pix2Pose: Pixel-Wise Coordinate Regression of Objects for 6D Pose Estimation, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Pix2Pose_Pixel-Wise_Coordinate_Regression_of_Objects_for_6D_Pose_Estimation_ICCV_2019_paper.pdf)]

**[ICCV]** Explaining the Ambiguity of Object Detection and 6D Pose From Visual Data, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Manhardt_Explaining_the_Ambiguity_of_Object_Detection_and_6D_Pose_From_ICCV_2019_paper.pdf)]

**[arXiv]** Active 6D Multi-Object Pose Estimation in Cluttered Scenarios with Deep Reinforcement Learning, [[paper](https://arxiv.org/abs/1910.08811)]

**[arXiv]** 6-PACK: Category-level 6D Pose Tracker with Anchor-Based Keypoints, [[paper](https://arxiv.org/abs/1910.10750)] [[code](https://github.com/j96w/6-PACK)]

**[arXiv]** Accurate 6D Object Pose Estimation by Pose Conditioned Mesh Reconstruction, [[paper](https://arxiv.org/abs/1910.10653)]

**[arXiv]** Learning Object Localization and 6D Pose Estimation from Simulation and Weakly Labeled Real Images, [[paper](https://arxiv.org/abs/1806.06888)]

**[ICHR]** Refining 6D Object Pose Predictions using Abstract Render-and-Compare, [[paper](https://arxiv.org/abs/1910.03412)]

**[CVPR]** Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation, [[paper](https://arxiv.org/abs/1901.02970)] [[code](https://github.com/hughw19/NOCS_CVPR2019)] 

**[CVPR]** Densefusion: 6d object pose estimation by iterative dense fusion, [[paper](https://arxiv.org/abs/1901.04780)] [[code](https://github.com/j96w/DenseFusion)]

**[arXiv]** Deep-6dpose: recovering 6d object pose from a single rgb image, [[paper](https://arxiv.org/abs/1901.04780)]

***2018:***

**[ECCV]** Implicit 3D Orientation Learning for 6D Object Detection From RGB Images, [[paper](https://arxiv.org/abs/1902.01275)] [[code](https://github.com/DLR-RM/AugmentedAutoencoder)]

**[ECCV]** DeepIM:Deep Iterative Matching for 6D Pose Estimation
[[paper](https://arxiv.org/abs/1804.00175)] [[code](https://github.com/liyi14/mx-DeepIM)]

**[RSS]** Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes, [[paper](https://arxiv.org/abs/1711.00199)] [[code](https://github.com/yuxng/PoseCNN)]

**[IROS]** Robust 6D Object Pose Estimation in Cluttered Scenes using Semantic Segmentation and Pose Regression Networks, [[paper](https://arxiv.org/abs/1810.03410)]

***2017:***

**[ICCV]** SSD-6D: Making rgb-based 3d detection and 6d pose estimation great again, [[paper](https://arxiv.org/abs/1711.10006)] [[code](https://github.com/wadimkehl/ssd-6d)]

###### 2) Indirectly way (Firstly regress feature points and use PnP methods)

***2019:***

**[arXiv]** PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation, [[paper](https://arxiv.org/abs/1911.04231)]

**[ICCV]** CDPN: Coordinates-Based Disentangled Pose Network for Real-Time RGB-Based 6-DoF Object Pose Estimation, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_CDPN_Coordinates-Based_Disentangled_Pose_Network_for_Real-Time_RGB-Based_6-DoF_Object_ICCV_2019_paper.html)]

**[CVPR]** PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation, [[paper](https://arxiv.org/abs/1812.11788)] [[code](https://github.com/zju3dv/pvnet)]

***2018:***

**[CVPR]** Real-time seamless single shot 6d object pose prediction, [[paper](https://arxiv.org/abs/1711.08848)] [[code](https://github.com/Microsoft/singleshotpose)]

***2017:***

**[ICCV]** BB8: a scalable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth, [[paper](https://arxiv.org/abs/1703.10896)]

___

***Datasets:***

HomebrewedDB: RGB-D Dataset for 6D Pose Estimation of 3D Objects, ICCVW, 2019 [[paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/R6D/Kaskman_HomebrewedDB_RGB-D_Dataset_for_6D_Pose_Estimation_of_3D_Objects_ICCVW_2019_paper.pdf)]

[YCB Datasets](http://www.ycbbenchmarks.com): The YCB Object and Model Set: Towards Common Benchmarks for Manipulation Research, IEEE International Conference on Advanced Robotics (ICAR), 2015 [[paper](http://dx.doi.org/10.1109/ICAR.2015.7251504)]

[T-LESS Datasets](http://cmp.felk.cvut.cz/t-less/): T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects, IEEE Winter Conference on Applications of Computer Vision (WACV), 2017 [[paper](https://arxiv.org/abs/1701.05498)]



#### 2.2.2 3D point cloud

The partial-view point cloud will be aligned to the complete shape in order to obtain the 6D pose. Generally, coarse registration should be conduct firstly to provide an intial alignment, and dense registration methods like ICP (Iterative Closest Point) will be conducted to obtain the final 6D pose.

##### a. Ransac-based methods

***2014:***

**[SGP]** Super 4PCS Fast Global Pointcloud Registration via Smart Indexing, [[paper](https://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/super4pcs.pdf)] [[code](https://github.com/nmellado/Super4PCS)]

##### b. 3D feature-based methods

***2017:***

**[CVPR]** 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions, [[paper](https://arxiv.org/abs/1603.08182)] [[code](https://github.com/andyzeng/3dmatch-toolbox)]

***2016:***

**[arXiv]** Lessons from the Amazon Picking Challenge, [[paper](https://arxiv.org/abs/1601.05484v2)]

**[arXiv]** Team Delft's Robot Winner of the Amazon Picking Challenge 2016, [[paper](https://arxiv.org/abs/1610.05514)]

##### c. Deep learning-based methods

***2019:***

**[arXiv]** DeepICP: An End-to-End Deep Neural Network for 3D Point Cloud Registration, [[paper](https://arxiv.org/pdf/1905.04153v2.pdf)]

**[NeurIPS]** PRNet: Self-Supervised Learning for Partial-to-Partial Registration, [[paper](https://arxiv.org/abs/1910.12240)]

**[CVPR]** PointNetLK: Robust & Efficient Point Cloud Registration using PointNet, [[paper](https://arxiv.org/abs/1903.05711)] [[code](https://github.com/hmgoforth/PointNetLK)]

**[ICCV]** End-to-End CAD Model Retrieval and 9DoF Alignment in 3D Scans, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Avetisyan_End-to-End_CAD_Model_Retrieval_and_9DoF_Alignment_in_3D_Scans_ICCV_2019_paper.pdf)]

**[arXiv]** Iterative Matching Point, [[paper](https://arxiv.org/abs/1910.10328)]

**[arXiv]** Deep Closest Point: Learning Representations for Point Cloud Registration, [[paper](https://arxiv.org/abs/1905.03304)] [[code](https://github.com/WangYueFt/dcp)]

**[arXiv]** PCRNet: Point Cloud Registration Network using PointNet Encoding, [[paper](https://arxiv.org/abs/1908.07906)] [[code](https://github.com/vinits5/pcrnet)]

***2017:***

**[ICRA]** Multi-view Self-supervised Deep Learning for 6D Pose Estimation in the Amazon Picking Challenge, [[paper](https://arxiv.org/abs/1609.09475)] [[code](https://github.com/andyzeng/apc-vision-toolbox)]



### 2.3 Deep learning-based methods (No existing 3D models)

In this situation, there exist no 3D models, an the 6-DoF grasps are estimated from available partial data. This can be implemented by directly estimating from partial view point cloud, or indirectly estimating after shape completion.

#### 2.3.1 Estimating 6-DoF grasps from partial view point cloud

***2019:***

**[IROS]** Grasping Unknown Objects Based on Gripper Workspace Spheres, [[paper](http://eprints.lincoln.ac.uk/36370/1/IROS19_1656_MS.pdf)]

**[arXiv]** Learning to Generate 6-DoF Grasp Poses with Reachability Awareness, [[paper](https://arxiv.org/abs/1910.06404)]

**[CoRL]** S4G: Amodal Single-view Single-Shot SE(3) Grasp Detection in Cluttered Scenes, [[paper](https://arxiv.org/abs/1910.14218)]

**[ICCV]** 6-DoF GraspNet: Variational Grasp Generation for Object Manipulation, [[paper](https://arxiv.org/abs/1905.10520)]

**[ICRA]** PointNetGPD: Detecting Grasp Configurations from Point Sets, [[paper](https://arxiv.org/abs/1809.06267)] [[code](https://github.com/lianghongzhuo/PointNetGPD)]

***2017:***

**[IJRR]** Grasp Pose Detection in Point Clouds, [[paper](https://arxiv.org/abs/1706.09911)] [[code](https://github.com/atenpas/gpd)]



#### 2.3.2 Grasp affordance

***2019:***

**[IROS]** Detecting Robotic Affordances on Novel Objects with Regional Attention and Attributes, [[paper](https://arxiv.org/abs/1909.05770)]

**[IROS]** Learning Grasp Affordance Reasoning through Semantic Relations, [[paper](https://arxiv.org/abs/1906.09836)]

**[arXiv]** Automatic pre-grasps generation for unknown 3D objects, [[paper](https://arxiv.org/abs/1908.00221)]

**[IECON]** A novel object slicing based grasp planner for 3D object grasping using underactuated robot gripper, [[paper](https://arxiv.org/abs/1907.09142)]

***2018:***

**[arXiv]** Workspace Aware Online Grasp Planning, [[paper](https://arxiv.org/abs/1806.11402)]



#### 2.3.3 Shape completion assisted grasp

***2019:***

**[arXiv]** ClearGrasp- 3D Shape Estimation of Transparent Objects for Manipulation, [[paper](https://arxiv.org/abs/1910.02550)]

**[arXiv]** kPAM-SC: Generalizable Manipulation Planning using KeyPoint Affordance and Shape Completion, [[paper](https://arxiv.org/abs/1909.06980)] [[code](https://sites.google.com/view/generalizable-manipulation/)]

**[arXiv]** Data-Efficient Learning for Sim-to-Real Robotic Grasping using Deep Point Cloud Prediction Networks, [[paper](https://arxiv.org/abs/1906.08989)]

**[arXiv]** Inferring Occluded Geometry Improves Performance when Retrieving an Object from Dense Clutter, [[paper](https://arxiv.org/abs/1907.08770)]

**[IROS]** Robust Grasp Planning Over Uncertain Shape Completions, [[paper](https://arxiv.org/abs/1903.00645)]

**[arXiv]** Multi-Modal Geometric Learning for Grasping and Manipulation, [[paper](https://arxiv.org/abs/1803.07671)]

***2018:***

**[ICRA]** Learning 6-DOF Grasping Interaction via Deep Geometry-aware 3D Representations, [[paper](https://arxiv.org/abs/1708.07303)]

**[IROS]** 3D Shape Perception from Monocular Vision, Touch, and Shape Priors, [[paper](https://arxiv.org/abs/1808.03247)]

***2016:***

**[IROS]** Shape Completion Enabled Robotic Grasping, [[paper](https://arxiv.org/abs/1609.08546)]



#### 2.3.4 Depth completion

***2020:***

**[AAAI]** CSPN++: Learning Context and Resource Aware Convolutional Spatial Propagation Networks for Depth Completion, [[paper](https://arxiv.org/abs/1911.05377)]

***2019:***

**[CVIU]** On the Benefit of Adversarial Training for Monocular Depth Estimation, [[paper](https://arxiv.org/abs/1910.13340)]

**[ICCV]** Learning Joint 2D-3D Representations for Depth Completion, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Joint_2D-3D_Representations_for_Depth_Completion_ICCV_2019_paper.pdf)]

**[ICCV]** Deep Optics for Monocular Depth Estimation and 3D Object Detection, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chang_Deep_Optics_for_Monocular_Depth_Estimation_and_3D_Object_Detection_ICCV_2019_paper.pdf)]

**[arXiv]** Deep Classification Network for Monocular Depth Estimation, [[paper](https://arxiv.org/abs/1910.10369)]

**[ICCV]** Depth Completion from Sparse LiDAR Data with Depth-Normal Constraints, [[paper](https://arxiv.org/abs/1910.06727)]

**[arXiv]** Image-based 3D Object Reconstruction: State-of-the-Art and Trends in the Deep Learning Era, [[paper](https://arxiv.org/abs/1906.06543)]

**[arXiv]** Real-time Vision-based Depth Reconstruction with NVidia Jetson, [[paper](https://arxiv.org/abs/1907.07210)]

**[IROS]** Self-supervised 3D Shape and Viewpoint Estimation from Single Images for Robotics, [[paper](https://arxiv.org/abs/1910.07948)]

**[arXiv]** Mesh R-CNN, [[paper](https://arxiv.org/abs/1906.02739)]

***2018:***

**[NeurIPS]** Learning to Reconstruct Shapes from Unseen Classes, [[paper](http://genre.csail.mit.edu/papers/genre_nips.pdf)] [[code](https://github.com/xiumingzhang/GenRe-ShapeHD)]

**[ECCV]** Learning Shape Priors for Single-View 3D Completion and Reconstruction, [[paper](https://arxiv.org/abs/1809.05068)] [[code](https://github.com/xiumingzhang/GenRe-ShapeHD)]

**[CVPR]** Deep Depth Completion of a Single RGB-D Image, [[paper](https://arxiv.org/abs/1803.09326)] [[code](https://github.com/yindaz/DeepCompletionRelease)]



## 3. Grasp Transfer

### 3.1 Task-oriented manipulation

***2019:***

**[arXiv]** KETO: Learning Keypoint Representations for Tool Manipulation, [[paper](https://arxiv.org/abs/1910.11977)]

**[arXiv]** Learning Task-Oriented Grasping from Human Activity Datasets, [[paper](https://arxiv.org/abs/1910.11669)]



### 3.2 Grasp transfer between shape parts

***2019:***

**[arXiv]** Using Synthetic Data and Deep Networks to Recognize Primitive Shapes for Object Grasping, [[paper](https://arxiv.org/abs/1909.08508)]

**[ICRA]** Transferring Grasp Configurations using Active Learning and Local Replanning, [[paper](https://arxiv.org/abs/1807.08341)]

***2017:***

**[AIP]** Fast grasping of unknown objects using principal component analysis, [[paper](https://aip.scitation.org/doi/10.1063/1.4991996)]

***2015:***

**[RAS]** Category-based task specific grasping, [[paper](https://www.sciencedirect.com/science/article/pii/S0921889015000846?via%3Dihub)]



### 3.3 Non-rigid shape matching

#### 3.3.1 Non-rigid registration

***2019:***

**[arXiv]** Non-Rigid Point Set Registration Networks, [[paper](https://arxiv.org/abs/1904.01428)] [[code](https://github.com/Lingjing324/PR-Net)]

***2018:***

**[RAL]** Transferring Category-based Functional Grasping Skills by Latent Space Non-Rigid Registration, [[paper](https://arxiv.org/abs/1809.05390)]

**[RAS]** Learning Postural Synergies for Categorical Grasping through Shape Space Registration, [[paper](https://arxiv.org/abs/1810.07967)]

**[RAS]** Autonomous Dual-Arm Manipulation of Familiar Objects, [[paper](https://arxiv.org/abs/1811.08716)]



#### 3.3.2 Shape correspondence

***2019:***

**[IROS]** Multi-step Pick-and-Place Tasks Using Object-centric Dense Correspondences, [[code](https://github.com/cychai1995/mcdons)]

**[arXiv]** Unsupervised cycle-consistent deformation for shape matching, [[paper](https://arxiv.org/abs/1907.03165)]

**[arXiv]** ZoomOut: Spectral Upsampling for Efficient Shape Correspondence, [[paper](https://arxiv.org/abs/1904.07865)]

**[C&G]** Partial correspondence of 3D shapes using properties of the nearest-neighbor field, [[paper](http://webee.technion.ac.il/~ayellet/Ps/19-ATZ.pdf)]



### 3.4 3D part segmentation

***2019:***

**[CVPR]** PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation, [[paper](https://arxiv.org/abs/1903.00709)] [[code](https://github.com/FoggYu/PartNet)]

**[C&G]** Autoencoder-based part clustering for part-in-whole retrieval of CAD models, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0097849319300391)]

***2016:***

**[SiggraphAsia]** A Scalable Active Framework for Region Annotation in 3D Shape Collections, [[paper](https://cs.stanford.edu/~ericyi/project_page/part_annotation/)]



## 4. Deterous Grippers

***2019:***

**[arXiv]** HMTNet:3D Hand Pose Estimation from Single Depth Image Based on Hand Morphological Topology, [[paper](https://arxiv.org/abs/1911.04930)]

**[arXiv]** UniGrasp: Learning a Unified Model to Grasp with N-Fingered Robotic Hands, [[paper](https://arxiv.org/abs/1910.10900)]

**[ScienceRobotics]** On the choice of grasp type and location when handing over an object, [[paper](https://robotics.sciencemag.org/content/4/27/eaau9757)]

**[arXiv]** Solving Rubik's Cube with a Robot Hand, [[paper](https://arxiv.org/abs/1910.07113)]

**[IJARS]** Fast geometry-based computation of grasping points on three-dimensional point clouds, [[paper](https://www.researchgate.net/publication/331358070_Fast_Geometry-based_Computation_of_Grasping_Points_on_Three-dimensional_Point_Clouds)] [[code](https://github.com/yayaneath/GeoGrasp)]

**[arXiv]** Learning better generative models for dexterous, single-view grasping of novel objects, [[paper](https://arxiv.org/abs/1907.06053)]

**[arXiv]** DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System, [[paper](https://arxiv.org/abs/1910.03135)]

**[IROS]** Optimization Model for Planning Precision Grasps with Multi-Fingered Hands, [[paper](https://arxiv.org/abs/1904.07332)]

**[IROS]** Generating Grasp Poses for a High-DOF Gripper Using Neural Networks, [[paper](https://arxiv.org/abs/1903.00425)]

**[arXiv]** Deep Dynamics Models for Learning Dexterous Manipulation, [[paper](https://arxiv.org/abs/1909.11652)]

**[CVPR]** Learning joint reconstruction of hands and manipulated objects, [[paper](https://arxiv.org/abs/1904.05767)]

**[CVPR]** H+O: Unified Egocentric Recognition of 3D Hand-Object Poses and Interactions, [[paper](https://arxiv.org/abs/1904.05349)]

**[IROS]** Efficient Grasp Planning and Execution with Multi-Fingered Hands by Surface Fitting, [[paper](https://arxiv.org/abs/1902.10841)]

**[arXiv]** Efficient Bimanual Manipulation Using Learned Task Schemas, [[paper](https://arxiv.org/abs/1909.13874)]

**[ICRA]** High-Fidelity Grasping in Virtual Reality using a Glove-based System, [[paper](https://github.com/zzlyw/ICRA19_VRGloveSystem/blob/master/doc/ICRA19.pdf)] [[code](https://github.com/zzlyw/ICRA19_VRGloveSystem)]



## 5. Simulation to Reality

***2019:***

**[arXiv]** Self-supervised 6D Object Pose Estimation for Robot Manipulation, [[paper](https://arxiv.org/abs/1909.10159)]

**[arXiv]** Accept Synthetic Objects as Real-End-to-End Training of Attentive Deep Visuomotor Policies for Manipulation in Clutter, [[paper](https://arxiv.org/abs/1909.11128)]

**[RSSW]** Generative grasp synthesis from demonstration using parametric mixtures, [[paper](https://arxiv.org/abs/1906.11548)]

***2018:***

**[RSS]** Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision, [[paper](https://arxiv.org/abs/1806.09266)]

**[CoRL]** Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects, [[paper](https://arxiv.org/abs/1809.10790)]

**[arXiv]** Multi-Task Domain Adaptation for Deep Learning of Instance Grasping from Simulation, [[paper](https://arxiv.org/abs/1710.06422)]

***2017:***

**[arXiv]** Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping, [[paper](https://arxiv.org/abs/1709.07857)]



## 6. Multi-source

***2019:***

**[ICRA]** Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks, [[paper](https://arxiv.org/abs/1907.13098)]

**[CVPR]**  ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging, [[paper](https://arxiv.org/abs/1904.06830)] [[code](https://github.com/samarth-robo/contactdb_utils)]

***2018:***

**[arXiv]** Learning to Grasp without Seeing, [[paper](https://arxiv.org/abs/1805.04201)]



## 7. Learning from Demonstration

***2019:***

**[arXiv]** Motion Reasoning for Goal-Based Imitation Learning, [[paper](https://arxiv.org/abs/1911.05864)]

**[IROS]** Robot Learning of Shifting Objects for Grasping in Cluttered Environments, [[paper](https://arxiv.org/abs/1907.11035)] [[code](https://github.com/pantor/learning-shifting-for-grasping)]

**[arXiv]** Learning Deep Parameterized Skills from Demonstration for Re-targetable Visuomotor Control, [[paper](https://arxiv.org/abs/1910.10628)]

**[arXiv]** Adversarial Skill Networks: Unsupervised Robot Skill Learning from Video, [[paper](https://arxiv.org/abs/1910.09430)]

**[IROS]** Learning Actions from Human Demonstration Video for Robotic Manipulation, [[paper](https://arxiv.org/abs/1909.04312)]

**[RSSW]** Generative grasp synthesis from demonstration using parametric mixtures, [[paper](https://arxiv.org/abs/1906.11548)]

***2018:***

**[arXiv]** Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation, [[paper](https://arxiv.org/abs/1710.04615)]



## 8. Reinforcement Learning

***2019:***

**[IROS]** Scaling Robot Supervision to Hundreds of Hours with RoboTurk: Robotic Manipulation Dataset through Human Reasoning and Dexterity, [[paper](https://arxiv.org/abs/1911.04052)]

**[arXiv]** IRIS: Implicit Reinforcement without Interaction at Scale for Learning Control from Offline Robot Manipulation Data, [[paper](https://arxiv.org/abs/1911.05321)]

**[arXiv]** Dynamic Cloth Manipulation with Deep Reinforcement Learning, [[paper](https://arxiv.org/abs/1910.14475)]

**[CoRL]** Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning, [[paper](https://arxiv.org/abs/1910.11956)] [[project](https://relay-policy-learning.github.io/)]

**[CoRL]** Asynchronous Methods for Model-Based Reinforcement Learning, [[paper](https://arxiv.org/abs/1910.12453)]

**[CoRL]** Entity Abstraction in Visual Model-Based Reinforcement Learning, [[paper](https://arxiv.org/abs/1910.12827)]

**[CoRL]** Dynamics Learning with Cascaded Variational Inference for Multi-Step Manipulation, [[paper](https://arxiv.org/abs/1910.13395)] [[project](http://pair.stanford.edu/cavin/)]

**[arXiv]** Contextual Imagined Goals for Self-Supervised Robotic Learning, [[paper](https://arxiv.org/abs/1910.11670)]

**[arXiv]** Learning to Manipulate Deformable Objects without Demonstrations, [[paper](https://arxiv.org/abs/1910.13439)] [[project](https://sites.google.com/view/alternating-pick-and-place)]

**[arXiv]** A Deep Learning Approach to Grasping the Invisible, [[paper](https://arxiv.org/abs/1909.04840)]

**[arXiv]** Knowledge Induced Deep Q-Network for a Slide-to-Wall Object Grasping, [[paper](https://arxiv.org/abs/1910.03781)]

**[arXiv]** Quantile QT-Opt for Risk-Aware Vision-Based Robotic Grasping, [[paper](https://arxiv.org/abs/1910.02787)]

**[arXiv]** Adaptive Curriculum Generation from Demonstrations for Sim-to-Real Visuomotor Control, [[paper](https://arxiv.org/abs/1910.07972)]

**[arXiv]** Reinforcement Learning for Robotic Manipulation using Simulated Locomotion Demonstrations, [[paper](https://arxiv.org/abs/1910.07294)]

**[arXiv]** Self-Supervised Sim-to-Real Adaptation for Visual Robotic Manipulation, [[paper](https://arxiv.org/abs/1910.09470)]

**[arXiv]** Object Perception and Grasping in Open-Ended Domains, [[paper](https://arxiv.org/abs/1907.10932)]

**[CoRL]** ROBEL: Robotics Benchmarks for Learning with Low-Cost Robots, [[paper](https://arxiv.org/abs/1909.11639)] [[code](https://sites.google.com/view/roboticsbenchmarks/)]

**[RSS]** End-to-End Robotic Reinforcement Learning without Reward Engineering, [[paper](https://arxiv.org/abs/1904.07854)]

**[arXiv]** Learning to combine primitive skills: A step towards versatile robotic manipulation, [[paper](https://arxiv.org/abs/1908.00722)]

**[CoRL]** A Survey on Reproducibility by Evaluating Deep Reinforcement Learning Algorithms on Real-World Robots, [[paper](https://arxiv.org/abs/1909.03772)] [[code](https://github.com/dti-research/SenseActExperiments/)]

**[ICCAS]** Deep Reinforcement Learning Based Robot Arm Manipulation with Efficient Training Data through Simulation, [[paper](https://arxiv.org/abs/1907.06884)]

**[CVPR]** CRAVES: Controlling Robotic Arm with a Vision-based Economic System, [[paper](https://arxiv.org/abs/1812.00725)] [[code](https://github.com/zuoym15/craves.ai)]

**[Report]** A Unified Framework for Manipulating Objects via Reinforcement Learning, [[paper](https://course.ie.cuhk.edu.hk/~ierg6130/report/team7.pdf)]

***2018:***

**[IROS]** Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning, [[paper](https://arxiv.org/abs/1803.09956)] [[code](https://github.com/andyzeng/visual-pushing-grasping)]

**[CoRL]** QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation, [[paper](https://arxiv.org/abs/1806.10293)]

**[arXiv]** Deep Reinforcement Learning for Vision-Based Robotic Grasping: A Simulated Comparative Evaluation of Off-Policy Methods, [[paper](https://arxiv.org/abs/1802.10264)]

**[arXiv]** Pick and Place Without Geometric Object Models, [[paper](https://arxiv.org/abs/1707.05615)]

***2017:***

**[arXiv]** Deep Reinforcement Learning for Robotic Manipulation-The state of the art, [[paper](https://arxiv.org/abs/1701.08878)]

***2016:***

**[IJRR]** Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning, [[paper](https://arxiv.org/abs/1603.02199)]

***2013:***

**[IJRR]** Reinforcement learning in robotics: A survey, [[paper](https://ri.cmu.edu/pub_files/2013/7/Kober_IJRR_2013.pdf)]



## 9. Visual servoing

***2019:***

**[ICRA]** Learning Driven Coarse-to-Fine Articulated Robot Tracking, [[paper](http://www.robots.ox.ac.uk/~mobile/drs/Papers/2019ICRA_rauch.pdf)]

**[CVPR]** Craves: controlling robotic arm with a vision-based, economic system, [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zuo_CRAVES_Controlling_Robotic_Arm_With_a_Vision-Based_Economic_System_CVPR_2019_paper.pdf)] [[code](https://github.com/zuoym15/craves.ai)]

***2018:***

**[arXiv]** Point-to-Pose Voting based Hand Pose Estimation using Residual Permutation Equivariant Layer, [[paper](https://arxiv.org/pdf/1812.02050.pdf)]

***2016:***

**[ICRA]** Robot Arm Pose Estimation by Pixel-wise Regression of Joint Angles, [[paper](https://www.is.mpg.de/uploads_file/attachment/attachment/311/ICRA16_felix_small.pdf)]

***2014:***

**[ICRA]** Robot Arm Pose Estimation through Pixel-Wise Part Classification, [[paper](https://www.is.mpg.de/uploads_file/attachment/attachment/176/2014_ICRA_brhs_small.pdf)]





## 10. Experts:

[Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg/)(CMU & FAIR): Robotics, machine learning

[Andreas ten Pas](http://www.ccs.neu.edu/home/atp/)(Northeastern University): Robotic Grasping, Deep Learning, Simulation-based Planning

[Andy Zeng](http://andyzeng.github.io/)(Princeton University & Google Brain Robotics): 3D Deep Learning, Robotic Grasping

[Animesh Garg](https://www.cs.toronto.edu/~garg/)(University of Toronto): Robotics, Reinforcement Learning

[Cewu Lu](http://mvig.sjtu.edu.cn/)(SJTU): Machine Vision

[Charles Ruizhongtai Qi](https://web.stanford.edu/~rqi/)(Waymo(Google)): 3D Deep Learning

[Danfei Xu](https://cs.stanford.edu/~danfei/)(Stanford University): Robotics, Computer Vision

[Deter Fox](https://homes.cs.washington.edu/~fox/)(Nvidia & University of Washington): Robotics, Artificial intelligence, State Estimation

[Fei-Fei Li](https://profiles.stanford.edu/fei-fei-li/?utm_campaign=Artificial%2BIntelligence%2Band%2BDeep%2BLearning%2BWeekly&utm_medium=web&utm_source=Artificial_Intelligence_and_Deep_Learning_Weekly_3)(Stanford University): Computer Vision

[Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/)(ZJU): 3D Vision, SLAM

[Hao Su](http://cseweb.ucsd.edu/~haosu/)(UC San Diego): 3D Deep Learning

[Jeannette Bohg](https://am.is.tuebingen.mpg.de/person/jbohg)(Stanford University): perception for autonomous robotic manipulation and grasping

[Jianping Shi](https://shijianping.me/)(SenseTime): Computer Vision

[Juxi Leitner](http://juxi.net/)(Australian Centre of Excellence for Robotic Vision (ACRV)): Robotic grasping

[Lerrel Pinto](https://cs.nyu.edu/~lp91/)(UC Berkeley): Robotics

[Lorenzo Jamone](http://lorejam.blogspot.com/)(Queen Mary University of London (QMUL)): Cognitive Robotics

[Lorenzo Natale](http://lornat75.github.io/index.html)(Italian Institute of Technology): Humanoid robotic sensing and perception

[Kaiming He](http://kaiminghe.com/)(Facebook AI Research (FAIR)): Deep Learning

[Kai Xu](https://kevinkaixu.net/)(NUDT): Graphics, Geometry

[Ken Goldberg](https://goldberg.berkeley.edu/)(UC Berkeley): Robotics

[Marc Pollefeys](https://inf.ethz.ch/personal/marc.pollefeys/)(Microsoft & ETH): Computer Vision

[Markus Vincze](https://www.acin.tuwien.ac.at/staff/vm/)(Technical University Wien (TUW)): Robotic Vision

[Oliver Brock](https://www.robotics.tu-berlin.de/menue/team/oliver_brock)(TU Berlin): Robotic manipulation

[Pascal Fua](https://icwww.epfl.ch/~fua/)(INRIA): Computer Vision

[Peter K. Allen.](http://www.cs.columbia.edu/~allen/)(Columbia University): Robotic Grasping, 3-D vision, Modeling, Medical robotics

[Peter Corke](http://petercorke.com/wordpress/)(Queensland University of Technology): Robotic vision

[Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/)(UC Berkeley): Artificial Intelligence, Advanced Robotics

[Raquel Urtasun](http://www.cs.toronto.edu/~urtasun/)(Uber ATG & University of Toronto): AI for self-driving cars, Computer Vision, Robotics

[Robert Platt](http://www.ccs.neu.edu/home/rplatt/)(Northeastern University): Robotic manipulation

[Ruigang Yang](http://research.baidu.com/People/index-view?id=114)(Baidu): Computer Vision, Robotics

[Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)(UC Berkeley): Reinforcement Learning

[Shuran Song](https://shurans.github.io/)(Columbia University), 3D Deep Learning, Robotics

[Silvio Savarese](http://cvgl.stanford.edu/silvio/)(Stanford University): Computer Vision

[Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/)(UCLA): Computer Vision

[Tamim Asfour](https://h2t.anthropomatik.kit.edu/english/21_66.php/)(Karlsruhe Institute of Technology (KIT)): Humanoid Robotics

[Thomas Funkhouser](https://www.cs.princeton.edu/~funk/)(Princeton University): Geometry, Graphics, Shape

[Valerio Ortenzi](https://www.birmingham.ac.uk/staff/profiles/metallurgy/ortenzi-valerio.aspx)(University of Birmingham): Robotic vision

[Vicient Lepetit](https://www.labri.fr/perso/vlepetit/index.php)(University of Bordeaux): Machine Learning, 3D Vision

[Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/)(Chinese University of Hong Kong): Deep Learning, Computer Vision 

[Xiaozhi Chen](https://xiaozhichen.github.io/)(DJI): Deep learning

[Yan Xinchen](https://sites.google.com/site/skywalkeryxc/)(Uber ATG): Deep Representation Learning, Generative Modeling

[Yu Xiang](https://yuxng.github.io/)(Nvidia): Robotics, Computer Vision

[Yue Wang](https://people.csail.mit.edu/yuewang/)(MIT): 3D Deep Learning