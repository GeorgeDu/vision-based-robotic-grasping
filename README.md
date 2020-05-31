## Vision-based Robotic Grasping: Papers and Codes

The essential information to grasp the target object is the 6D gripper pose in the camera coordinate, which contains the 3D gripper position and the 3D gripper orientation to execute the grasp. Within the methods of vision-based robotic grasping, the estimation of 6D gripper poses varies aiming at different grasp manners, which can be categorized into __2D planar grasp__ and __6DoF grasp__.

__2D planar grasp__ means that the target object lies on a plane workspace and the grasp is constrained from one direction. The essential information is simplified from 6D into 3D, which are the 2D in-plane positions and 1D rotation angle. There exist methods of __evaluating grasp contact points__ and methods of __evaluating grasp oriented rectangles__.

__6DoF grasp__ means that the gripper can grasp the object from various angles in the 3D domain, and the essential 6D gripper pose could not be simplified. Based on whether the grasp is conducted on the complete shape or on the single-view point cloud, methods are categorized into __methods based on the partial point cloud__ and __methods based on the complete shape__. Methods based on the partial point cloud contains __methods of estimating candidate grasps__ and __methods of transferring grasps__ from existing grasps database.  Methods based on the complete shape contains __methods of estimating 6D object pose__ and __methods of shape completion__. Most of current 6DoF grasp methods aim at known objects where the grasps could be precomputed manually or by simulation, and the problem is thus transformed into a __6D object pose estimation__ problem.

Besides, most of the robotic grasping approaches require __the target object’s location__ in the input data first. This involves three different stages: __object localization without classification__, __object detection__ and __object instance segmentation__. Object localization without classification only outputs the potential regions of the target objects without knowing their categories. Object detection provides bounding boxes of the target objects with their categories. Object instance segmentation further provides pixel or point-level regions of the target objects with their categories.

I summarize all above kinds of methods in this repository, and hope to present a big picture for friends work on vision-based robotic grasping. The table of content is listed as follows.

  * [Vision-based Robotic Grasping: Papers and Codes](#vision-based-robotic-grasping-papers-and-codes)
  * [0. Review Papers](#0-review-papers)
  * [1. Object Localization](#1-object-localization)
     * [1.1 Object Localization without Classification](#11-object-localization-without-classification)
        * [1.1.1 2D-based Methods](#111-2d-based-methods)
           * [a. Fitting 2D Shape Primitives](#afitting-2d-shape-primitives)
           * [b. Saliency Detection](#b-saliency-detection)
        * [1.1.2 3D-based Methods](#112-3d-based-methods)
           * [a. Fitting 3D Shape Primitives](#afitting-3d-shape-primitives)
           * [b. Saliency Detection](#b-saliency-detection-1)
     * [1.2 Object Detection](#12-object-detection)
        * [1.2.1 2D Object Detection](#121-2d-object-detection)
           * [a. Two-stage methods](#a-two-stage-methods)
           * [b. Single-stage methods](#b-single-stage-methods)
        * [1.2.2 3D Object Detection](#122-3d-object-detection)
           * [a. RGB-based methods](#a-rgb-based-methods)
           * [b. Point cloud-based methods](#b-point-cloud-based-methods)
           * [c. Fusion methods](#c-fusion-methods)
     * [1.3 Object Instance Segmentation](#13-object-instance-segmentation)
        * [1.3.1 2D Instance Segmentation](#131-2d-instance-segmentation)
           * [a. Survey papers](#a-survey-papers)
           * [b. Two-stage methods](#b-two-stage-methods)
           * [c. One-stage methods](#c-one-stage-methods)
           * [d. Panoptic segmentation](#d-panoptic-segmentation)
        * [1.3.2 3D Instance Segmentation](#132-3d-instance-segmentation)
           * [a. Two-stage methods](#a-two-stage-methods-1)
           * [b. One-stage Methods](#b-one-stage-methods)
           * [c. 3D deep learning networks](#c-3d-deep-learning-networks)
  * [2. Object Pose Estimation](#2-object-pose-estimation)
     * [2.1 RGB-D Image-based Methods](#21-rgb-d-image-based-methods)
        * [2.1.1 Correspondence-based Methods](#211-correspondence-based-methods)
           * [a. Match 2D feature points](#a-match-2d-feature-points)
           * [b. Regress 2D projections](#b-regress-2d-projections)
        * [2.1.2 Template-based Methods](#212-template-based-methods)
        * [2.1.3 Voting-based Methods](#213-voting-based-methods)
     * [2.2 Point Cloud-based Methods](#22-point-cloud-based-methods)
        * [2.2.1 Correspondence-based Methods](#221-correspondence-based-methods)
        * [2.2.2 Template-based Methods](#222-template-based-methods)
        * [2.2.3 Voting-based Methods](#223-voting-based-methods)
     * [2.3 Category-level Methods](#23-category-level-methods)
        * [2.3.1 Category-level 6D pose estimation](#231-category-level-6d-pose-estimation)
        * [2.3.2 3D shape reconstruction from images](#232-3d-shape-reconstruction-from-images)
        * [2.3.3 3D shape rendering](#233-3d-shape-rendering)
  * [3. 2D Planar Grasp](#3-2d-planar-grasp)
     * [3.1 Estimating Grasp Contact Points](#31-estimating-grasp-contact-points)
     * [3.2 Estimating Oriented Rectangles](#32-estimating-oriented-rectangles)
  * [4. 6DoF Grasp](#4-6dof-grasp)
     * [4.1 Methods based on Single-view Point Cloud](#41-methods-based-on-single-view-point-cloud)
        * [4.1.1 Methods of Estimating Candidate Grasps](#411-methods-of-estimating-candidate-grasps)
        * [4.1.2 Methods of Transferring Grasps](#412-methods-of-transferring-grasps)
           * [a. Grasp transfer](#a-grasp-transfer)
           * [b. Non-rigid registration](#b-non-rigid-registration)
           * [c. Shape correspondence](#c-shape-correspondence)
     * [4.2 Methods based on Complete Shape](#42-methods-based-on-complete-shape)
        * [4.2.1 Methods of Estimating 6D Object Pose](#421-methods-of-estimating-6d-object-pose)
        * [4.2.2 Methods of Shape Completion](#422-methods-of-shape-completion)
           * [a. Shape Completion-based Grasp](#a-shape-completion-based-grasp)
           * [b. Shape Completion or Generation](#b-shape-completion-or-generation)
           * [c. Depth Completion and Estimation](#c-depth-completion-and-estimation)
           * [d. Point Cloud Denoising and Samping](#d-point-cloud-denoising-and-samping)
  * [5. Task-oriented Methods](#5-task-oriented-methods)
     * [5.1 Task-oriented Manipulation](#51-task-oriented-manipulation)
     * [5.2 Grasp Affordance](#52-grasp-affordance)
     * [5.3 3D Part Segmentation](#53-3d-part-segmentation)
  * [6. Dexterous Grippers](#6-dexterous-grippers)
  * [7. Data Generation](#7-data-generation)
     * [7.1 Simulation to Reality](#71-simulation-to-reality)
     * [7.2 Self-supervised Methods](#72-self-supervised-methods)
  * [8. Multi-source](#8-multi-source)
  * [9. Motion Planning](#9-motion-planning)
     * [9.1 Visual servoing](#91-visual-servoing)
     * [9.2 Path Planning](#92-path-planning)
  * [10. Imitation Learning](#10-imitation-learning)
  * [11. Reinforcement Learning](#11-reinforcement-learning)
  * [12. Experts](#12-experts)

## 0. Review Papers

**[arXiv]** 2020-Affordances in Robotic Tasks - A Survey, [[paper](https://arxiv.org/pdf/2004.07400.pdf)]

**[arXiv]** 2019-A Review of Robot Learning for Manipulation- Challenges, Representations, and Algorithms, [[paper](https://arxiv.org/abs/1907.03146)]

**[arXiv]** 2019-Vision-based Robotic Grasping from Object Localization, Pose Estimation, Grasp Detection to Motion Planning: A Review, [[paper](https://arxiv.org/abs/1905.06658)]

**[arXiv]** 2018-The Limits and Potentials of Deep Learning for Robotics, [[paper](https://arxiv.org/abs/1804.06557)]

**[MTI]** 2018-Review of Deep Learning Methods in Robotic Grasp Detection, [[paper](https://www.mdpi.com/2414-4088/2/3/57)]

**[ToR]** 2016-Data-Driven Grasp Synthesis - A Survey, [[paper](https://arxiv.org/abs/1309.2660)]

**[RAS]** 2012-An overview of 3D object grasp synthesis algorithms - A Survey, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0921889011001485)]



## 1. Object Localization

### 1.1 Object Localization without Classification



#### 1.1.1 2D-based Methods

##### a.Fitting 2D Shape Primitives

**[BMVC]** A buyer’s guide to conic fitting, [[paper](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.36.695)] [[code](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=fitellipse)]

**[IJGIG]** Algorithms for the reduction of the number of points required to represent a digitized line or its caricature, [[paper](https://pdfs.semanticscholar.org/e46a/c802d7207e0e51b5333456a3f46519c2f92d.pdf)] [[code](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=fitellipse#approxpolydp)]



##### b. Saliency Detection

***Survey papers:***

**[arXiv]** 2019-Salient object detection in the deep learning era: An in-depth survey, [[paper](https://arxiv.org/pdf/1904.09146.pdf)]

**[CVM]** 2014-Salient object detection: A survey, [[paper](https://arxiv.org/pdf/1411.5878.pdf)]

***2020:***

**[arXiv]** Bilateral Attention Network for RGB-D Salient Object Detection, [[paper](https://arxiv.org/pdf/2004.14582.pdf)]

**[arXiv]** Salient Object Detection Combining a Self-attention Module and a Feature Pyramid Network, [[paper](https://arxiv.org/pdf/2004.14552.pdf)]

**[arXiv]** JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection, [[paper](https://arxiv.org/pdf/2004.08515.pdf)]

**[arXiv]** UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders, [[paper](https://arxiv.org/pdf/2004.05763.pdf)]

**[arXiv]** Cross-layer Feature Pyramid Network for Salient Object Detection, [[paper](https://arxiv.org/pdf/2002.10864.pdf)]

**[arXiv]** Depth Potentiality-Aware Gated Attention Network for RGB-D Salient Object Detection, [[paper](https://arxiv.org/pdf/2003.08608.pdf)]

**[arXiv]** Weakly-Supervised Salient Object Detection via Scribble Annotations, [[paper](https://arxiv.org/pdf/2003.07685.pdf)]

**[arXiv]** Highly Efficient Salient Object Detection with 100K Parameters, [[paper](https://arxiv.org/pdf/2003.05643.pdf)]

**[arXiv]** Global Context-Aware Progressive Aggregation Network for Salient Object Detection, [[paper](https://arxiv.org/pdf/2003.00651.pdf)]

**[arXiv]** Adaptive Graph Convolutional Network with Attention Graph Clustering for Co-saliency Detection, [[paper](https://arxiv.org/pdf/2003.06167.pdf)]

***2019:***

**[ICCV]** Employing deep part-object relationships for salient object detection, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Employing_Deep_Part-Object_Relationships_for_Salient_Object_Detection_ICCV_2019_paper.pdf)]

**[ICME]** Multi-scale capsule attention-based salient object detection with multi-crossed layer connections, [[paper](https://ieeexplore.ieee.org/document/8784786)]

***2018:***

**[CVPR]** Picanet: Learning pixel-wise contextual attention for saliency detection, [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_PiCANet_Learning_Pixel-Wise_CVPR_2018_paper.pdf)]

**[SPM]** Advanced deep-learning techniques for salient and category-specific object detection: a survey, [[paper](https://ieeexplore.ieee.org/abstract/document/8253582)]

***2017:***

**[CVPR]** Deeply supervised salient object detection with short connections, [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hou_Deeply_Supervised_Salient_CVPR_2017_paper.pdf)]

**[TOC]** Video saliency detection using object proposals, [[paper](https://ueaeprints.uea.ac.uk/id/eprint/65433/1/Accepted_manuscript.pdf)]

***2016:***

**[CVPR]** Unconstrained salient object detection via proposal subset optimization, [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Unconstrained_Salient_Object_CVPR_2016_paper.pdf)]

**[CVPR]** Deep hierarchical saliency network for salient object detection, [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf)]

**[TPAMI]** Salient object detection via structured matrix decomposition, [[paper](https://eprints.bbk.ac.uk/14986/1/14986.pdf)]

**[TIP]** Correspondence driven saliency transfer, [[paper](https://ueaeprints.uea.ac.uk/id/eprint/62909/1/Accepted_manuscript.pdf)] 

***2015:***

**[CVPR]** Saliency detection by multi-context deep learning, [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhao_Saliency_Detection_by_2015_CVPR_paper.pdf)]

**[TPAMI]** Hierarchical image saliency detection on extended CSSD, [[paper](https://arxiv.org/pdf/1408.5418.pdf)]

***2014:***

**[CVPR]** Saliency optimization from robust background detection, [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Zhu_Saliency_Optimization_from_2014_CVPR_paper.pdf)]

**[TPAMI]** Global contrast based salient region detection, [[paper](https://ieeexplore.ieee.org/abstract/document/6871397)]

***2013:***

**[CVPR]** Salient object detection: A discriminative regional feature integration approach, [[paper](https://arxiv.org/pdf/1410.5926.pdf)]

**[CVPR]** Saliency detection via graph-based manifold ranking, [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Yang_Saliency_Detection_via_2013_CVPR_paper.pdf)]

***2012:***

**[ECCV]** Geodesic saliency using background priors, [[paper](http://jiansun.org/papers/ECCV12_GeodesicSaliency.pdf)]



#### 1.1.2 3D-based Methods

##### a.Fitting 3D Shape Primitives

***Survey papers:***

**[CGF]** 2019-A survey of simple geometric primitives detection methods for captured 3d data, [[paper](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13451)]

***2020:***

**[ICRA]** PrimiTect: Fast Continuous Hough Voting for Primitive Detection, [[paper](https://arxiv.org/pdf/2005.07457.pdf)] [[code](https://github.com/c-sommer/primitect)]

**[arXiv]** ParSeNet: A Parametric Surface Fitting Network for 3D Point Clouds, [[paper](https://arxiv.org/pdf/2003.12181.pdf)]

***2015:***

**[CVPR]** Separating objects and clutter in indoor scenes, [[paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Khan_Separating_Objects_and_2015_CVPR_paper.pdf)]

***2013:***

**[CVPR]** A linear approach to matching cuboids in rgbd images, [[paper](https://ieeexplore.ieee.org/document/6619126)]

***2012:***

**[GCR]** Robustly segmenting cylindrical and box-like objects in cluttered scenes using depth cameras, [[paper](https://ieeexplore.ieee.org/document/6309565)]

***2009:***

**[IROS]** Close-range scene segmentation and reconstruction of 3d point cloud maps for mobile manipulation in domestic environments, [[paper](https://ieeexplore.ieee.org/document/5354683)]

***2005:***

**[ISPRS]** Efficient hough transform for automatic detection of cylinders in point clouds, [[paper](https://www.isprs.org/proceedings/XXXVI/3-W19/papers/060.pdf)]

***1981:***

**[COM]** Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography, [[paper](http://www.cs.ait.ac.th/~mdailey/cvreadings/Fischler-RANSAC.pdf)]



##### b. Saliency Detection

***2019:***

**[PR]** Multi-modal fusion network with multi-scale multi-path and cross-modal interactions for RGB-D salient object detection, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320318303054)]

**[ICCV]** Depth-Induced Multi-Scale Recurrent Attention Network for Saliency Detection, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Piao_Depth-Induced_Multi-Scale_Recurrent_Attention_Network_for_Saliency_Detection_ICCV_2019_paper.pdf)]

**[ICCV]** Pointcloud saliency maps, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zheng_PointCloud_Saliency_Maps_ICCV_2019_paper.pdf)]

**[arXiv]** CNN-based RGB-D Salient Object Detection: Learn, Select and Fuse, [[paper](https://arxiv.org/pdf/1909.09309.pdf)]

***2018:***

**[CVPR]** Progressively complementarity-aware fusion network for RGB-D salient object detection, [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Progressively_Complementarity-Aware_Fusion_CVPR_2018_paper.pdf)]

***2017:***

**[TIP]** RGBD salient object detection via deep fusion, [[paper](https://arxiv.org/pdf/1607.03333.pdf)]

***2015:***

**[CVPRW]** Exploiting global priors for RGB-D saliency detection, [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W14/papers/Ren_Exploiting_Global_Priors_2015_CVPR_paper.pdf)]

***2014:***

**[ECCV]** Rgbd salient object detection: a benchmark and algorithms, [[paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ECCV-2014/papers/8691/86910092.pdf)]

***2013:***

**[JSIP]** Segmenting salient objects in 3d point clouds of indoor scenes using geodesic distances, [[paper](https://www.scirp.org/html/38079.html)]

***2008:***

**[WACV]** Segmentation of salient regions in outdoor scenes using imagery and 3-d data, [[paper](https://ieeexplore.ieee.org/abstract/document/4544014)]



### 1.2 Object Detection

Detailed paper lists can refer to [hoya012](https://github.com/hoya012/deep_learning_object_detection) or [amusi](https://github.com/amusi/awesome-object-detection).

#### 1.2.1 2D Object Detection

***Survey papers:***

***2020:***

**[arXiv]** Deep Domain Adaptive Object Detection: a Survey, [[paper](https://arxiv.org/pdf/2002.06797.pdf)]

**[IJCV]** Deep Learning for Generic Object Detection: A Survey, [[paper](https://link.springer.com/content/pdf/10.1007%2Fs11263-019-01247-4.pdf)]

***2019:***

**[arXiv]** Object Detection in 20 Years A Survey, [[paper](https://arxiv.org/pdf/1905.05055.pdf)]

**[arXiv]** Object Detection with Deep Learning: A Review, [[paper](https://arxiv.org/pdf/1807.05511.pdf)]

**[arXiv]** A Review of Object Detection Models based on Convolutional Neural Network, [[paper](https://arxiv.org/pdf/1905.01614.pdf)]

**[arXiv]** A Review of methods for Textureless Object Recognition, [[paper](https://arxiv.org/abs/1910.14255)]

##### a. Two-stage methods

***2020:***

**[arXiv]** Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation, [[paper](https://arxiv.org/pdf/2005.03572.pdf)]

**[arXiv]** Instance-Aware, Context-Focused, and Memory-Efficient Weakly Supervised Object Detection, [[paper](https://arxiv.org/pdf/2004.04725.pdf)]

**[arXiv]** Scalable Active Learning for Object Detection, [[paper](https://arxiv.org/pdf/2004.04699.pdf)]

**[arXiv]** Any-Shot Object Detection, [[paper](https://arxiv.org/pdf/2003.07003.pdf)]

**[arXiv]** Frustratingly Simple Few-Shot Object Detection, [[paper](https://arxiv.org/pdf/2003.06957.pdf)]

**[arXiv]** Rethinking the Route Towards Weakly Supervised Object Localization, [[paper](https://arxiv.org/pdf/2002.11359.pdf)]

**[arXiv]** Universal-RCNN: Universal Object Detector via Transferable Graph R-CNN, [[paper](https://arxiv.org/pdf/2002.07417.pdf)]

**[arXiv]** Unsupervised Image-generation Enhanced Adaptation for Object Detection in Thermal images, [[paper](https://arxiv.org/pdf/2002.06770.pdf)]

**[arXiv]** PCSGAN: Perceptual Cyclic-Synthesized Generative Adversarial Networks for Thermal and NIR to Visible Image Transformation, [[paper](https://arxiv.org/pdf/2002.07082.pdf)]

**[arXiv]** SpotNet: Self-Attention Multi-Task Network for Object Detection, [[paper](https://arxiv.org/pdf/2002.05540.pdf)]

**[arXiv]** Real-Time Object Detection and Recognition on Low-Compute Humanoid Robots using Deep Learning, [[paper](https://arxiv.org/pdf/2002.03735.pdf)]

**[arXiv]** FedVision: An Online Visual Object Detection Platform Powered by Federated Learning, [[paper](https://arxiv.org/pdf/2001.06202.pdf)]

***2019:***

**[arXiv]** Combining Deep Learning and Verification for Precise Object Instance Detection, [[paper](https://arxiv.org/pdf/1912.12270.pdf)]

**[arXiv]** cmSalGAN: RGB-D Salient Object Detection with Cross-View Generative Adversarial Networks, [[paper](https://arxiv.org/pdf/1912.10280.pdf)]

**[arXiv]** OpenLORIS-Object: A Dataset and Benchmark towards Lifelong Object Recognition, [[paper](https://arxiv.org/abs/1911.06487)] [[project](https://lifelong-robotic-vision.github.io/dataset/Data_Object-Recognition.html)]

**[IROS]** Look Further to Recognize Better: Learning Shared Topics and Category-Specific Dictionaries for Open-Ended 3D Object Recognition, [[paper](https://arxiv.org/abs/1907.12924)]

**[IROS]** Recurrent Convolutional Fusion for RGB-D Object Recognition, [[paper](https://arxiv.org/pdf/1806.01673.pdf)] [[code](https://github.com/MRLoghmani/rcfusion)]

**[ICCVW]** An Annotation Saved is an Annotation Earned: Using Fully Synthetic Training for Object Detection, [[paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/R6D/Hinterstoisser_An_Annotation_Saved_is_an_Annotation_Earned_Using_Fully_Synthetic_ICCVW_2019_paper.pdf)]

***2017:***

**[CVPR]** FPN: Feature pyramid networks for object detection, [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)]

**[arXiv]** Light-Head R-CNN: In Defense of Two-Stage Object Detector, [[paper](https://arxiv.org/pdf/1711.07264.pdf)] [[code](https://github.com/zengarden/light_head_rcnn)]

***2016:***

**[NeurIPS]** R-FCN: Object Detection via Region-based Fully Convolutional Networks, [[paper](https://arxiv.org/pdf/1605.06409.pdf)] [[code](https://github.com/daijifeng001/R-FCN)]

**[TPAMI]** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, [[paper](https://arxiv.org/abs/1506.01497)] [[code](https://github.com/rbgirshick/py-faster-rcnn)]

**[ECCV]** Visual relationship detection with language priors, [[paper](https://arxiv.org/pdf/1608.00187.pdf)]

***2015:***

**[ICCV]** Fast R-CNN, [[paper](https://arxiv.org/pdf/1504.08083.pdf)] [[code](https://github.com/rbgirshick/fast-rcnn)]

***2014:***

**[ECCV]** SPPNet: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, [[paper](https://arxiv.org/pdf/1406.4729.pdf)] [[code](https://github.com/ShaoqingRen/SPP_net)]

**[CVPR]** R-CNN: Rich feature hierarchies for accurate object detection and semantic segmentation, [[paper](https://arxiv.org/pdf/1311.2524.pdf)] [[code](https://github.com/rbgirshick/rcnn)]

**[CVPR]** Scalable object detection using deep neural networks, [[paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Erhan_Scalable_Object_Detection_2014_CVPR_paper.pdf)]

**[arXiv]** Scalable, high-quality object detection, [[paper](https://arxiv.org/pdf/1412.1441.pdf)]

**[ICLR]** OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks, [[paper](https://arxiv.org/pdf/1312.6229.pdf)] [[code](https://github.com/sermanet/OverFeat)]

***2011:***

**[ICCV]** ORB: An efficient alternative to SIFT or SURF, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.370.4395&rep=rep1&type=pdf)]

***2006:***

**[ECCV]** SURF: Speeded up robust features, [[paper](https://link.springer.com/chapter/10.1007/11744023_32)]

***2005:***

**[ICCV]** FAST: Fusing points and lines for high performance tracking, [[paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.60.4715&rep=rep1&type=pdf)]

***1999:***

**[ICCV]** SIFT: Object Recognition from Local Scale-Invariant Features, [[paper](https://ieeexplore.ieee.org/abstract/document/790410)]



##### b. Single-stage methods

***2020:***

**[arXiv]** YOLOv4: Optimal Speed and Accuracy of Object Detection, [[paper](https://arxiv.org/pdf/2004.10934.pdf)]

**[arXiv]** SaccadeNet: A Fast and Accurate Object Detector, [[paper](https://arxiv.org/pdf/2003.12125.pdf)]

**[arXiv]** CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection, [[paper](https://arxiv.org/pdf/2003.09119.pdf)]

**[arXiv]** Real Time Detection of Small Objects, [[paper](https://arxiv.org/pdf/2003.07442.pdf)]

**[arXiv]** OS2D: One-Stage One-Shot Object Detection by Matching Anchor Features, [[paper](https://arxiv.org/pdf/2003.06800.pdf)]

***2019:***

**[arXiv]** CenterNet: Objects as Points, [[paper](https://arxiv.org/pdf/1904.07850.pdf)]

**[arXiv]** CenterNet: Keypoint Triplets for Object Detection, [[paper](https://arxiv.org/pdf/1904.08189.pdf)]

**[ECCV]** CornerNet: Detecting Objects as Paired Keypoints, [[paper](https://arxiv.org/pdf/1808.01244.pdf)]

**[arXiv]** FCOS: Fully Convolutional One-Stage Object Detection, [[paper](https://arxiv.org/pdf/1904.01355.pdf)]

**[arXiv]** Bottom-up Object Detection by Grouping Extreme and Center Points, [[paper](https://arxiv.org/pdf/1901.08043.pdf)]

***2018:***

**[arXiv]** YOLOv3: An Incremental Improvement, [[paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)] [[code](https://github.com/eriklindernoren/PyTorch-YOLOv3)]

***2017:***

**[CVPR]** YOLO9000: Better, Faster, Stronger, [[paper](https://arxiv.org/pdf/1612.08242.pdf)] [[code](https://github.com/longcw/yolo2-pytorch)]

**[ICCV]** RetinaNet: Focal loss for dense object detection, [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)]

***2016:***

**[CVPR]** YOLO: You only look once: Unified, real-time object detection, [[paper](https://arxiv.org/abs/1506.02640)] [[code](https://github.com/gliese581gg/YOLO_tensorflow)]

**[ECCV]** SSD: Single Shot MultiBox Detector, [[paper](https://arxiv.org/abs/1512.02325)] [[code](https://github.com/balancap/SSD-Tensorflow)]



------

***Dataset:***

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/): The PASCAL Visual Object Classes (VOC) Challenge, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf)]

[ILSVRC](http://www.image-net.org/challenges/LSVRC/): ImageNet large scale visual recognition challenge, [[paper](https://arxiv.org/pdf/1409.0575.pdf)]

[Microsoft COCO](http://cocodataset.org/): Common Objects in Context, is a large-scale object detection, segmentation, and captioning dataset, [[paper](https://arxiv.org/pdf/1405.0312.pdf)]

[Open Images](https://storage.googleapis.com/openimages/web/index.html): a collaborative release of ~9 million images annotated with labels spanning thousands of object categories, [[paper](https://arxiv.org/pdf/1811.00982v1.pdf)]



#### 1.2.2 3D Object Detection

This kind of methods can be divided into three kinds: RGB-based methods, point cloud-based methods, and fusion methods which consume images and point cloud. Most of these works are focus on autonomous driving.

##### a. RGB-based methods

Most of this kind of methods estimate depth images from RGB images, and then conduct 3D detection.

***2020:***

**[arXiv]** Center3D: Center-based Monocular 3D Object Detection with Joint Depth Understanding, [[paper](https://arxiv.org/pdf/2005.13423.pdf)]

**[ICITS]** Exploring the Capabilities and Limits of 3D Monocular Object Detection - A Study on Simulation and Real World Data, [[paper](https://arxiv.org/pdf/2005.07424.pdf)]

**[arXiv]** Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation, [[paper](https://arxiv.org/pdf/2004.03572.pdf)]

**[arXiv]** End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection, [[paper](https://arxiv.org/pdf/2004.03080.pdf)]

**[arXiv]** Confidence Guided Stereo 3D Object Detection with Split Depth Estimation, [[paper](https://arxiv.org/pdf/2003.05505.pdf)]

**[arXiv]** Monocular 3D Object Detection in Cylindrical Images from Fisheye Cameras, [[paper](https://arxiv.org/pdf/2003.03759.pdf)]

**[arXiv]** ZoomNet: Part-Aware Adaptive Zooming Neural Network for 3D Object Detection, [[paper](https://arxiv.org/pdf/2003.00529.pdf)]

**[arXiv]** MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, [[paper](https://arxiv.org/pdf/2003.00504.pdf)]

**[arXiv]** Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes from a Single Image, [[paper](https://arxiv.org/pdf/2002.12212.pdf)]

**[arXiv]** SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation, [[paper](https://arxiv.org/pdf/2002.10111.pdf)]

**[arXiv]** siaNMS: Non-Maximum Suppression with Siamese Networks for Multi-Camera 3D Object Detection, [[paper](https://arxiv.org/pdf/2002.08239.pdf)]

**[AAAI]** Monocular 3D Object Detection with Decoupled Structured Polygon Estimation and Height-Guided Depth Estimation, [[paper](https://arxiv.org/pdf/2002.01619.pdf)]

**[arXiv]** SDOD: Real-time Segmenting and Detecting 3D Objects by Depth, [[paper](https://arxiv.org/pdf/2001.09425.pdf)]

**[arXiv]** DSGN: Deep Stereo Geometry Network for 3D Object Detection, [[paper](https://arxiv.org/pdf/2001.03398.pdf)]

**[arXiv]** RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving, [[paper](https://arxiv.org/pdf/2001.03343.pdf)]

***2019:***

**[NeurIPS]** PerspectiveNet: 3D Object Detection from a Single RGB Image via Perspective Points, [[paper](https://arxiv.org/abs/1912.07744)]

**[arXiv]** Single-Stage Monocular 3D Object Detection with Virtual Cameras, [[paper](https://arxiv.org/abs/1912.08035)]

**[arXiv]** Environment reconstruction on depth images using Generative Adversarial Networks, [[paper](https://arxiv.org/abs/1912.03992)] [[code](https://github.com/nuneslu/VeIGAN)]

**[arXiv]** Learning Depth-Guided Convolutions for Monocular 3D Object Detection, [[paper](https://arxiv.org/abs/1912.04799)]

**[arXiv]** RefinedMPL: Refined Monocular PseudoLiDAR for 3D Object Detection in Autonomous Driving, [[paper](https://arxiv.org/abs/1911.09712)]

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

This kind of methods only consume the 3D point cloud data.

***Survey papers:***

**[arXiv]** 2019-Deep Learning for 3D Point Clouds: A Survey, [[paper](https://arxiv.org/pdf/1912.12033.pdf)]



***2020:***

**[arXiv]** P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds, [[paper](https://arxiv.org/pdf/2005.13888.pdf)]

**[arXiv]** Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection, [[paper](https://arxiv.org/pdf/2005.09927.pdf)]

**[arXiv]** Drosophila-Inspired 3D Moving Object Detection Based on Point Clouds, [[paper](https://arxiv.org/pdf/2005.02696.pdf)]

**[arXiv]** Streaming Object Detection for 3-D Point Clouds, [[paper](https://arxiv.org/pdf/2005.01864.pdf)]

**[arXiv]** SS3D: Single Shot 3D Object Detector, [[paper](https://arxiv.org/pdf/2004.14674.pdf)]

**[arXiv]** MLCVNet: Multi-Level Context VoteNet for 3D Object Detection, [[paper](https://arxiv.org/pdf/2004.05679.pdf)]

**[arXiv]** 3D IoU-Net: IoU Guided 3D Object Detector for Point Clouds, [[paper](https://arxiv.org/pdf/2004.04962.pdf)]

**[arXiv]** Finding Your (3D) Center: 3D Object Detection Using a Learned Loss, [[paper](https://arxiv.org/pdf/2004.02693.pdf)]

**[arXiv]** LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention, [[paper](https://arxiv.org/pdf/2004.01389.pdf)]

**[arXiv]** Quantifying Data Augmentation for LiDAR based 3D Object Detection, [[paper](https://arxiv.org/pdf/2004.01643.pdf)]

**[arXiv]** DOPS: Learning to Detect 3D Objects and Predict their 3D Shapes, [[paper](https://arxiv.org/pdf/2004.01170.pdf)]

**[arXiv]** Improving 3D Object Detection through Progressive Population Based Augmentation, [[paper](https://arxiv.org/pdf/2004.00831.pdf)]

**[arXiv]** Boundary-Aware Dense Feature Indicator for Single-Stage 3D Object Detection from Point Clouds, [[paper](https://arxiv.org/pdf/2004.00186.pdf)]

**[arXiv]** Physically Realizable Adversarial Examples for LiDAR Object Detection, [[paper](https://arxiv.org/pdf/2004.00543.pdf)]

**[arXiv]** Real-time 3D object proposal generation and classification under limited processing resources, [[paper](https://arxiv.org/pdf/2003.10670.pdf)]

**[arXiv]** 3D Object Detection From LiDAR Data Using Distance Dependent Feature Extraction, [[paper](https://arxiv.org/pdf/2003.00888.pdf)]

**[arXiv]** HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection, [[paper](https://arxiv.org/pdf/2003.00186.pdf)]

**[arXiv]** Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud, [[paper](https://arxiv.org/pdf/2003.01251.pdf)]

**[arXiv]** PointTrackNet: An End-to-End Network for 3-D Object Detection and Tracking from Point Clouds, [[paper](https://arxiv.org/pdf/2002.11559.pdf)]

**[arXiv]** 3DSSD: Point-based 3D Single Stage Object Detector, [[paper](https://arxiv.org/pdf/2002.10187.pdf)]

**[ariv]** SegVoxelNet: Exploring Semantic Context and Depth-aware Features for 3D Vehicle Detection from Point Cloud, [[paper](https://arxiv.org/pdf/2002.05316.pdf)]

**[arXiv]** Investigating the Importance of Shape Features, Color Constancy, Color Spaces and Similarity Measures in Open-Ended 3D Object Recognition, [[paper](https://arxiv.org/pdf/2002.03779.pdf)]

**[arXiv]** Probabilistic 3D Multi-Object Tracking for Autonomous Driving, [[paper](https://arxiv.org/pdf/2001.05673.pdf)]

**[AAAI]** TANet: Robust 3D Object Detection from Point Clouds with Triple Attention, [[paper](https://arxiv.org/abs/1912.05163)]

***2019:***

**[arXiv]** Class-balanced grouping and sampling for point cloud 3d object detection, [[paper](https://arxiv.org/pdf/1908.09492.pdf)] [[code](https://github.com/poodarchu/Det3D)]

**[arXiv]** SESS: Self-Ensembling Semi-Supervised 3D Object Detection, [[paper](https://arxiv.org/pdf/1912.11803.pdf)]

**[arXiv]** Deep SCNN-based Real-time Object Detection for Self-driving Vehicles Using LiDAR Temporal Data, [[paper](https://arxiv.org/abs/1912.07906)]

**[arXiv]** Pillar in Pillar: Multi-Scale and Dynamic Feature Extraction for 3D Object Detection in Point Clouds, [[paper](https://arxiv.org/abs/1912.04775)]

**[arXiv]** What You See is What You Get: Exploiting Visibility for 3D Object Detection, [[paper](https://arxiv.org/abs/1912.04986)]

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

**[arXiv]** 3D Object Detection Method Based on YOLO and K-Means for Image and Point Clouds, [[paper](https://arxiv.org/pdf/2005.02132.pdf)]

**[arXiv]** 3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection, [[paper](https://arxiv.org/pdf/2004.12636.pdf)]

**[arXiv]** ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes, [[paper](https://arxiv.org/pdf/2001.10692.pdf)]

**[arXiv]** JRMOT: A Real-Time 3D Multi-Object Tracker and a New Large-Scale Dataset, [[paper](https://arxiv.org/pdf/2002.08397.pdf)]

**[AAAI]** PI-RCNN: An Efficient Multi-sensor 3D Object Detector with Point-based Attentive Cont-conv Fusion Module, [[paper](https://arxiv.org/abs/1911.06084)]

***2019:***

**[arXiv]** PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection, [[paper](https://arxiv.org/pdf/1912.13192.pdf)]

**[arXiv]** Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots, [[paper](https://arxiv.org/pdf/1912.12791.pdf)]

**[arXiv]** ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language, [[paper](https://arxiv.org/abs/1912.08830)]

**[arXiv]** Relation Graph Network for 3D Object Detection in Point Clouds, [[paper](https://arxiv.org/abs/1912.00202)]

**[arXiv]** PointPainting: Sequential Fusion for 3D Object Detection, [[paper](https://arxiv.org/abs/1911.10150)]

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

***2016:***

**[CVPR]** Deep sliding shapes for amodal 3d object detection in rgb-d images, [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Sliding_Shapes_CVPR_2016_paper.pdf)]



### 1.3 Object Instance Segmentation

#### 1.3.1 2D Instance Segmentation

##### a. Survey papers

***2020:***

**[arXiv]** Evolution of Image Segmentation using Deep Convolutional Neural Network: A Survey, [[paper](https://arxiv.org/pdf/2001.04074.pdf)]

**[arXiv]** Image Segmentation Using Deep Learning: A Survey, [[paper](https://arxiv.org/pdf/2001.05566.pdf)]

##### b. Two-stage methods

***2020:***

**[arXiv]** A novel Region of Interest Extraction Layer for Instance Segmentation, [[paper](https://arxiv.org/pdf/2004.13665.pdf)]

**[arXiv]** 1st Place Solutions for OpenImage2019 - Object Detection and Instance Segmentation, [[paper](https://arxiv.org/pdf/2003.07557.pdf)]

**[arXiv]** Fully Convolutional Networks for Automatically Generating Image Masks to Train Mask R-CNN, [[paper](https://arxiv.org/pdf/2003.01383.pdf)]

**[arXiv]** FGN: Fully Guided Network for Few-Shot Instance Segmentation, [[paper](https://arxiv.org/pdf/2003.13954.pdf)]

**[arXiv]** PointRend: Image Segmentation as Rendering, [[paper](https://arxiv.org/abs/1912.08193)]

***2019:***

**[CVPR]** HTC: Hybrid task cascade for instance segmentation, [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Hybrid_Task_Cascade_for_Instance_Segmentation_CVPR_2019_paper.pdf)]

***2018:***

**[CVPR]** PANet: Path aggregation network for instance segmentation, [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Path_Aggregation_Network_CVPR_2018_paper.pdf)]

**[CVPR]** Masklab: Instance segmentation by refining object detection with semantic and direction features, [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_MaskLab_Instance_Segmentation_CVPR_2018_paper.pdf)]

***2017:***

**[ICCV]** Mask r-cnn, [[paper](https://arxiv.org/abs/1703.06870)] [[code](https://github.com/matterport/Mask_RCNN)]

***2016:***

**[CVPR]** Instance-aware semantic segmentation via multi-task network cascades, [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.pdf)]

***2014:***

**[ECCV]** Simultaneous detection and segmentation, [[paper](https://arxiv.org/pdf/1407.1808.pdf)]



##### c. One-stage methods

***2020:***

**[arXiv]** POLY-YOLO: HIGHER SPEED, MORE PRECISE DETECTION AND INSTANCE SEGMENTATION FOR YOLOV3, [[paper](https://arxiv.org/pdf/2005.13243.pdf)]

**[CVPR]** CenterMask: single shot instance segmentation with point representation, [[paper](https://arxiv.org/pdf/2004.04446.pdf)]

**[arXiv]** BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation, [[paper](https://arxiv.org/pdf/2001.00309.pdf)]

**[arXiv]** SOLOv2: Dynamic, Faster and Stronger, [[paper](https://arxiv.org/pdf/2003.10152.pdf)] [[code](https://github.com/aim-uofa/AdelaiDet/)]

**[arXiv]** Mask Encoding for Single Shot Instance Segmentation, [[paper](https://arxiv.org/pdf/2003.11712.pdf)]

**[arXiv]** Deep Affinity Net: Instance Segmentation via Affinity, [[paper](https://arxiv.org/pdf/2003.06849.pdf)]

**[arXiv]** PointINS: Point-based Instance Segmentation, [[paper](https://arxiv.org/pdf/2003.06148.pdf)]

**[arXiv]** Conditional Convolutions for Instance Segmentation, [[paper](https://arxiv.org/pdf/2003.05664.pdf)]

**[arXiv]** Real-time Semantic Background Subtraction, [[paper](https://arxiv.org/pdf/2002.04993.pdf)]

**[arXiv]** FourierNet: Compact mask representation for instance segmentation using differentiable shape decoders, [[paper](https://arxiv.org/pdf/2002.02709.pdf)]

***2019:***

**[arXiv]** CenterMask:Real-Time Anchor-Free Instance Segmentation, [[paper](https://arxiv.org/pdf/1911.06667.pdf)] [[code](https://github.com/youngwanLEE/centermask2)]

**[arXiv]** SAIS: Single-stage Anchor-free Instance Segmentation, [[paper](https://arxiv.org/pdf/1912.01176.pdf)]

**[arXiv]** YOLACT++ Better Real-time Instance Segmentation, [[paper](https://arxiv.org/pdf/1912.06218.pdf)] [[code](https://github.com/dbolya/yolact)]

**[ICCV]** YOLACT: Real-time Instance Segmentation, [[paper](https://arxiv.org/pdf/1904.02689.pdf)] [[code](https://github.com/dbolya/yolact)]

**[ICCV]** TensorMask: A Foundation for Dense Object Segmentation, [[paper](https://arxiv.org/pdf/1903.12174.pdf)] [[code](https://github.com/facebookresearch/detectron2/tree/master/projects/TensorMask)]

**[CASE]** Deep Workpiece Region Segmentation for Bin Picking, [[paper](https://arxiv.org/abs/1909.03462)]

***2018:***

**[CVPR]** PANet: Path Aggregation Network for Instance Segmentation, [[paper](https://arxiv.org/pdf/1803.01534.pdf)] [[code](https://github.com/ShuLiu1993/PANet)]

**[CVPR]** MaskLab: Instance Segmentation by Refining Object Detection with Semantic and Direction Features, [[paper](https://arxiv.org/pdf/1712.04837.pdf)]

***2017:***

**[CVPR]** Fully Convolutional Instance-aware Semantic Segmentation, [[paper](https://arxiv.org/pdf/1611.07709.pdf)]

***2016:***

**[ECCV]** SharpMask: Learning to Refine Object Segments, [[paper](https://arxiv.org/pdf/1603.08695.pdf)] [[code](https://github.com/facebookresearch/deepmask)]

**[BMVC]** MultiPathNet: A MultiPath Network for Object Detection, [[paper](https://arxiv.org/pdf/1604.02135.pdf)] [[code](https://github.com/facebookresearch/multipathnet)]

**[CVPR]** MNC: Instance-aware Semantic Segmentation via Multi-task Network Cascades, [[paper](https://arxiv.org/pdf/1512.04412.pdf)]

***2015:***

**[NeurIPS]** DeepMask: Learning to Segment Object Candidates, [[paper](https://arxiv.org/pdf/1506.06204.pdf)] [[code](https://github.com/facebookresearch/deepmask)]

**[CVPR]** Hypercolumns for Object Segmentation and Fine-grained Localization, [[paper](https://arxiv.org/pdf/1411.5752.pdf)]

***2014:***

**[ECCV]** SDS: Simultaneous Detection and Segmentation, [[paper](https://arxiv.org/pdf/1407.1808.pdf)]



***Applications in Robotics:***

***2020:***

**[arXiv]** Self-Supervised Object-in-Gripper Segmentation from Robotic Motions, [[paper](https://arxiv.org/pdf/2002.04487.pdf)]

**[arXiv]** Segmenting unseen industrial components in a heavy clutter using rgb-d fusion and synthetic data, [[paper](https://arxiv.org/pdf/2002.03501.pdf)]

**[arXiv]** Instance Segmentation of Visible and Occluded Regions for Finding and Picking Target from a Pile of Objects, [[paper](https://arxiv.org/pdf/2001.07475.pdf)]

**[arXiv]** Joint Learning of Instance and Semantic Segmentation for Robotic Pick-and-Place with Heavy Occlusions in Clutter, [[paper](https://arxiv.org/pdf/2001.07481.pdf)]



##### d. Panoptic segmentation

***2020:***

**[arXiv]** BANet: Bidirectional Aggregation Network with Occlusion Handling for Panoptic Segmentation, [[paper](https://arxiv.org/pdf/2003.14031.pdf)]

**[arXiv]** Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation, [[paper](https://arxiv.org/pdf/2003.07853.pdf)]

**[arXiv]** Towards Bounding-Box Free Panoptic Segmentation, [[paper](https://arxiv.org/pdf/2002.07705.pdf)]

***2019:***

**[CVPR]** An End-to-End Network for Panoptic Segmentation, [[paper](https://arxiv.org/pdf/1903.05027.pdf)]

**[CVPR]** Panoptic Segmentation, [[paper](https://arxiv.org/pdf/1801.00868.pdf)]

**[CVPR]** Panoptic Feature Pyramid Networks, [[paper](https://arxiv.org/pdf/1901.02446.pdf)] 

**[CVPR]** UPSNet: A Unified Panoptic Segmentation Network, [[paper](https://arxiv.org/pdf/1901.03784.pdf)]

**[IV]** Single Network Panoptic Segmentation for Street Scene Understanding, [[paper](https://arxiv.org/pdf/1902.02678.pdf)] [[code](https://github.com/DdeGeus/single-network-panoptic-segmentation)]

**[ITSC]** Multi-task Network for Panoptic Segmentation in Automated Driving, [[paper](https://ieeexplore.ieee.org/abstract/document/8917422)]



#### 1.3.2 3D Instance Segmentation

##### a. Two-stage methods

***2020:***

**[arXiv]** PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation, [[paper](https://arxiv.org/pdf/2004.01658.pdf)]

**[arXiv]** 3D-MPA: Multi Proposal Aggregation for 3D Semantic Instance Segmentation, [[paper](https://arxiv.org/pdf/2003.13867.pdf)]

**[arXiv]** OccuSeg: Occupancy-aware 3D Instance Segmentation, [[paper](https://arxiv.org/pdf/2003.06537.pdf)]

**[arXiv]** Learning to Segment 3D Point Clouds in 2D Image Space, [[paper](https://arxiv.org/pdf/2003.05593.pdf)]

**[arXiv]** Bi-Directional Attention for Joint Instance and Semantic Segmentation in Point Clouds, [[paper](https://arxiv.org/pdf/2003.05420.pdf)]

**[arXiv]** 3DCFS: Fast and Robust Joint 3D Semantic-Instance Segmentation via Coupled Feature Selection, [[paper](https://arxiv.org/pdf/2003.00535.pdf)]

**[RAL]** From Planes to Corners: Multi-Purpose Primitive Detection in Unorganized 3D Point Clouds, [[paper](https://arxiv.org/pdf/2001.07360.pdf)]

**[arXiv]** Learning and Memorizing Representative Prototypes for 3D Point Cloud Semantic and Instance Segmentation, [[paper](https://arxiv.org/pdf/2001.01349.pdf)]

**[WACV]** FuseSeg: LiDAR Point Cloud Segmentation Fusing Multi-Modal Data, [[paper](https://arxiv.org/abs/1912.08487)]

***2019:***

**[arXiv]** Point2Node: Correlation Learning of Dynamic-Node for Point Cloud Feature Modeling, [[paper](https://arxiv.org/pdf/1912.10775.pdf)]

**[arXiv]** LatticeNet: Fast Point Cloud Segmentation Using Permutohedral Lattices, [[paper](https://arxiv.org/abs/1912.05905)]

**[arXiv]** Learning to Optimally Segment Point Clouds, [[paper](https://arxiv.org/abs/1912.04976)]

**[arXiv]** Point Cloud Instance Segmentation using Probabilistic Embeddings, [[paper](https://arxiv.org/abs/1912.00145)]

**[NeurIPS]** Exploiting Local and Global Structure for Point Cloud Semantic Segmentation with Contextual Point Representations, [[paper](https://arxiv.org/abs/1911.05277)]

**[arXiv]** Addressing the Sim2Real Gap in Robotic 3D Object Classification, [[paper](https://arxiv.org/abs/1910.12585)]

**[IROS]** LDLS: 3-D Object Segmentation Through Label Diffusion From 2-D Images, [[paper](https://arxiv.org/abs/1910.13955)]

**[CoRL]** The Best of Both Modes: Separately Leveraging RGB and Depth for Unseen Object Instance Segmentation, [[paper](https://arxiv.org/abs/1907.13236)] [[code](https://arxiv.org/abs/1907.13236)]

**[arXiv]** GSPN: Generative Shape Proposal Network for 3D Instance Segmentation in Point Cloud, [[paper](https://arxiv.org/abs/1812.03320)]



##### b. One-stage Methods

***2020:***

**[AAAI]** JSNet: Joint Instance and Semantic Segmentation of 3D Point Clouds, [[paper](https://arxiv.org/abs/1912.09654)] [[code](https://github.com/dlinzhao/JSNet)]

**[ICRA]** LiDARSeg: Instance segmentation of lidar point clouds, [[paper](http://www.feihuzhang.com/ICRA2020.pdf)]

***2019:***

**[NeurIPS]** 3D-BoNet: Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds, [[paper](https://arxiv.org/pdf/1906.01140.pdf)] [[code](https://github.com/Yang7879/3D-BoNet)]

**[arXiv]** MASC: multi-scale affinity with sparse convolution for 3d instance segmentation, [[paper](https://arxiv.org/pdf/1902.04478.pdf)]

**[CVPR]** ASIS: Associatively segmenting instances and semantics in point clouds, [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Associatively_Segmenting_Instances_and_Semantics_in_Point_Clouds_CVPR_2019_paper.pdf)]

**[CVPR]** SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation, [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SGPN_Similarity_Group_CVPR_2018_paper.pdf)]

**[CVPR]** JSIS3D: joint semantic-instance segmentation of 3d point clouds with multi-task pointwise networks and multi-value conditional random fields, [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pham_JSIS3D_Joint_Semantic-Instance_Segmentation_of_3D_Point_Clouds_With_Multi-Task_CVPR_2019_paper.pdf)]



##### c. 3D deep learning networks

***2020:***

**[arXiv]** PAI-Conv: Permutable Anisotropic Convolutional Networks for Learning on Point Clouds, [[paper](https://arxiv.org/pdf/2005.13135.pdf)]

**[arXiv]** Shape-Oriented Convolution Neural Network for Point Cloud Analysis, [[paper](https://arxiv.org/pdf/2004.09411.pdf)]

**[arXiv]** LightConvPoint: convolution for points, [[paper](https://arxiv.org/pdf/2004.04462.pdf)]

**[arXiv]** Review: deep learning on 3D point clouds, [[paper](https://arxiv.org/pdf/2001.06280.pdf)]

**[arXiv]** Improving Semantic Analysis on Point Clouds via Auxiliary Supervision of Local Geometric Priors, [[paper](https://arxiv.org/pdf/2001.04803.pdf)]

***2019:***

**[arXiv]** QUATERNION EQUIVARIANT CAPSULE NETWORKS FOR 3D POINT CLOUDS, [[paper](https://arxiv.org/pdf/1912.12098.pdf)]

**[arXiv]** Geometry Sharing Network for 3D Point Cloud Classification and Segmentation, [[paper](https://arxiv.org/pdf/1912.10644.pdf)]

**[arXiv]** Geometric Capsule Autoencoders for 3D Point Clouds, [[paper](https://arxiv.org/abs/1912.03310)]

**[arXiv]** Utility Analysis of Network Architectures for 3D Point Cloud Processing, [[paper](https://arxiv.org/abs/1911.09053)]

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



## 2. Object Pose Estimation

This part mainly discuss 6D object pose estimation methods, which can be categorized into __RGB-D image-based methods__ and __point cloud-based methods__. RGB-D image-based methods mainly utilized the 2D RGB image and the 2.5D Depth image. Point cloud-based methods utilize registration-based methods.

### 2.1 RGB-D Image-based Methods

***Survey papers:***

***2020:***

**[arXiv]** A Review on Object Pose Recovery: from 3D Bounding Box Detectors to Full 6D Pose Estimators, [[paper](https://arxiv.org/pdf/2001.10609.pdf)]

***2016:***

**[ECCVW]** A Summary of the 4th International Workshop on Recovering 6D Object Pose, [[paper](https://arxiv.org/abs/1810.03758)]

#### 2.1.1 Correspondence-based Methods

##### a. Match 2D feature points

***2020:***

**[arXiv]** S2DNet: Learning Accurate Correspondences for Sparse-to-Dense Feature Matching, [[paper](https://arxiv.org/pdf/2004.01673.pdf)]

**[arXiv]** SK-Net: Deep Learning on Point Cloud via End-to-end Discovery of Spatial Keypoints, [[paper](https://arxiv.org/pdf/2003.14014.pdf)]

**[arXiv]** LRC-Net: Learning Discriminative Features on Point Clouds by Encoding Local Region Contexts, [[paper](https://arxiv.org/pdf/2003.08240.pdf)]

**[arXiv]** Table-Top Scene Analysis Using Knowledge-Supervised MCMC, [[paper](https://arxiv.org/pdf/2002.08417.pdf)]

**[arXiv]** AprilTags 3D: Dynamic Fiducial Markers for Robust Pose Estimation in Highly Reflective Environments and Indirect Communication in Swarm Robotics, [[paper](https://arxiv.org/pdf/2001.08622.pdf)]

**[AAAI]** LCD: Learned Cross-Domain Descriptors for 2D-3D Matching, [[paper](https://arxiv.org/abs/1911.09326)] [[project](https://hkust-vgd.github.io/lcd/)]

***2019:***

**[ICCV]** GLAMpoints: Greedily Learned Accurate Match points, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Truong_GLAMpoints_Greedily_Learned_Accurate_Match_Points_ICCV_2019_paper.pdf)]

***2016:***

**[ECCV]** LIFT: Learned Invariant Feature Transform, [[paper](https://arxiv.org/pdf/1603.09114.pdf)]

***2012:***

**[3DIMPVT]** 3D Object Detection and Localization using Multimodal Point Pair Features, [[paper](http://far.in.tum.de/pub/drost20123dimpvt/drost20123dimpvt.pdf)]



##### b. Regress 2D projections

***2020:***

**[arXiv]** EPOS: Estimating 6D Pose of Objects with Symmetries, [[paper](https://arxiv.org/pdf/2004.00605.pdf)]

**[arXiv]** Tackling Two Challenges of 6D Object Pose Estimation: Lack of Real Annotated RGB Images and Scalability to Number of Objects, [[paper](https://arxiv.org/pdf/2003.12344.pdf)]

**[arXiv]** Squeezed Deep 6DoF Object Detection using Knowledge Distillation, [[paper](https://arxiv.org/pdf/2003.13586.pdf)]

**[arXiv]** Learning 2D–3D Correspondences To Solve The Blind Perspective-n-Point Problem, [[paper](https://arxiv.org/pdf/2003.06752.pdf)]

**[arXiv]** PnP-Net: A hybrid Perspective-n-Point Network, [[paper](https://arxiv.org/pdf/2003.04626.pdf)]

**[arXiv]** Object 6D Pose Estimation with Non-local Attention, [[paper](https://arxiv.org/pdf/2002.08749.pdf)]

**[arXiv]** 6DoF Object Pose Estimation via Differentiable Proxy Voting Loss, [[paper](https://arxiv.org/pdf/2002.03923.pdf)]

***2019:***

**[arXiv]** DPOD: 6D Pose Object Detector and Refiner, [[paper](https://arxiv.org/pdf/1902.11020.pdf)]

**[CVPR]** Segmentation-driven 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1812.02541)]

**[arXiv]** Single-Stage 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1911.08324)]

**[arXiv]** W-PoseNet: Dense Correspondence Regularized Pixel Pair Pose Regression, [[paper](https://arxiv.org/pdf/1912.11888.pdf)]

**[arXiv]** KeyPose: Multi-view 3D Labeling and Keypoint Estimation for Transparent Objects, [[paper](https://arxiv.org/abs/1912.02805)]

***2018:***

**[CVPR]** Real-time seamless single shot 6d object pose prediction, [[paper](https://arxiv.org/abs/1711.08848)] [[code](https://github.com/Microsoft/singleshotpose)]

**[arXiv]** Estimating 6D Pose From Localizing Designated Surface Keypoints, [[paper](https://arxiv.org/abs/1812.01387)]

***2017:***

**[ICCV]** BB8: a scalable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth, [[paper](https://arxiv.org/abs/1703.10896)]

**[ICCV]** SSD-6D: Making rgb-based 3d detection and 6d pose estimation great again, [[paper](https://arxiv.org/abs/1711.10006)] [[code](https://github.com/wadimkehl/ssd-6d)]

**[ICRA]** 6-DoF Object Pose from Semantic Keypoints, [[paper](https://arxiv.org/abs/1703.04670)]



#### 2.1.2 Template-based Methods

This kind of methods can be regarded as regression-based methods.

***2020:***

**[arXiv]** Neural Object Learning for 6D Pose Estimation Using a Few Cluttered Images, [[paper](https://arxiv.org/pdf/2005.03717.pdf)]

**[arXiv]** How to track your dragon: A Multi-Attentional Framework for real-time RGB-D 6-DOF Object Pose Tracking, [[paper](https://arxiv.org/pdf/2004.10335.pdf)]

**[arXiv]** Self6D: Self-Supervised Monocular 6D Object Pose Estimation, [[paper](https://arxiv.org/pdf/2004.06468.pdf)]

**[arXiv]** A Novel Pose Proposal Network and Refinement Pipeline for Better Object Pose Estimation, [[paper](https://arxiv.org/pdf/2004.05507.pdf)]

**[arXiv]** G2L-Net: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features, [[paper](https://arxiv.org/pdf/2003.11089.pdf)]

**[arXiv]** Neural Mesh Refiner for 6-DoF Pose Estimation, [[paper](https://arxiv.org/pdf/2003.07561.pdf)]

**[arXiv]** MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak Shape Supervision, [[paper](https://arxiv.org/pdf/2003.03522.pdf)]

**[arXiv]** Robust 6D Object Pose Estimation by Learning RGB-D Features, [[paper](https://arxiv.org/pdf/2003.00188.pdf)]

**[arXiv]** HybridPose: 6D Object Pose Estimation under Hybrid Representations, [[paper](https://arxiv.org/pdf/2001.01869.pdf)]

***2019:***

**[arXiv]** P<sup>2</sup>GNet: Pose-Guided Point Cloud Generating Networks for 6-DoF Object Pose Estimation, [[paper](https://arxiv.org/abs/1912.09316)]

**[arXiv]** ConvPoseCNN: Dense Convolutional 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1912.07333)]

**[arXiv]** PointPoseNet: Accurate Object Detection and 6 DoF Pose Estimation in Point Clouds, [[paper](https://arxiv.org/abs/1912.09057)]

**[RSS]** PoseRBPF: A Rao-Blackwellized Particle Filter for 6D Object Pose Tracking, [[paper](https://arxiv.org/abs/1905.09304)]

**[arXiv]** Multi-View Matching Network for 6D Pose Estimation, [[paper](https://arxiv.org/abs/1911.12330)]

**[arXiv]** Fast 3D Pose Refinement with RGB Images, [[paper](https://arxiv.org/pdf/1911.07347.pdf)]

**[arXiv]** MaskedFusion: Mask-based 6D Object Pose Detection, [[paper](https://arxiv.org/abs/1911.07771)]

**[CoRL]** Scene-level Pose Estimation for Multiple Instances of Densely Packed Objects, [[paper](https://arxiv.org/abs/1910.04953)]

**[IROS]** Learning to Estimate Pose and Shape of Hand-Held Objects from RGB Images, [[paper](https://arxiv.org/abs/1903.03340)]

**[IROSW]** Motion-Nets: 6D Tracking of Unknown Objects in Unseen Environments using RGB, [[paper](https://arxiv.org/abs/1910.13942)]

**[ICCV]** DPOD: 6D Pose Object Detector and Refiner, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Zakharov_DPOD_6D_Pose_Object_Detector_and_Refiner_ICCV_2019_paper.html)]

**[ICCV]** CDPN: Coordinates-Based Disentangled Pose Network for Real-Time RGB-Based 6-DoF Object Pose Estimation, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_CDPN_Coordinates-Based_Disentangled_Pose_Network_for_Real-Time_RGB-Based_6-DoF_Object_ICCV_2019_paper.html)]

**[ICCV]** Pix2Pose: Pixel-Wise Coordinate Regression of Objects for 6D Pose Estimation, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Pix2Pose_Pixel-Wise_Coordinate_Regression_of_Objects_for_6D_Pose_Estimation_ICCV_2019_paper.pdf)]

**[ICCV]** Explaining the Ambiguity of Object Detection and 6D Pose From Visual Data, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Manhardt_Explaining_the_Ambiguity_of_Object_Detection_and_6D_Pose_From_ICCV_2019_paper.pdf)]

**[arXiv]** Active 6D Multi-Object Pose Estimation in Cluttered Scenarios with Deep Reinforcement Learning, [[paper](https://arxiv.org/abs/1910.08811)]

**[arXiv]** Accurate 6D Object Pose Estimation by Pose Conditioned Mesh Reconstruction, [[paper](https://arxiv.org/abs/1910.10653)]

**[arXiv]** Learning Object Localization and 6D Pose Estimation from Simulation and Weakly Labeled Real Images, [[paper](https://arxiv.org/abs/1806.06888)]

**[ICHR]** Refining 6D Object Pose Predictions using Abstract Render-and-Compare, [[paper](https://arxiv.org/abs/1910.03412)]

**[arXiv]** Deep-6dpose: recovering 6d object pose from a single rgb image, [[paper](https://arxiv.org/abs/1901.04780)]

**[arXiv]** Real-time Background-aware 3D Textureless Object Pose Estimation, [[paper](https://arxiv.org/abs/1907.09128)]

**[IROS]** SilhoNet: An RGB Method for 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1809.06893)]

***2018:***

**[ECCV]** AAE: Implicit 3D Orientation Learning for 6D Object Detection From RGB Images, [[paper](https://arxiv.org/abs/1902.01275)] [[code](https://github.com/DLR-RM/AugmentedAutoencoder)]

**[ECCV]** DeepIM:Deep Iterative Matching for 6D Pose Estimation, [[paper](https://arxiv.org/abs/1804.00175)] [[code](https://github.com/liyi14/mx-DeepIM)]

**[RSS]** Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes, [[paper](https://arxiv.org/abs/1711.00199)] [[code](https://github.com/yuxng/PoseCNN)]

**[IROS]** Robust 6D Object Pose Estimation in Cluttered Scenes using Semantic Segmentation and Pose Regression Networks, [[paper](https://arxiv.org/abs/1810.03410)]

***2012:***

**[ACCV]** Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.250.6547&rep=rep1&type=pdf)]



#### 2.1.3 Voting-based Methods

***2019:***

**[CVPR]** PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation, [[paper](https://arxiv.org/abs/1812.11788)] [[code](https://github.com/zju3dv/pvnet)]

***2017:***

**[TPAMI]** Robust 3D Object Tracking from Monocular Images Using Stable Parts, [[paper](https://ieeexplore.ieee.org/document/7934426)]

**[Access]** Fast Object Pose Estimation Using Adaptive Threshold for Bin-picking, [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9046779)]

***2014:***

**[ECCV]** Learning 6d object pose estimation using 3d object coordinate, [[paper](http://wwwpub.zih.tu-dresden.de/~cvweb/publications/papers/2014/PoseEstimationECCV2014.pdf)]

**[ECCV]** Latent-class hough forests for 3d object detection and pose estimation, [[paper](https://labicvl.github.io/docs/pubs/Aly_ECCV_2014.pdf)]



***Datasets:***

[LineMOD](http://campar.in.tum.de/Main/StefanHinterstoisser): Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes, ACCV, 2012 [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.250.6547&rep=rep1&type=pdf)] [[database](https://github.com/paroj/linemod_dataset)]

[YCB Datasets](http://www.ycbbenchmarks.com): The YCB Object and Model Set: Towards Common Benchmarks for Manipulation Research, IEEE International Conference on Advanced Robotics (ICAR), 2015 [[paper](http://dx.doi.org/10.1109/ICAR.2015.7251504)]

[T-LESS Datasets](http://cmp.felk.cvut.cz/t-less/): T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects, IEEE Winter Conference on Applications of Computer Vision (WACV), 2017 [[paper](https://arxiv.org/abs/1701.05498)]

HomebrewedDB: RGB-D Dataset for 6D Pose Estimation of 3D Objects, ICCVW, 2019 [[paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/R6D/Kaskman_HomebrewedDB_RGB-D_Dataset_for_6D_Pose_Estimation_of_3D_Objects_ICCVW_2019_paper.pdf)]

YCB-M: A Multi-Camera RGB-D Dataset for Object Recognition and 6DoF Pose Estimation, arXiv, 2020, [[paper](https://arxiv.org/pdf/2004.11657.pdf)] [[database](https://zenodo.org/record/2579173#.XqgpkxMzbX8)]



### 2.2 Point Cloud-based Methods

The partial-view point cloud will be aligned to the complete shape in order to obtain the 6D pose. Generally, coarse registration should be conduct firstly to provide an intial alignment, and dense registration methods like ICP (Iterative Closest Point) will be conducted to obtain the final 6D pose.

#### 2.2.1 Correspondence-based Methods

***2020:***

**[PRL]** Fuzzy Logic and Histogram of Normal Orientation-based 3D Keypoint Detection for Point Clouds, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S016786552030180X)]

**[arXiv]** Latent Fingerprint Registration via Matching Densely Sampled Points, [[paper](https://arxiv.org/pdf/2005.05878.pdf)]

**[arXiv]** RPM-Net: Robust Point Matching using Learned Features, [[paper](https://arxiv.org/pdf/2003.13479.pdf)]

**[arXiv]** End-to-End Learning Local Multi-view Descriptors for 3D Point Clouds, [[paper](https://arxiv.org/pdf/2003.05855.pdf)]

**[arXiv]** D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features, [[paper](https://arxiv.org/pdf/2003.03164.pdf)]

**[arXiv]** Self-supervised Point Set Local Descriptors for Point Cloud Registration, [[paper](https://arxiv.org/pdf/2003.05199.pdf)]

**[arXiv]** StickyPillars: Robust feature matching on point clouds using Graph Neural Networks, [[paper](https://arxiv.org/pdf/2002.03983.pdf)]

***2019:***

**[arXiv]** 3DRegNet: A Deep Neural Network for 3D Point Registration, [[paper](https://arxiv.org/abs/1904.01701)] [[code](https://github.com/goncalo120/3DRegNet)]

**[CVPR]** The Perfect Match: 3D Point Cloud Matching with Smoothed Densities, [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gojcic_The_Perfect_Match_3D_Point_Cloud_Matching_With_Smoothed_Densities_CVPR_2019_paper.pdf)]

**[arXiv]** LCD: Learned Cross-Domain Descriptors for 2D-3D Matching, [[paper](https://arxiv.org/pdf/1911.09326.pdf)]

***2018:***

**[arXiv]** Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation, [[paper](https://arxiv.org/abs/1806.08756)]

**[ECCV]** 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration, [[paper](https://arxiv.org/abs/1807.09413)] [[code](https://github.com/yewzijian/3DFeatNet)]

***2017:***

**[CVPR]** 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions, [[paper](https://arxiv.org/abs/1603.08182)] [[code](https://github.com/andyzeng/3dmatch-toolbox)]

***2016:***

**[arXiv]** Lessons from the Amazon Picking Challenge, [[paper](https://arxiv.org/abs/1601.05484v2)]

**[arXiv]** Team Delft's Robot Winner of the Amazon Picking Challenge 2016, [[paper](https://arxiv.org/abs/1610.05514)]

**[IJCV]** A comprehensive performance evaluation of 3D local feature descriptors, [[paper](https://link.springer.com/article/10.1007/s11263-015-0824-y)]

***2014:***

**[CVIU]** SHOT: Unique signatures of histograms for surface and texture description, [[paper](https://www.sciencedirect.com/science/article/pii/S1077314214000988))]

***2011:***

**[ICCVW]** CAD-model recognition and 6DOF pose estimation using 3D cues, [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6130296)]

***2009:***

**[ICRA]** Fast Point Feature Histograms (FPFH) for 3D registration, [[paper](https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf)]



#### 2.2.2 Template-based Methods

***Survey papers:***

***2020:***

**[arXiv]** When Deep Learning Meets Data Alignment: A Review on Deep Registration Networks (DRNs), [[paper](https://arxiv.org/pdf/2003.03167.pdf)]

**[arXiv]** Least Squares Optimization: from Theory to Practice, [[paper](https://arxiv.org/pdf/2002.11051.pdf)]



***2020:***

**[arXiv]** A Dynamical Perspective on Point Cloud Registration, [[paper](https://arxiv.org/pdf/2005.03190.pdf)]

**[arXiv]** Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences, [[paper](https://arxiv.org/pdf/2005.01014.pdf)]

**[CVPR]** Deep Global Registration, [[paper](https://arxiv.org/pdf/2004.11540.pdf)]

**[arXiv]** DPDist : Comparing Point Clouds Using Deep Point Cloud Distance, [[paper](https://arxiv.org/pdf/2004.11784.pdf)]

**[arXiv]** Single Shot 6D Object Pose Estimation, [[paper](https://arxiv.org/pdf/2004.12729.pdf)]

**[arXiv]** A Benchmark for Point Clouds Registration Algorithms, [[paper](https://arxiv.org/pdf/2003.12841.pdf)] [[code](https://github.com/iralabdisco/point_clouds_registration_benchmark)]

**[arXiv]** PointGMM: a Neural GMM Network for Point Clouds, [[paper](https://arxiv.org/pdf/2003.13326.pdf)]

**[arXiv]** SceneCAD: Predicting Object Alignments and Layouts in RGB-D Scans, [[paper](https://arxiv.org/pdf/2003.12622.pdf)]

**[arXiv]** TEASER: Fast and Certifiable Point Cloud Registration, [[paper](https://arxiv.org/pdf/2001.07715.pdf)] [[code](https://github.com/MIT-SPARK/TEASER-plusplus)]

**[arXiv]** Plane Pair Matching for Efficient 3D View Registration, [[paper](https://arxiv.org/pdf/2001.07058.pdf)]

**[arXiv]** Learning multiview 3D point cloud registration, [[paper](https://arxiv.org/pdf/2001.05119.pdf)]

**[arXiv]** Robust, Occlusion-aware Pose Estimation for Objects Grasped by Adaptive Hands, [[paper](https://arxiv.org/pdf/2003.03518.pdf)]

**[arXiv]** Non-iterative One-step Solution for Point Set Registration Problem on Pose Estimation without Correspondence, [[paper](https://arxiv.org/pdf/2003.00457.pdf)]

**[arXiv]** 6D Object Pose Regression via Supervised Learning on Point Clouds, [[paper](https://arxiv.org/pdf/2001.08942.pdf)]

***2019:***

**[arXiv]** One Framework to Register Them All: PointNet Encoding for Point Cloud Alignment, [[paper](https://arxiv.org/abs/1912.05766)]

**[arXiv]** DeepICP: An End-to-End Deep Neural Network for 3D Point Cloud Registration, [[paper](https://arxiv.org/pdf/1905.04153v2.pdf)]

**[NeurIPS]** PRNet: Self-Supervised Learning for Partial-to-Partial Registration, [[paper](https://arxiv.org/abs/1910.12240)]

**[CVPR]** PointNetLK: Robust & Efficient Point Cloud Registration using PointNet, [[paper](https://arxiv.org/abs/1903.05711)] [[code](https://github.com/hmgoforth/PointNetLK)]

**[ICCV]** End-to-End CAD Model Retrieval and 9DoF Alignment in 3D Scans, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Avetisyan_End-to-End_CAD_Model_Retrieval_and_9DoF_Alignment_in_3D_Scans_ICCV_2019_paper.pdf)]

**[arXiv]** Iterative Matching Point, [[paper](https://arxiv.org/abs/1910.10328)]

**[arXiv]** Deep Closest Point: Learning Representations for Point Cloud Registration, [[paper](https://arxiv.org/abs/1905.03304)] [[code](https://github.com/WangYueFt/dcp)]

**[arXiv]** PCRNet: Point Cloud Registration Network using PointNet Encoding, [[paper](https://arxiv.org/abs/1908.07906)] [[code](https://github.com/vinits5/pcrnet)]

***2016:***

**[TPAMI]** Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration, [[paper](https://arxiv.org/pdf/1605.03344.pdf)] [[code](https://github.com/yangjiaolong/Go-ICP)]

***2014:***

**[SGP]** Super 4PCS Fast Global Pointcloud Registration via Smart Indexing, [[paper](https://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/super4pcs.pdf)] [[code](https://github.com/nmellado/Super4PCS)]



#### 2.2.3 Voting-based Methods



***2020:***

**[arXiv]** MoreFusion: Multi-object Reasoning for 6D Pose Estimation from Volumetric Fusion, [[paper](https://arxiv.org/pdf/2004.04336.pdf)]

**[arXiv]** YOLOff: You Only Learn Offsets for robust 6DoF object pose estimation, [[paper](https://arxiv.org/pdf/2002.00911.pdf)]

**[arXiv]** LRF-Net: Learning Local Reference Frames for 3D Local Shape Description and Matching, [[paper](https://arxiv.org/pdf/2001.07832.pdf)]

***2019:***

**[arXiv]** PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation, [[paper](https://arxiv.org/abs/1911.04231)]

**[CVPR]** Densefusion: 6d object pose estimation by iterative dense fusion, [[paper](https://arxiv.org/abs/1901.04780)] [[code](https://github.com/j96w/DenseFusion)]



### 2.3 Category-level Methods

#### 2.3.1 Category-level 6D pose estimation

***2020:***

**[arXiv]** CPS: Class-level 6D Pose and Shape Estimation From Monocular Images, [[paper](https://arxiv.org/pdf/2003.05848v1.pdf)]

**[arXiv]** Learning Canonical Shape Space for Category-Level 6D Object Pose and Size Estimation, [[paper](https://arxiv.org/pdf/2001.09322.pdf)]

***2019:***

**[arXiv]** Category-Level Articulated Object Pose Estimation, [[paper](https://arxiv.org/pdf/1912.11913.pdf)]

**[arXiv]** LatentFusion: End-to-End Differentiable Reconstruction and Rendering for Unseen Object Pose Estimation, [[paper](https://arxiv.org/abs/1912.00416)]

**[arXiv]** 6-PACK: Category-level 6D Pose Tracker with Anchor-Based Keypoints, [[paper](https://arxiv.org/abs/1910.10750)] [[code](https://github.com/j96w/6-PACK)]

**[arXiv]** Self-Supervised 3D Keypoint Learning for Ego-motion Estimation, [[paper](https://arxiv.org/abs/1912.03426)]

**[CVPR]** Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation, [[paper](https://arxiv.org/abs/1901.02970)] [[code](https://github.com/hughw19/NOCS_CVPR2019)] 

**[arXiv]** Instance- and Category-level 6D Object Pose Estimation, [[paper](https://arxiv.org/pdf/1903.04229.pdf)]

**[arXiv]** kPAM: KeyPoint Affordances for Category-Level Robotic Manipulation, [[paper](https://arxiv.org/abs/1903.06684)]



#### 2.3.2 3D shape reconstruction from images

***2020:***

**[arXiv]** SurfaceNet+: An End-to-end 3D Neural Network for Very Sparse Multi-view Stereopsis, [[paper](https://arxiv.org/pdf/2005.12690.pdf)]

**[arXiv]** FroDO: From Detections to 3D Objects, [[paper](https://arxiv.org/pdf/2005.05125.pdf)]

**[arXiv]** CoReNet: Coherent 3D scene reconstruction from a single RGB image, [[paper](https://arxiv.org/pdf/2004.12989.pdf)]

**[arXiv]** Reconstruct, Rasterize and Backprop: Dense shape and pose estimation from a single image, [[paper](https://arxiv.org/pdf/2004.12232.pdf)]

**[arXiv]** Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes, [[paper](https://arxiv.org/pdf/2004.10904.pdf)]

**[arXiv]** Few-Shot Single-View 3-D Object Reconstruction with Compositional Priors, [[paper](https://arxiv.org/pdf/2004.06302.pdf)]

**[arXiv]** Neural Object Descriptors for Multi-View Shape Reconstruction, [[paper](https://arxiv.org/pdf/2004.04485.pdf)]

**[arXiv]** Leveraging 2D Data to Learn Textured 3D Mesh Generation, [[paper](https://arxiv.org/pdf/2004.04180.pdf)]

**[arXiv]** Deep 3D Capture: Geometry and Reflectance from Sparse Multi-View Images, [[paper](https://arxiv.org/pdf/2003.12642.pdf)]

**[arXiv]** Self-Supervised 2D Image to 3D Shape Translation with Disentangled Representations, [[paper](https://arxiv.org/pdf/2003.10016.pdf)]

**[arXiv]** Atlas: End-to-End 3D Scene Reconstruction from Posed Images, [[paper](https://arxiv.org/pdf/2003.10432.pdf)]

**[arXiv]** Instant recovery of shape from spectrum via latent space connections, [[paper](https://arxiv.org/pdf/2003.06523.pdf)]

**[arXiv]** Self-supervised Single-view 3D Reconstruction via Semantic Consistency, [[paper](https://arxiv.org/pdf/2003.06473.pdf)]

**[arXiv]** Meta3D: Single-View 3D Object Reconstruction from Shape Priors in Memory, [[paper](https://arxiv.org/pdf/2003.03711.pdf)]

**[arXiv]** STD-Net: Structure-preserving and Topology-adaptive Deformation Network for 3D Reconstruction from a Single Image, [[paper](https://arxiv.org/pdf/2003.03551.pdf)]

**[arXiv]** Inverse Graphics GAN: Learning to Generate 3D Shapes from Unstructured 2D Data, [[paper](https://arxiv.org/pdf/2002.12674.pdf)]

**[arXiv]** Deep NRSfM++: Towards 3D Reconstruction in the Wild, [[paper](https://arxiv.org/pdf/2001.10090.pdf)]

**[arXiv]** Learning to Correct 3D Reconstructions from Multiple Views, [[paper](https://arxiv.org/pdf/2001.08098.pdf)]

***2019:***

**[arXiv]** Boundary Cues for 3D Object Shape Recovery, [[paper](https://arxiv.org/pdf/1912.11566.pdf)]

**[arXiv]** Learning to Generate Dense Point Clouds with Textures on Multiple Categories, [[paper](https://arxiv.org/pdf/1912.10545.pdf)]

**[arXiv]** Front2Back: Single View 3D Shape Reconstruction via Front to Back Prediction, [[paper](https://arxiv.org/pdf/1912.10589.pdf)]

**[arXiv]** Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision, [[paper](https://arxiv.org/abs/1912.07372)]

**[arXiv]** SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization, [[paper](https://arxiv.org/abs/1912.07109)]

**[arXiv]** 3D-GMNet: Learning to Estimate 3D Shape from A Single Image As A Gaussian Mixture, [[paper](https://arxiv.org/abs/1912.04663)]

**[arXiv]** Deep-Learning Assisted High-Resolution Binocular Stereo Depth Reconstruction, [[paper](https://arxiv.org/abs/1912.05012)]



#### 2.3.3 3D shape rendering

***2019:***

**[arXiv]** SynSin: End-to-end View Synthesis from a Single Image, [[paper](https://arxiv.org/abs/1912.08804)] [[project](http://www.robots.ox.ac.uk/~ow/synsin.html)]

**[arXiv]** Neural Point Cloud Rendering via Multi-Plane Projection, [[paper](https://arxiv.org/abs/1912.04645)]

**[arXiv]** Neural Voxel Renderer: Learning an Accurate and Controllable Rendering Tool, [[paper](https://arxiv.org/abs/1912.04591)]



## 3. 2D Planar Grasp

### 3.1 Estimating Grasp Contact Points

***2019:***

**[IROS]** GQ-STN: Optimizing One-Shot Grasp Detection based on Robustness Classifier, [[paper](https://arxiv.org/abs/1903.02489)]

**[ICRA]** Mechanical Search: Multi-Step Retrieval of a Target Object Occluded by Clutter, [[paper](https://arxiv.org/abs/1903.01588)]

**[ICRA]** MetaGrasp: Data Efficient Grasping by Affordance Interpreter Network, [[paper](https://arxiv.org/abs/1902.06554)]

**[IROS]** GlassLoc: Plenoptic Grasp Pose Detection in Transparent Clutter, [[paper](https://arxiv.org/abs/1909.04269)]

**[ICRA]** Multi-View Picking: Next-best-view Reaching for Improved Grasping in Clutter, [[paper](https://arxiv.org/abs/1809.08564)] [[code](https://github.com/dougsm/mvp_grasp)]

***2018:***

**[RSS]** Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach, [[paper](https://arxiv.org/pdf/1804.05172.pdf)]

**[BMVC]** EnsembleNet: Improving Grasp Detection using an Ensemble of Convolutional Neural Networks, [[paper](http://bmvc2018.org/contents/papers/0322.pdf)]

**[ICRA]** Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching, [[paper](https://arxiv.org/abs/1710.01330)] [[code](https://github.com/andyzeng/arc-robot-vision)]

***2017:***

**[RSS]** Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics, [[paper](https://arxiv.org/pdf/1703.09312.pdf)] [[code](https://github.com/BerkeleyAutomation/gqcnn)]

***2014:***

**[ICRA]** Fast graspability evaluation on single depth maps for bin picking with general grippers, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.681.316&rep=rep1&type=pdf)]

***Dataset:***

[Dex-Net](https://berkeleyautomation.github.io/dex-net/#dexnet_2), a synthetic dataset of 6.7 million point clouds, grasps, and robust analytic grasp metrics generated from thousands of 3D models.



### 3.2 Estimating Oriented Rectangles

***2020:***

**[arXiv]** Stereo Vision Based Single-Shot 6D Object Pose Estimation for Bin-Picking by a Robot Manipulator , [[paper](https://arxiv.org/pdf/2005.13759.pdf)]

**[arXiv]** SGDN: Segmentation-Based Grasp Detection Network For Unsymmetrical Three-Finger Gripper, [[paper](https://arxiv.org/pdf/2005.08222.pdf)]

**[arXiv]** Event-based Robotic Grasping Detection with Neuromorphic Vision Sensor and Event-Stream Dataset, [[paper](https://arxiv.org/pdf/2004.13652.pdf)]

**[arXiv]** Online Self-Supervised Learning for Object Picking: Detecting Optimum Grasping Position using a Metric Learning Approach, [[paper](https://arxiv.org/pdf/2003.03717.pdf)]

**[arXiv]** A Multi-task Learning Framework for Grasping-Position Detection and Few-Shot Classification, [[paper](https://arxiv.org/pdf/2003.05624.pdf)]

**[arXiv]** Rigid-Soft Interactive Learning for Robust Grasping*, [[paper](https://arxiv.org/pdf/2003.01584.pdf)]

**[arXiv]** Optimizing Correlated Graspability Score and Grasp Regression for Better Grasp Prediction, [[paper](https://arxiv.org/pdf/2002.00872.pdf)]

**[arXiv]** Semi-supervised Grasp Detection by Representation Learning in a Vector Quantized Latent Space, [[paper](https://arxiv.org/pdf/2001.08477.pdf)]

***2019:***

**[arXiv]** Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network, [[paper](https://arxiv.org/pdf/1909.04810.pdf)]

**[IROS]** Domain Independent Unsupervised Learning to grasp the Novel Objects, [[paper](https://arxiv.org/pdf/2001.05856.pdf)]

**[Sensors]** Vision for Robust Robot Manipulation, [[paper](https://www.mdpi.com/1424-8220/19/7/1648/htm)]

**[arXiv]** Form2Fit: Learning Shape Priors for Generalizable Assembly from Disassembly, [[paper](https://arxiv.org/abs/1910.13675)] [[code](https://form2fit.github.io/)]

**[IROS]** GRIP: Generative Robust Inference and Perception for Semantic Robot Manipulation in Adversarial Environments, [[paper](https://arxiv.org/abs/1903.08352)]

**[arXiv]** Efficient Fully Convolution Neural Network for Generating Pixel Wise Robotic Grasps With High Resolution Images, [[paper](https://arxiv.org/abs/1902.08950)]

**[arXiv]** A Single Multi-Task Deep Neural Network with Post-Processing for Object Detection with Reasoning and Robotic Grasp Detection, [[paper](https://arxiv.org/abs/1909.07050)]

**[IROS]** ROI-based Robotic Grasp Detection for Object Overlapping Scenes, [[paper](https://arxiv.org/abs/1808.10313)]

**[RO-MAN]** Real-time Grasp Pose Estimation for Novel Objects in Densely Cluttered Environment, [[paper](https://arxiv.org/pdf/2001.02076.pdf)]

***2018:***

**[IROS]** Fully convolutional grasp detection network with oriented anchor box, [[paper](https://arxiv.org/pdf/1803.02209.pdf)]

**[arXiv]** Real-world Multi-object, Multi-grasp Detection, [[paper](https://arxiv.org/abs/1802.00520)]

**[arXiv]** Classification based grasp detection using spatial transformer network, [[paper](https://arxiv.org/pdf/1803.01356.pdf)]

**[arXiv]** A Multi-task Convolutional Neural Network for Autonomous Robotic Grasping in Object Stacking Scenes, [[paper](https://arxiv.org/pdf/1809.07081.pdf)]

***2017:***

**[IROS]** Robotic Grasp Detection using Deep Convolutional Neural Networks, [[paper](https://arxiv.org/abs/1611.08036)]

**[ICMITE]** Robust Robot Grasp Detection in Multimodal Fusion, [[paper](https://www.matec-conferences.org/articles/matecconf/pdf/2017/53/matecconf_icmite2017_00060.pdf)]

**[ICRA]** A hybrid deep architecture for robotic grasp detection, [[paper](https://ieeexplore.ieee.org/abstract/document/7989191)]

***2016:***

**[ICRA]** Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours, [[paper](https://arxiv.org/abs/1509.06825)]

**[ICRA]** Object discovery and grasp detection with a shared convolutional neural network, [[paper](https://ieeexplore.ieee.org/abstract/document/7487351)]

***2015:***

**[ICRA]** Real-time grasp detection using convolutional neural networks, [[paper](https://arxiv.org/abs/1412.3128)] [[code](https://github.com/tnikolla/robot-grasp-detection)]

**[IJRR]** Deep Learning for Detecting Robotic Grasps, [[paper](https://arxiv.org/abs/1301.3592)]

***2011:***

**[ICRA]** Efficient grasping from rgbd images: Learning using a new rectangle representation, [[paper](http://pr.cs.cornell.edu/grasping/jiang_rectanglerepresentation_fastgrasping.pdf)]

------

***Datasets:***

[Cornell dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php), the dataset consists of 1035 images of 280 different objects.

[Jacquard Dataset](https://jacquard.liris.cnrs.fr), Jacquard: A Large Scale Dataset for Robotic Grasp Detection” in *IEEE International Conference on Intelligent Robots and Systems*, 2018, [[paper](https://arxiv.org/abs/1803.11469)]



## 4. 6DoF Grasp

**Grasp Representation:**
The grasp is represented as 6DoF pose in 3D domain, and the gripper can grasp the object from various angles. The input to this task is 3D point cloud from RGB-D sensors, and this task contains two stages. In the first stage, the target object should be extracted from the scene. In the second stage, if there exists an existing 3D model, the 6D pose of the object could be computed. If there exists no 3D models, the 6DoF grasp pose will be computed from some other methods.

### 4.1 Methods based on Single-view Point Cloud

In this situation, there exist no 3D models, an the 6-DoF grasps are estimated from available partial data. This can be implemented by directly estimating from partial view point cloud, or indirectly estimating after shape completion.

#### 4.1.1 Methods of Estimating Candidate Grasps

***2020:***

**[arXiv]** Transferable Active Grasping and Real Embodied Dataset, [[paper](https://arxiv.org/pdf/2004.13358.pdf)] [[code](https://github.com/cxy1997/Transferable-Active-Grasping)]

**[arXiv]** Go Fetch: Mobile Manipulation in Unstructured Environments, [[paper](https://arxiv.org/pdf/2004.00899.pdf)]

**[arXiv]** Real-time Fruit Recognition and Grasp Estimation for Autonomous Apple harvesting, [[paper](https://arxiv.org/pdf/2003.13298.pdf)]

**[arXiv]** PointNet++ Grasping: Learning An End-to-end Spatial Grasp Generation Algorithm from Sparse Point Clouds, [[paper](https://arxiv.org/pdf/2003.09644.pdf)][[code](https://github.com/pyni/PointNet2_Grasping_Data_Part)]

**[arXiv]** EGAD! an Evolved Grasping Analysis Dataset for diversity and reproducibility in robotic manipulation, [[paper](https://arxiv.org/pdf/2003.01314.pdf)]

**[ariXiv]** REGNet: REgion-based Grasp Network for Single-shot Grasp Detection in Point Clouds, [[paper](https://arxiv.org/pdf/2002.12647.pdf)]

**[RAL]** GRASPA 1.0: GRASPA is a Robot Arm graSping Performance benchmArk, [[paper](https://arxiv.org/pdf/2002.05017.pdf)] [[code](https://github.com/robotology/GRASPA-benchmark)]

**[arXiv]** GraspNet: A Large-Scale Clustered and Densely Annotated Dataset for Object Grasping, [[paper](https://arxiv.org/pdf/1912.13470.pdf)]

***2019:***

**[ISRR]** A Billion Ways to Grasp: An Evaluation of Grasp Sampling Schemes on a Dense, Physics-based Grasp Data Set, [[paper](https://arxiv.org/abs/1912.05604)] [[project](https://sites.google.com/view/abillionwaystograsp)]

**[arXiv]** 6-DOF Grasping for Target-driven Object Manipulation in Clutter, [[paper](https://arxiv.org/abs/1912.03628)]

**[IROS]** Grasping Unknown Objects Based on Gripper Workspace Spheres, [[paper](http://eprints.lincoln.ac.uk/36370/1/IROS19_1656_MS.pdf)]

**[arXiv]** Learning to Generate 6-DoF Grasp Poses with Reachability Awareness, [[paper](https://arxiv.org/abs/1910.06404)]

**[CoRL]** S4G: Amodal Single-view Single-Shot SE(3) Grasp Detection in Cluttered Scenes, [[paper](https://arxiv.org/abs/1910.14218)]

**[ICCV]** 6-DoF GraspNet: Variational Grasp Generation for Object Manipulation, [[paper](https://arxiv.org/abs/1905.10520)] [[code](https://github.com/NVlabs/6dof-graspnet)]

**[ICRA]** PointNetGPD: Detecting Grasp Configurations from Point Sets, [[paper](https://arxiv.org/abs/1809.06267)] [[code](https://github.com/lianghongzhuo/PointNetGPD)]

**[IJARS]** Fast geometry-based computation of grasping points on three-dimensional point clouds, [[paper](https://journals.sagepub.com/doi/10.1177/1729881419831846)]

***2017:***

**[IJRR]** Grasp Pose Detection in Point Clouds, [[paper](https://arxiv.org/abs/1706.09911)] [[code](https://github.com/atenpas/gpd)]

**[ICINCO]** Using geometry to detect grasping points on 3D unknown point cloud, [[paper](https://rua.ua.es/dspace/bitstream/10045/75568/1/ICINCO_2017_182_CR.pdf)]

***2015:***

**[arXiv]** Using geometry to detect grasps in 3d point clouds, [[paper](https://arxiv.org/pdf/1501.03100.pdf)]

***2010:***

**[RAS]** Learning grasping points with shape context, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0921889009001699)]



#### 4.1.2 Methods of Transferring Grasps

##### a. Grasp transfer

***2020:***

**[arXiv]** DGCM-Net: Dense Geometrical Correspondence Matching Network for Incremental Experience-based Robotic Grasping, [[paper](https://arxiv.org/pdf/2001.05279.pdf)]

***2019:***

**[arXiv]** Using Synthetic Data and Deep Networks to Recognize Primitive Shapes for Object Grasping, [[paper](https://arxiv.org/abs/1909.08508)]

**[ICRA]** Transferring Grasp Configurations using Active Learning and Local Replanning, [[paper](https://arxiv.org/abs/1807.08341)]

***2017:***

**[AIP]** Fast grasping of unknown objects using principal component analysis, [[paper](https://aip.scitation.org/doi/10.1063/1.4991996)]

***2016:***

**[Humanoids]** Part-based grasp planning for familiar objects, [[paper](http://h2t.anthropomatik.kit.edu/pdf/Vahrenkamp2016b.pdf)]

***2015:***

**[RAS]** Category-based task specific grasping, [[paper](https://www.sciencedirect.com/science/article/pii/S0921889015000846?via%3Dihub)]

***2003:***

**[ICRA]** Automatic grasp planning using shape primitives, [[paper](https://ieeexplore.ieee.org/abstract/document/1241860)]



##### b. Non-rigid registration

***2020:***

**[arXiv]** Quasi-Newton Solver for Robust Non-Rigid Registration, [[paper](https://arxiv.org/pdf/2004.04322.pdf)]

**[arXiv]** MINA: Convex Mixed-Integer Programming for Non-Rigid Shape Alignment, [[paper](https://arxiv.org/pdf/2002.12623.pdf)]

***2019:***

**[arXiv]** Non-Rigid Point Set Registration Networks, [[paper](https://arxiv.org/abs/1904.01428)] [[code](https://github.com/Lingjing324/PR-Net)]

***2018:***

**[RAL]** Transferring Category-based Functional Grasping Skills by Latent Space Non-Rigid Registration, [[paper](https://arxiv.org/abs/1809.05390)]

**[RAS]** Learning Postural Synergies for Categorical Grasping through Shape Space Registration, [[paper](https://arxiv.org/abs/1810.07967)]

**[RAS]** Autonomous Dual-Arm Manipulation of Familiar Objects, [[paper](https://arxiv.org/abs/1811.08716)]



##### c. Shape correspondence

***2020:***

**[JMSE]** Geometric Deep Learning for Shape Correspondence in Mass Customization by Three-Dimensional Printing, [[paper](https://users.encs.concordia.ca/~thkwok/publication/JMSE20_GDL.pdf)]

**[arXiv]** Semantic Correspondence via 2D-3D-2D Cycle, [[paper](https://arxiv.org/pdf/2004.09061.pdf)]

**[arXiv]** Self-supervised Feature Learning by Cross-modality and Cross-view Correspondences, [[paper](https://arxiv.org/pdf/2004.05749.pdf)]

**[arXiv]** Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence, [[paper](https://arxiv.org/pdf/2003.14286.pdf)] [[code](https://github.com/LIX-shape-analysis/GeomFmaps)]

**[arXiv]** Efficient and Robust Shape Correspondence via Sparsity-Enforced Quadratic Assignment, [[paper](https://arxiv.org/pdf/2003.08680.pdf)]

**[CVM]** Learning local shape descriptors for computing non-rigid dense correspondence, [[paper](https://link.springer.com/content/pdf/10.1007/s41095-020-0163-y.pdf)]

**[JCDE]** Embedded spectral descriptors: learning the point-wise correspondence metric via Siamese neural networks, [[paper](https://arxiv.org/abs/1710.06368)]

**[arXiv]** SAPIEN: A SimulAted Part-based Interactive ENvironment, [[paper](https://arxiv.org/pdf/2003.08515.pdf)]

**[TVCG]** Voting for Distortion Points in Geometric Processing, [[paper](https://arxiv.org/pdf/1909.13066.pdf)]

**[arXiv]** SketchDesc: Learning Local Sketch Descriptors for Multi-view Correspondence, [[paper](https://arxiv.org/pdf/2001.05744.pdf)]

***2019:***

**[arXiv]** Fine-grained Object Semantic Understanding from Correspondences, [[paper](https://arxiv.org/pdf/1912.12577.pdf)]

**[IROS]** Multi-step Pick-and-Place Tasks Using Object-centric Dense Correspondences, [[paper](https://ieeexplore.ieee.org/document/8968294)] [[code](https://github.com/cychai1995/mcdons)]

**[arXiv]** Unsupervised cycle-consistent deformation for shape matching, [[paper](https://arxiv.org/abs/1907.03165)]

**[arXiv]** ZoomOut: Spectral Upsampling for Efficient Shape Correspondence, [[paper](https://arxiv.org/abs/1904.07865)]

**[C&G]** Partial correspondence of 3D shapes using properties of the nearest-neighbor field, [[paper](http://webee.technion.ac.il/~ayellet/Ps/19-ATZ.pdf)]



### 4.2 Methods based on Complete Shape

#### 4.2.1 Methods of Estimating 6D Object Pose

***2017:***

**[IROS]** SegICP: Integrated Deep Semantic Segmentation and Pose Estimation, [[paper](https://arxiv.org/abs/1703.01661)]

**[ICRA]** Multi-view Self-supervised Deep Learning for 6D Pose Estimation in the Amazon Picking Challenge, [[paper](https://arxiv.org/abs/1609.09475)] [[code](https://github.com/andyzeng/apc-vision-toolbox)]

***2010:***

**[SIMPAR]** OpenGRASP: A Toolkit for Robot Grasping Simulation, [[paper](https://h2t.anthropomatik.kit.edu/pdf/Leon2010_SIMPAR.pdf)]

***2009:***

**[ICAR]** An automatic grasp planning system for service robots, [[paper](https://ieeexplore.ieee.org/abstract/document/5174759)]

***2004:***

**[RAM]** Graspit! A versatile simulator for robotic grasping, [[paper](https://ieeexplore.ieee.org/abstract/document/1371616)] [[code](https://github.com/graspit-simulator/graspit)]



#### 4.2.2 Methods of Shape Completion

##### a. Shape Completion-based Grasp

***2020:***

**[arXiv]** Robotic Grasping through Combined Image-Based Grasp Proposal and 3D Reconstruction, [[paper](https://arxiv.org/pdf/2003.01649.pdf)]

***2019:***

**[arXiv]** ClearGrasp: 3D Shape Estimation of Transparent Objects for Manipulation, [[paper](https://arxiv.org/abs/1910.02550)]

**[arXiv]** kPAM-SC: Generalizable Manipulation Planning using KeyPoint Affordance and Shape Completion, [[paper](https://arxiv.org/abs/1909.06980)] [[code](https://sites.google.com/view/generalizable-manipulation/)]

**[arXiv]** Data-Efficient Learning for Sim-to-Real Robotic Grasping using Deep Point Cloud Prediction Networks, [[paper](https://arxiv.org/abs/1906.08989)]

**[IROS]** Robust Grasp Planning Over Uncertain Shape Completions, [[paper](https://arxiv.org/abs/1903.00645)]

**[arXiv]** Multi-Modal Geometric Learning for Grasping and Manipulation, [[paper](https://arxiv.org/abs/1803.07671)]

***2018:***

**[ICRA]** Learning 6-DOF Grasping Interaction via Deep Geometry-aware 3D Representations, [[paper](https://arxiv.org/abs/1708.07303)]

**[IROS]** 3D Shape Perception from Monocular Vision, Touch, and Shape Priors, [[paper](https://arxiv.org/abs/1808.03247)]

***2017:***

**[IROS]** Shape Completion Enabled Robotic Grasping, [[paper](https://arxiv.org/abs/1609.08546)]



##### b. Shape Completion or Generation

***2020:***

**[arXiv]** Point Cloud Completion by Skip-attention Network with Hierarchical Folding, [[paper](https://arxiv.org/pdf/2005.03871.pdf)]

**[arXiv]** PointTriNet: Learned Triangulation of 3D Point Sets, [[paper](https://arxiv.org/pdf/2005.02138.pdf)]

**[arXiv]** DeepSDF x Sim(3): Extending DeepSDF for automatic 3D shape retrieval and similarity transform estimation, [[paper](https://arxiv.org/pdf/2004.09048.pdf)]

**[arXiv]** Anisotropic Convolutional Networks for 3D Semantic Scene Completion, [[paper](https://arxiv.org/pdf/2004.02122.pdf)]

**[arXiv]** Cascaded Refinement Network for Point Cloud Completio, [[paper](https://arxiv.org/pdf/2004.03327.pdf)]

**[arXiv]** Generative PointNet: Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction and Classification, [[paper](https://arxiv.org/pdf/2004.01301.pdf)]

**[arXiv]** Intrinsic Point Cloud Interpolation via Dual Latent Space Navigation, [[paper](https://arxiv.org/pdf/2004.01661.pdf)]

**[arXiv]** Modeling 3D Shapes by Reinforcement Learning, [[paper](https://arxiv.org/pdf/2003.12397.pdf)]

**[arXiv]** PF-Net: Point Fractal Network for 3D Point Cloud Completion, [[paper](https://arxiv.org/pdf/2003.00410.pdf)]

**[arXiv]** Hypernetwork approach to generating point clouds, [[paper](https://arxiv.org/pdf/2003.00802.pdf)]

**[arXiv]** Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion, [[paper](https://arxiv.org/pdf/2003.01456.pdf)]

**[arXiv]** PolyGen: An Autoregressive Generative Model of 3D Meshes, [[paper](https://arxiv.org/pdf/2002.10880.pdf)]

**[arXiv]** BlockGAN Learning 3D Object-aware Scene Representations from Unlabelled Images, [[paper](https://arxiv.org/pdf/2002.08988.pdf)]

**[arXiv]** Implicit Geometric Regularization for Learning Shapes, [[paper](https://arxiv.org/pdf/2002.10099.pdf)]

**[arXiv]** The Whole Is Greater Than the Sum of Its Nonrigid Parts, [[paper](https://arxiv.org/pdf/2001.09650.pdf)]

**[arXiv]** PT2PC: Learning to Generate 3D Point Cloud Shapes from Part Tree Conditions, [[paper](https://arxiv.org/pdf/2003.08624.pdf)]

**[arXiv]** Multimodal Shape Completion via Conditional Generative Adversarial Networks, [[paper](https://arxiv.org/pdf/2003.07717.pdf)]

**[arXiv]** Symmetry Detection of Occluded Point Cloud Using Deep Learning, [[paper](https://arxiv.org/pdf/2003.06520.pdf)]

***2019:***

**[arXiv]** Inferring Occluded Geometry Improves Performance when Retrieving an Object from Dense Clutter, [[paper](https://arxiv.org/abs/1907.08770)]

***2018:***

**[3DORW]** Completion of Cultural Heritage Objects with Rotational Symmetry, [[paper](https://diglib.eg.org/bitstream/handle/10.2312/3dor20181057/087-093.pdf?sequence=1&isAllowed=y)]



##### c. Depth Completion and Estimation

***2020:***

**[CVPR]** Bi3D: Stereo Depth Estimation via Binary Classifications, [[paper](https://arxiv.org/pdf/2005.07274.pdf)]

**[CVPR]** Focus on defocus: bridging the synthetic to real domain gap for depth estimation, [[paper](https://arxiv.org/pdf/2005.09623.pdf)]

**[arXiv]** Decoder Modulation for Indoor Depth Completion, [[paper](https://arxiv.org/pdf/2005.08607.pdf)]

**[CVPR]** On the uncertainty of self-supervised monocular depth estimation, [[paper](https://arxiv.org/pdf/2005.06209.pdf)] [[code](https://github.com/mattpoggi/mono-uncertainty)]

**[arXiv]** Consistent Video Depth Estimation, [[paper](https://arxiv.org/pdf/2004.15021.pdf)]

**[arXiv]** Self-Supervised Attention Learning for Depth and Ego-motion Estimation, [[paper](https://arxiv.org/pdf/2004.13077.pdf)]

**[arXiv]** Pseudo RGB-D for Self-Improving Monocular SLAM and Depth Prediction, [[paper](https://arxiv.org/pdf/2004.10681.pdf)]

**[arXiv]** DepthNet Nano: A Highly Compact Self-Normalizing Neural Network for Monocular Depth Estimation, [[paper](https://arxiv.org/pdf/2004.08008.pdf)]

**[arXiv]** RealMonoDepth: Self-Supervised Monocular Depth Estimation for General Scenes, [[paper](https://arxiv.org/pdf/2004.06267.pdf)]

**[arXiv]** Monocular Depth Estimation with Self-supervised Instance Adaptation, [[paper](https://arxiv.org/pdf/2004.05821.pdf)]

**[arXiv]** Guiding Monocular Depth Estimation Using Depth-Attention Volume, [[paper](https://arxiv.org/pdf/2004.02760.pdf)]

**[arXiv]** 3D Photography using Context-aware Layered Depth Inpainting, [[paper](https://arxiv.org/pdf/2004.04727.pdf)]

**[arXiv]** Occlusion-Aware Depth Estimation with Adaptive Normal Constraints, [[paper](https://arxiv.org/pdf/2004.00845.pdf)]

**[arXiv]** The Edge of Depth: Explicit Constraints between Segmentation and Depth, [[paper](https://arxiv.org/pdf/2004.00171.pdf)]

**[arXiv]** Self-supervised Monocular Trained Depth Estimation using Self-attention and Discrete Disparity Volume, [[paper](https://arxiv.org/pdf/2003.13951.pdf)]

**[arXiv]** DeFeat-Net: General Monocular Depth via Simultaneous Unsupervised Representation Learning, [[paper](https://arxiv.org/pdf/2003.13446.pdf)]

**[arXiv]** Adversarial Attacks on Monocular Depth Estimation, [[paper](https://arxiv.org/pdf/2003.10315.pdf)]

**[arXiv]** Monocular Depth Prediction Through Continuous 3D Loss, [[paper](https://arxiv.org/pdf/2003.09763.pdf)]

**[arXiv]** 3dDepthNet: Point Cloud Guided Depth Completion Network for Sparse Depth and Single Color Image, [[paper](https://arxiv.org/pdf/2003.09175.pdf)]

**[arXiv]** Depth Estimation by Learning Triangulation and Densification of Sparse Points for Multi-view Stereo, [[paper](https://arxiv.org/pdf/2003.08933.pdf)]

**[arXiv]** Monocular Depth Estimation Based On Deep Learning: An Overview, [[paper](https://arxiv.org/pdf/2003.06620.pdf)]

**[arXiv]** Scene Completenesss-Aware Lidar Depth Completion for Driving Scenario, [[paper](https://arxiv.org/pdf/2003.06945.pdf)]

**[arXiv]** Fast Depth Estimation for View Synthesis, [[paper](https://arxiv.org/pdf/2003.06637.pdf)]

**[arXiv]** Active Depth Estimation: Stability Analysis and its Applications, [[paper](https://arxiv.org/pdf/2003.07137.pdf)]

**[arXiv]** Uncertainty depth estimation with gated images for 3D reconstruction, [[paper](https://arxiv.org/pdf/2003.05122.pdf)]

**[arXiv]** Unsupervised Learning of Depth, Optical Flow and Pose with Occlusion from 3D Geometry, [[paper](https://arxiv.org/pdf/2003.00766.pdf)]

**[arXiv]** A-TVSNet: Aggregated Two-View Stereo Network for Multi-View Stereo Depth Estimation, [[paper](https://arxiv.org/pdf/2003.00711.pdf)]

**[arXiv]** Predicting Sharp and Accurate Occlusion Boundaries in Monocular Depth Estimation Using Displacement Fields, [[paper](https://arxiv.org/pdf/2002.12730.pdf)]

**[ICLR]** SEMANTICALLY-GUIDED REPRESENTATION LEARNING FOR SELF-SUPERVISED MONOCULAR DEPTH, [[paper](https://arxiv.org/pdf/2002.12319.pdf)]

**[arXiv]** 3D Gated Recurrent Fusion for Semantic Scene Completion, [[paper](https://arxiv.org/pdf/2002.07269.pdf)]

**[arXiv]** Applying Depth-Sensing to Automated Surgical Manipulation with a da Vinci Robot, [[paper](https://arxiv.org/pdf/2002.06302.pdf)]

**[arXiv]** Fast Generation of High Fidelity RGB-D Images by Deep-Learning with Adaptive Convolution, [[paper](https://arxiv.org/pdf/2002.05067.pdf)]

**[arXiv]** DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data, [[paper](https://arxiv.org/pdf/2002.00569.pdf)]

**[arXiv]** Depth Map Estimation of Dynamic Scenes Using Prior Depth Information, [[paper](https://arxiv.org/pdf/2002.00297.pdf)]

**[arXiv]** FIS-Nets: Full-image Supervised Networks for Monocular Depth Estimation, [[paper](https://arxiv.org/pdf/2001.11092.pdf)]

**[ICRA]** Depth Based Semantic Scene Completion with Position Importance Aware Loss, [[paper](https://arxiv.org/pdf/2001.10709.pdf)]

**[arXiv]** ResDepth: Learned Residual Stereo Reconstruction, [[paper](https://arxiv.org/pdf/2001.08026.pdf)]

**[arXiv]** Single Image Depth Estimation Trained via Depth from Defocus Cues, [[paper](https://arxiv.org/pdf/2001.05036.pdf)]

**[arXiv]** RoutedFusion: Learning Real-time Depth Map Fusion, [[paper](https://arxiv.org/pdf/2001.04388.pdf)]

**[arXiv]** Don't Forget The Past: Recurrent Depth Estimation from Monocular Video, [[paper](https://arxiv.org/pdf/2001.02613.pdf)]

**[AAAI]** Morphing and Sampling Network for Dense Point Cloud Completion, [[paper](https://arxiv.org/abs/1912.00280)] [[code](https://github.com/Colin97/MSN-Point-Cloud-Completion)]

**[AAAI]** CSPN++: Learning Context and Resource Aware Convolutional Spatial Propagation Networks for Depth Completion, [[paper](https://arxiv.org/abs/1911.05377)]

***2019:***

**[arXiv]** Normal Assisted Stereo Depth Estimation, [[paper](https://arxiv.org/pdf/1911.10444.pdf)]

**[arXiv]** GEOMETRY-AWARE GENERATION OF ADVERSARIAL AND COOPERATIVE POINT CLOUDS, [[paper](https://arxiv.org/pdf/1912.11171.pdf)]

**[arXiv]** DeepSFM: Structure From Motion Via Deep Bundle Adjustment, [[paper](https://arxiv.org/abs/1912.09697)]

**[CVIU]** On the Benefit of Adversarial Training for Monocular Depth Estimation, [[paper](https://arxiv.org/abs/1910.13340)]

**[ICCV]** Learning Joint 2D-3D Representations for Depth Completion, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Joint_2D-3D_Representations_for_Depth_Completion_ICCV_2019_paper.pdf)]

**[ICCV]** Deep Optics for Monocular Depth Estimation and 3D Object Detection, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chang_Deep_Optics_for_Monocular_Depth_Estimation_and_3D_Object_Detection_ICCV_2019_paper.pdf)]

**[arXiv]** Deep Classification Network for Monocular Depth Estimation, [[paper](https://arxiv.org/abs/1910.10369)]

**[ICCV]** Depth Completion from Sparse LiDAR Data with Depth-Normal Constraints, [[paper](https://arxiv.org/abs/1910.06727)]

**[arXiv]** Image-based 3D Object Reconstruction: State-of-the-Art and Trends in the Deep Learning Era, [[paper](https://arxiv.org/abs/1906.06543)]

**[arXiv]** Real-time Vision-based Depth Reconstruction with NVidia Jetson, [[paper](https://arxiv.org/abs/1907.07210)]

**[IROS]** Self-supervised 3D Shape and Viewpoint Estimation from Single Images for Robotics, [[paper](https://arxiv.org/abs/1910.07948)]

**[arXiv]** Mesh R-CNN, [[paper](https://arxiv.org/abs/1906.02739)]

**[arXiv]** Monocular depth estimation: a survey, [[paper](https://arxiv.org/pdf/1901.09402.pdf)]

***2018:***

**[3DV]** PCN: Point Completion Network, [[paper](https://arxiv.org/abs/1808.00671)] [[code](https://github.com/wentaoyuan/pcn)]

**[NeurIPS]** Learning to Reconstruct Shapes from Unseen Classes, [[paper](http://genre.csail.mit.edu/papers/genre_nips.pdf)] [[code](https://github.com/xiumingzhang/GenRe-ShapeHD)]

**[ECCV]** Learning Shape Priors for Single-View 3D Completion and Reconstruction, [[paper](https://arxiv.org/abs/1809.05068)] [[code](https://github.com/xiumingzhang/GenRe-ShapeHD)]

**[CVPR]** Deep Depth Completion of a Single RGB-D Image, [[paper](https://arxiv.org/abs/1803.09326)] [[code](https://github.com/yindaz/DeepCompletionRelease)]



##### d. Point Cloud Denoising and Samping

***2020:***

**[arXiv]** MOPS-Net: A Matrix Optimization-driven Network for Task-Oriented 3D Point Cloud Downsampling, [[paper](https://arxiv.org/pdf/2005.00383.pdf)]

**[arXiv]** Deep Feature-preserving Normal Estimation for Point Cloud Filtering, [[paper](https://arxiv.org/pdf/2004.11563.pdf)]

**[arXiv]** Self-Supervised Learning for Domain Adaptation on Point-Clouds, [[paper](https://arxiv.org/pdf/2003.12641.pdf)]

**[arXiv]** Non-Local Part-Aware Point Cloud Denoising, [[paper](https://arxiv.org/pdf/2003.06631.pdf)]

**[arXiv]** PUGeo-Net: A Geometry-centric Network for 3D Point Cloud Upsampling, [[paper](https://arxiv.org/pdf/2002.10277.pdf)]

***2019:***

**[arXiv]** CNN-based Lidar Point Cloud De-Noising in Adverse Weather, [[paper](https://arxiv.org/abs/1912.03874)]

**[arXiv]** PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks, [[paper](https://arxiv.org/abs/1912.03264)] [[code](https://github.com/guochengqian/PU-GCN)]

**[ICCV]** PU-GAN: a Point Cloud Upsampling Adversarial Network, [[paper](https://arxiv.org/pdf/1907.10844.pdf)] [[code](https://github.com/liruihui/PU-GAN)]

**[CVPR]** Patch-based Progressive 3D Point Set Upsampling, [[paper](https://arxiv.org/abs/1811.11286)] [[code](https://github.com/yifita/3PU)]

**[arXiv]** SampleNet: Differentiable Point Cloud Sampling, [[paper](https://arxiv.org/abs/1912.03663)] [[code](https://github.com/itailang/SampleNet)]

***2018:***

**[CVPR]** PU-Net: Point Cloud Upsampling Network, [[paper](https://arxiv.org/abs/1801.06761)] [[code](https://github.com/yulequan/PU-Net)]



## 5. Task-oriented Methods

### 5.1 Task-oriented Manipulation

***2020:***

**[arXiv]** Vision-based control of a knuckle boom crane with online cable length estimation, [[paper](https://arxiv.org/pdf/2005.11794.pdf)]

**[arXiv]** A Point Cloud-Based Method for Automatic Groove Detection and Trajectory Generation of Robotic Arc Welding Tasks, [[paper](https://arxiv.org/pdf/2004.12281.pdf)]

**[arXiv]** Neuromorphic Event-Based Slip Detection and Suppression in Robotic Grasping and Manipulation, [[paper](https://arxiv.org/pdf/2004.07386.pdf)]

**[arXiv]** Combinatorial 3D Shape Generation via Sequential Assembly, [[paper](https://arxiv.org/pdf/2004.07414.pdf)]

**[arXiv]** Learning visual policies for building 3D shape categories, [[paper](https://arxiv.org/pdf/2004.07950.pdf)]

**[arXiv]** Where to relocate?: Object rearrangement inside cluttered and confined environments for robotic manipulation, [[paper](https://arxiv.org/pdf/2003.10863.pdf)]

**[arXiv]** Correspondence Networks with Adaptive Neighbourhood Consensus, [[paper](https://arxiv.org/pdf/2003.12059.pdf)]

**[arXiv]** Development of a Robotic System for Automated Decaking of 3D-Printed Parts, [[paper](https://arxiv.org/pdf/2003.05115.pdf)]

**[arXiv]** Team O2AS at the World Robot Summit 2018: An Approach to Robotic Kitting and Assembly Tasks using General Purpose Grippers and Tools, [[paper](https://arxiv.org/pdf/2003.02427.pdf)]

**[arXiv]** Towards Mobile Multi-Task Manipulation in a Confined and Integrated Environment with Irregular Objects, [[paper](https://arxiv.org/pdf/2003.01776.pdf)]

**[arXiv]** Autonomous Industrial Assembly using Force, Torque, and RGB-D sensing, [[paper](https://arxiv.org/pdf/2002.02580.pdf)]

**[RAL]** A Deep Learning Approach to Grasping the Invisible, [[paper](https://arxiv.org/pdf/1909.04840.pdf)] [[code](https://github.com/choicelab/grasping-invisible)]

***2019:***

**[arXiv]** KETO: Learning Keypoint Representations for Tool Manipulation, [[paper](https://arxiv.org/abs/1910.11977)]

**[arXiv]** Learning Task-Oriented Grasping from Human Activity Datasets, [[paper](https://arxiv.org/abs/1910.11669)]



### 5.2 Grasp Affordance

***2020:***

**[arXiv]** Learning to Grasp 3D Objects using Deep Residual U-Nets, [[paper](https://arxiv.org/pdf/2002.03892.pdf)]

***2019:***

**[IROS]** Detecting Robotic Affordances on Novel Objects with Regional Attention and Attributes, [[paper](https://arxiv.org/abs/1909.05770)]

**[IROS]** Learning Grasp Affordance Reasoning through Semantic Relations, [[paper](https://arxiv.org/abs/1906.09836)]

**[arXiv]** Automatic pre-grasps generation for unknown 3D objects, [[paper](https://arxiv.org/abs/1908.00221)]

**[IECON]** A novel object slicing based grasp planner for 3D object grasping using underactuated robot gripper, [[paper](https://arxiv.org/abs/1907.09142)]

***2018:***

**[ICRA]** AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection, [[paper](https://arxiv.org/pdf/1709.07326.pdf)]

**[arXiv]** Workspace Aware Online Grasp Planning, [[paper](https://arxiv.org/abs/1806.11402)]



### 5.3 3D Part Segmentation

***2020:***

**[arXiv]** Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image, [[paper](https://arxiv.org/pdf/2004.01176.pdf)]

**[arXiv]** Learning 3D Part Assembly from a Single Image, [[paper](https://arxiv.org/pdf/2003.09754.pdf)]

**[ICLR]** Learning to Group: A Bottom-Up Framework for 3D Part Discovery in Unseen Categories, [[paper](https://arxiv.org/pdf/2002.06478.pdf)]

***2019:***

**[arXiv]** Skeleton Extraction from 3D Point Clouds by Decomposing the Object into Parts, [[paper](https://arxiv.org/pdf/1912.11932.pdf)]

**[arXiv]** Neural Shape Parsers for Constructive Solid Geometry, [[paper](https://arxiv.org/pdf/1912.11393.pdf)]

**[arXiv]** PQ-NET: A Generative Part Seq2Seq Network for 3D Shapes, [[paper](https://arxiv.org/abs/1911.10949)]

**[CVPR]** PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation, [[paper](https://arxiv.org/abs/1903.00709)] [[code](https://github.com/FoggYu/PartNet)]

**[C&G]** Autoencoder-based part clustering for part-in-whole retrieval of CAD models, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0097849319300391)]

***2016:***

**[SiggraphAsia]** A Scalable Active Framework for Region Annotation in 3D Shape Collections, [[paper](https://cs.stanford.edu/~ericyi/project_page/part_annotation/)]



## 6. Dexterous Grippers

***2020:***

**[arXiv]** Learning Compliance Adaptation in Contact-Rich Manipulation, [[paper](https://arxiv.org/pdf/2005.00227.pdf)]

**[arXiv]** Leveraging Photometric Consistency over Time for Sparsely Supervised Hand-Object Reconstruction, [[paper](https://arxiv.org/pdf/2004.13449.pdf)]

**[arXiv]** HandVoxNet: Deep Voxel-Based Network for 3D Hand Shape and Pose Estimation from a Single Depth Map, [[paper](https://arxiv.org/pdf/2004.01588.pdf)]

**[arXiv]** Functionally Divided Manipulation Synergy for Controlling Multi-fingered Hands, [[paper](https://arxiv.org/pdf/2003.11699.pdf)]

**[arXiv]** The State of Service Robots: Current Bottlenecks in Object Perception and Manipulation, [[paper](https://arxiv.org/pdf/2003.08151.pdf)]

**[arXiv]** Selecting and Designing Grippers for an Assembly Task in a Structured Approach, [[paper](https://arxiv.org/pdf/2003.04087.pdf)]

**[arXiv]** A Mobile Robot Hand-Arm Teleoperation System by Vision and IMU, [[paper](https://arxiv.org/pdf/2003.05212.pdf)]

**[arXiv]** Robust High-Transparency Haptic Exploration for Dexterous Telemanipulation, [[paper](https://arxiv.org/pdf/2003.01463.pdf)]

**[arXiv]** Tactile Dexterity: Manipulation Primitives with Tactile Feedback, [[paper](https://arxiv.org/pdf/2002.03236.pdf)]

**[arXiv]** Deep Differentiable Grasp Planner for High-DOF Grippers, [[paper](https://arxiv.org/pdf/2002.01530.pdf)]

**[arXiv]** Multi-Fingered Grasp Planning via Inference in Deep Neural Networks, [[paper](https://arxiv.org/pdf/2001.09242.pdf)]

**[RAL]** Benchmarking In-Hand Manipulation, [[paper](https://arxiv.org/pdf/2001.03070.pdf)]

***2019:***

**[arXiv]** GraphPoseGAN: 3D Hand Pose Estimation from a Monocular RGB Image via Adversarial Learning on Graphs, [[paper](https://arxiv.org/abs/1912.01875)]

**[arXiv]** HMTNet:3D Hand Pose Estimation from Single Depth Image Based on Hand Morphological Topology, [[paper](https://arxiv.org/abs/1911.04930)]

**[arXiv]** UniGrasp: Learning a Unified Model to Grasp with N-Fingered Robotic Hands, [[paper](https://arxiv.org/abs/1910.10900)]

**[ScienceRobotics]** On the choice of grasp type and location when handing over an object, [[paper](https://robotics.sciencemag.org/content/4/27/eaau9757)]

**[arXiv]** Solving Rubik's Cube with a Robot Hand, [[paper](https://arxiv.org/abs/1910.07113)]

**[IJARS]** Fast geometry-based computation of grasping points on three-dimensional point clouds, [[paper](https://journals.sagepub.com/doi/pdf/10.1177/1729881419831846)] [[code](https://github.com/yayaneath/GeoGrasp)]

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



## 7. Data Generation

### 7.1 Simulation to Reality

***2020:***

**[arXiv]** How to Close Sim-Real Gap? Transfer with Segmentation!, [[paper](https://arxiv.org/pdf/2005.07695.pdf)]

**[arXiv]** A Study on the Challenges of Using Robotics Simulators for Testing, [[paper](https://arxiv.org/pdf/2004.07368.pdf)]

**[arXiv]** Joint Supervised and Self-Supervised Learning for 3D Real-World Challenges, [[paper](https://arxiv.org/pdf/2004.07392.pdf)]

**[arXiv]** RoboTHOR: An Open Simulation-to-Real Embodied AI Platform, [[paper](https://arxiv.org/pdf/2004.06799.pdf)]

**[arXiv]** On the Effectiveness of Virtual Reality-based Training for Robotic Setup, [[paper](https://arxiv.org/pdf/2003.01540.pdf)]

**[arXiv]** LiDARNet: A Boundary-Aware Domain Adaptation Model for Lidar Point Cloud Semantic Segmentation, [[paper](https://arxiv.org/pdf/2003.01174.pdf)]

**[arXiv]** Multi-source Domain Adaptation in the Deep Learning Era: A Systematic Survey, [[paper](https://arxiv.org/pdf/2002.12169.pdf)]

**[arXiv]** Learning Machines from Simulation to Real World, [[paper](https://arxiv.org/pdf/2002.10853.pdf)]

**[arXiv]** Sim2Real2Sim: Bridging the Gap Between Simulation and Real-World in Flexible Object Manipulation, [[paper](https://arxiv.org/pdf/2002.02538.pdf)]

***2019:***

**[IROS]** Learning to Augment Synthetic Images for Sim2Real Policy Transfer, [[paper](https://arxiv.org/pdf/1903.07740.pdf)]

**[arXiv]** Accept Synthetic Objects as Real-End-to-End Training of Attentive Deep Visuomotor Policies for Manipulation in Clutter, [[paper](https://arxiv.org/abs/1909.11128)]

**[RSSW]** Generative grasp synthesis from demonstration using parametric mixtures, [[paper](https://arxiv.org/abs/1906.11548)]

***2018:***

**[RSS]** Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision, [[paper](https://arxiv.org/abs/1806.09266)]

**[CoRL]** Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects, [[paper](https://arxiv.org/abs/1809.10790)]

**[arXiv]** Multi-Task Domain Adaptation for Deep Learning of Instance Grasping from Simulation, [[paper](https://arxiv.org/abs/1710.06422)]

***2017:***

**[arXiv]** Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping, [[paper](https://arxiv.org/abs/1709.07857)]

### 7.2 Self-supervised Methods

***2019:***

**[arXiv]** Self-supervised 6D Object Pose Estimation for Robot Manipulation, [[paper](https://arxiv.org/abs/1909.10159)]

***2018:***

**[RSS]** Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision, [[paper](https://arxiv.org/abs/1806.09266)]



## 8. Multi-source

***2020:***

**[arXiv]** Teaching Cameras to Feel: Estimating Tactile Physical Properties of Surfaces From Images, [[paper](https://arxiv.org/pdf/2004.14487.pdf)]

**[arXiv]** Multimodal Material Classification for Robots using Spectroscopy and High Resolution Texture Imaging, [[paper](https://arxiv.org/pdf/2004.01160.pdf)]

**[arXiv]** Understanding Contexts Inside Robot and Human Manipulation Tasks through a Vision-Language Model and Ontology System in a Video Stream, [[paper](https://arxiv.org/pdf/2003.01163.pdf)]

**[ToR]** A Transfer Learning Approach to Cross-modal Object Recognition: from Visual Observation to Robotic Haptic Exploration, [[paper](https://arxiv.org/pdf/2001.06673.pdf)]

**[arXiv]** Accurate Vision-based Manipulation through Contact Reasoning,  [[paper](https://arxiv.org/abs/1911.03112)]

***2019:***

**[arXiv]** RoboSherlock: Cognition-enabled Robot Perception for Everyday Manipulation Tasks, [[paper](https://arxiv.org/abs/1911.10079)]

**[ICRA]** Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks, [[paper](https://arxiv.org/abs/1907.13098)]

**[CVPR]**  ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging, [[paper](https://arxiv.org/abs/1904.06830)] [[code](https://github.com/samarth-robo/contactdb_utils)]

***2018:***

**[arXiv]** Learning to Grasp without Seeing, [[paper](https://arxiv.org/abs/1805.04201)]



## 9. Motion Planning

### 9.1 Visual servoing

***2020:***

**[arXiv]** Detailed 2D-3D Joint Representation for Human-Object Interaction, [[paper](https://arxiv.org/pdf/2004.08154.pdf)]

**[arXiv]** Neuromorphic Eye-in-Hand Visual Servoing, [[paper](https://arxiv.org/pdf/2004.07398.pdf)]

**[arXiv]** Predicting Target Feature Configuration of Non-stationary Objects for Grasping with Image-Based Visual Servoing, [[paper](https://arxiv.org/pdf/2001.05650.pdf)]

**[AAAI]** That and There: Judging the Intent of Pointing Actions with Robotic Arms, [[paper](https://arxiv.org/abs/1912.06602)]

***2019:***

**[arXiv]** Camera-to-Robot Pose Estimation from a Single Image, [[paper](https://arxiv.org/abs/1911.09231)]

**[ICRA]** Learning Driven Coarse-to-Fine Articulated Robot Tracking, [[paper](http://www.robots.ox.ac.uk/~mobile/drs/Papers/2019ICRA_rauch.pdf)]

**[CVPR]** Craves: controlling robotic arm with a vision-based, economic system, [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zuo_CRAVES_Controlling_Robotic_Arm_With_a_Vision-Based_Economic_System_CVPR_2019_paper.pdf)] [[code](https://github.com/zuoym15/craves.ai)]

***2018:***

**[arXiv]** Point-to-Pose Voting based Hand Pose Estimation using Residual Permutation Equivariant Layer, [[paper](https://arxiv.org/pdf/1812.02050.pdf)]

***2016:***

**[ICRA]** Robot Arm Pose Estimation by Pixel-wise Regression of Joint Angles, [[paper](https://www.is.mpg.de/uploads_file/attachment/attachment/311/ICRA16_felix_small.pdf)]

***2014:***

**[ICRA]** Robot Arm Pose Estimation through Pixel-Wise Part Classification, [[paper](https://www.is.mpg.de/uploads_file/attachment/attachment/176/2014_ICRA_brhs_small.pdf)]



### 9.2 Path Planning

***2020:***

**[arXiv]** Human-Guided Planner for Non-Prehensile Manipulation, [[paper](https://arxiv.org/pdf/2004.00946.pdf))]

**[arXiv]** Latent Space Roadmap for Visual Action Planning of Deformable and Rigid Object Manipulation, [[paper](https://arxiv.org/pdf/2003.08974.pdf)]

**[arXiv]** GOMP: Grasp-Optimized Motion Planning for Bin Picking, [[paper](https://arxiv.org/pdf/2003.02401.pdf)]

**[arXiv]** Describing Physics For Physical Reasoning: Force-based Sequential Manipulation Planning, [[paper](https://arxiv.org/pdf/2002.12780.pdf)]

**[arXiv]** Reaching, Grasping and Re-grasping: Learning Fine Coordinated Motor Skills, [[paper](https://arxiv.org/pdf/2002.04498.pdf)]

***2019:***

**[arXiv]** Manipulation Trajectory Optimization with Online Grasp Synthesis and Selection, [[paper](https://arxiv.org/abs/1911.10280)]

**[arXiv]** Parareal with a Learned Coarse Model for Robotic Manipulation, [[paper](https://arxiv.org/abs/1912.05958)]



## 10. Imitation Learning

***2020:***

**[arXiv]** HOPE-Net: A Graph-based Model for Hand-Object Pose Estimation, [[paper](https://arxiv.org/pdf/2004.00060.pdf)]

**[arXiv]** SQUIRL: Robust and Efficient Learning from Video Demonstration of Long-Horizon Robotic Manipulation Tasks, [[paper](https://arxiv.org/pdf/2003.04956.pdf)]

**[arXiv]** A Geometric Perspective on Visual Imitation Learning, [[paper](https://arxiv.org/pdf/2003.02768.pdf)]

**[arXiv]** Vision-based Robot Manipulation Learning via Human Demonstrations, [[paper](https://arxiv.org/pdf/2003.00385.pdf)]

**[arXiv]** Gaussian-Process-based Robot Learning from Demonstration, [[paper](https://arxiv.org/pdf/2002.09979.pdf)]

***2019:***

**[arXiv]** Grasping in the Wild: Learning 6DoF Closed-Loop Grasping from Low-Cost Demonstrations, [[paper](https://arxiv.org/abs/1912.04344)] [[project](https://graspinwild.cs.columbia.edu/)]

**[arXiv]** Motion Reasoning for Goal-Based Imitation Learning, [[paper](https://arxiv.org/abs/1911.05864)]

**[IROS]** Robot Learning of Shifting Objects for Grasping in Cluttered Environments, [[paper](https://arxiv.org/abs/1907.11035)] [[code](https://github.com/pantor/learning-shifting-for-grasping)]

**[arXiv]** Learning Deep Parameterized Skills from Demonstration for Re-targetable Visuomotor Control, [[paper](https://arxiv.org/abs/1910.10628)]

**[arXiv]** Adversarial Skill Networks: Unsupervised Robot Skill Learning from Video, [[paper](https://arxiv.org/abs/1910.09430)]

**[IROS]** Learning Actions from Human Demonstration Video for Robotic Manipulation, [[paper](https://arxiv.org/abs/1909.04312)]

**[RSSW]** Generative grasp synthesis from demonstration using parametric mixtures, [[paper](https://arxiv.org/abs/1906.11548)]

***2018:***

**[arXiv]** Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation, [[paper](https://arxiv.org/abs/1710.04615)]



## 11. Reinforcement Learning

***2020:***

**[ICLR]** The Ingredients of Real World Robotic Reinforcement Learning, [[paper](https://arxiv.org/pdf/2004.12570.pdf)]

**[arXiv]** Efficient Adaptation for End-to-End Vision-Based Robotic Manipulation, [[paper](https://arxiv.org/pdf/2004.10190.pdf)]

**[arXiv]** Spatial Action Maps for Mobile Manipulation, [[paper](https://arxiv.org/pdf/2004.09141.pdf)]

**[arXiv]** Learning Precise 3D Manipulation from Multiple Uncalibrated Cameras, [[paper](https://arxiv.org/pdf/2002.09107.pdf)]

**[arXiv]** The Surprising Effectiveness of Linear Models for Visual Foresight in Object Pile Manipulation, [[paper](https://arxiv.org/pdf/2002.09093.pdf)]

**[arXiv]** Learning Pregrasp Manipulation of Objects from Ungraspable Poses, [[paper](https://arxiv.org/pdf/2002.06344.pdf)]

**[arXiv]** Deep Reinforcement Learning for Autonomous Driving: A Survey, [[paper](https://arxiv.org/pdf/2002.00444.pdf)]

**[arXiv]** Lyceum: An efficient and scalable ecosystem for robot learning, [[paper](https://arxiv.org/pdf/2001.07343.pdf)]

**[arXiv]** Planning an Efficient and Robust Base Sequence for a Mobile Manipulator Performing Multiple Pick-and-place Tasks, [[paper](https://arxiv.org/pdf/2001.08042.pdf)]

**[arXiv]** Reward Engineering for Object Pick and Place Training, [[paper](https://arxiv.org/pdf/2001.03792.pdf)]

***2019:***

**[arXiv]** Towards Practical Multi-Object Manipulation using Relational Reinforcement Learning, [[paper](https://arxiv.org/abs/1912.11032)] [[project](https://richardrl.github.io/relational-rl/)] [[code](https://github.com/richardrl/rlkit-relational)]

**[ROBIO]** Efficient Robotic Task Generalization Using Deep Model Fusion Reinforcement Learning, [[paper](https://arxiv.org/abs/1912.05205)]

**[arXiv]** Contextual Reinforcement Learning of Visuo-tactile Multi-fingered Grasping Policies, [[paper](https://arxiv.org/abs/1911.09233)]

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

**[Report]** A Unified Framework for Manipulating Objects via Reinforcement Learning, [[paper](https://course.ie.cuhk.edu.hk/~ierg6130/2019/report/team7.pdf)]

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



## 12. Experts

[Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg/)(CMU & FAIR): Robotics, machine learning

[Andreas ten Pas](http://www.ccs.neu.edu/home/atp/)(Northeastern University): Robotic Grasping, Deep Learning, Simulation-based Planning

[Andy Zeng](http://andyzeng.github.io/)(Princeton University & Google Brain Robotics): 3D Deep Learning, Robotic Grasping

[Animesh Garg](https://www.cs.toronto.edu/~garg/)(University of Toronto): Robotics, Reinforcement Learning

[Bugra Tekin](https://btekin.github.io/)(Microsoft MR): Pose Estimation 

[Cewu Lu](http://mvig.sjtu.edu.cn/)(SJTU): Machine Vision

[Charles Ruizhongtai Qi](https://web.stanford.edu/~rqi/)(Waymo(Google)): 3D Deep Learning

[Danfei Xu](https://cs.stanford.edu/~danfei/)(Stanford University): Robotics, Computer Vision

[Deter Fox](https://homes.cs.washington.edu/~fox/)(Nvidia & University of Washington): Robotics, Artificial intelligence, State Estimation

[Fei-Fei Li](https://profiles.stanford.edu/fei-fei-li/?utm_campaign=Artificial%2BIntelligence%2Band%2BDeep%2BLearning%2BWeekly&utm_medium=web&utm_source=Artificial_Intelligence_and_Deep_Learning_Weekly_3)(Stanford University): Computer Vision

[Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/)(ZJU): 3D Vision, SLAM

[Hao Su](http://cseweb.ucsd.edu/~haosu/)(UC San Diego): 3D Deep Learning

[Jeannette Bohg](https://am.is.tuebingen.mpg.de/person/jbohg)(Stanford University): Perception for robotic manipulation and grasping

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

[Matthias Nießner](https://www.niessnerlab.org/members/matthias_niessner/profile.html)(TUM): 3D reconstruction, Semantic 3D Scene Understanding

[Oliver Brock](https://www.robotics.tu-berlin.de/menue/team/oliver_brock)(TU Berlin): Robotic manipulation

[Pascal Fua](https://icwww.epfl.ch/~fua/)(EPFL): Computer Vision

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

[Yasutaka Furukawa](https://www2.cs.sfu.ca/~furukawa/)(SFU): 3D Reconstruction

[Yu Xiang](https://yuxng.github.io/)(Nvidia): Robotics, Computer Vision

[Yue Wang](https://people.csail.mit.edu/yuewang/)(MIT): 3D Deep Learning
