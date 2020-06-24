# Unsupervised Learning 

## Domain Generalization (DG)
### 2020


## Domain Adaptation (DA)

### Medical related

- **[CVPR/2020]** [Unsupervised Instance Segmentation in Microscopy Images via Panoptic DomainAdaptation and Task Re-weighting](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Unsupervised_Instance_Segmentation_in_Microscopy_Images_via_Panoptic_Domain_Adaptation_CVPR_2020_paper.pdf): CycleGAN + 3 kinds of (GRL) adaptation: image, instance, and semantic  **Tasks**: nuclei segmentation of histopathology patch image
- **SIFA [AAAI/2019]** [Synergistic image and feature adaptation:  Towardscross-modality  domain  adaptation  for  medical  image  seg-mentation](https://arxiv.org/abs/1901.08211): CycleGAN variance. shared encoder for segfmenattion and image generation. **Tasks**: CT-MR 
- **SeUDA [MICCAI workshop/2018]** [Semantic-Aware Generative Adversarial Netsfor Unsupervised Domain Adaptationin Chest X-ray Segmentation](https://arxiv.org/pdf/1806.00600.pdf):CycleGAN **Tasks**: different chest X-ray
- **[MICCAI/2018]** [Adversarial Domain Adaptation for Classification of Prostate Histopathology Whole-Slide Images](https://arxiv.org/abs/1806.01357):feature-level adverarial learning. **Task**: whole slide Prostate image classification
- **TD-GAN [MICCAI/2018]** [Task Driven Generative Modeling for Unsupervised Domain Adaptation: Application to X-ray Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_67):CycleGAN+Segmentation Net **Task**: CT-X-ray
- **[JBHI/2017]** [Epithelium-Stroma Classification via Convolutional Neural Networks and Unsupervised Domain Adaptation in Histopathological Images ](https://ieeexplore.ieee.org/document/7893702/)



### Domain randomization - 2D
- **[CVPR/2020]** [Learning Texture Invariant Representationfor Domain Adaptation of Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Learning_Texture_Invariant_Representation_for_Domain_Adaptation_of_Semantic_Segmentation_CVPR_2020_paper.pdf): like domain randomization, first generate styled images then train with usually GAN. **Datasets**: GTA5, SYNTHIA, Cityscapes
- **[ICCV/2019]** []()

### Adversarial-based UDA - 2D Semantic segmentation
- :star::star::star::star::star::star: **SIM [CVPR/2020]** [Differential Treatment for Stuff and Things:A Simple Unsupervised Domain Adaptation Method for Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Differential_Treatment_for_Stuff_and_Things_A_Simple_Unsupervised_Domain_CVPR_2020_paper.pdf): pseudo label; Performance is better than oral paper below, solved GAN training decrease. **Datasets**: Sources[GTA5, SYNTHIA],  Targ et[Cityscapes validation set]
- **[CVPR/2020]** [Unsupervised Intra-domain Adaptation for Semantic Segmentationthrough Self-Supervision](https://arxiv.org/pdf/2004.07703v1.pdf): ADVENT variance, split the dataset into easy and hard subsets. Then adversaril learning between them by a discriminator. Improvement is evident compared with ADVENT but similar to MaxSquare. **Datasets**: Sources[GTA5, SYNTHIA, Synscapes],  Targ et[Cityscapes validation set]
- **Dec [AAAI/2020]** [Joint Adversarial Learning for Domain Adaptationin Semantic Segmentation](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ZhangY.4858.pdf) **Datasets**:GTA5, Cityscapes, SYNTHIA
- **[AAAI/2020]** [An Adversarial Perturbation Oriented Domain Adaptation Approach forSemantic Segmentation](https://arxiv.org/pdf/1912.08954v1.pdf)
- :star::star::star::star::star::star:**[NIPS/2019]** [Category Anchor-Guided Unsupervised DomainAdaptation for Semantic Segmentation](https://arxiv.org/pdf/1910.13049.pdf)
- **SIBAN [ICCV/2019]** [Significance-aware Information Bottleneck for Domain AdaptiveSemantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Significance-Aware_Information_Bottleneck_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2019_paper.pdf) **Datasets**: GTA5, Synthiam, Cityscapes.
- **SSF-DAN [ICCV/2019]** [SSF-DAN: Separated Semantic Feature based Domain Adaptation Network forSemantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Du_SSF-DAN_Separated_Semantic_Feature_Based_Domain_Adaptation_Network_for_Semantic_ICCV_2019_paper.pdf): Semantic-wise Separable Discriminator. **Datasets**: GTA5, Synthiam, Cityscapes.
- **MaxSquare [ICCV/2019]** [Domain Adaptation for Semantic Segmentation with Maximum Squares Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Domain_Adaptation_for_Semantic_Segmentation_With_Maximum_Squares_Loss_ICCV_2019_paper.pdf): Change loss function and entropy calculation way with square and margin with multi-scale training. **Datasets**: Office-31, Sources[GTA5, SYNTHIA],  Target[Cityscapes validation set]
- **Patch alignment [ICCV/2019]** [Domain Adaptation for Structured Output via DDiscriminative Patch Representations](https://arxiv.org/pdf/1901.05427.pdf): performance is not so good. **Datasets**: Sources[GTA5, SYNTHIA],  Target[Cityscapes validation set]
- **(TGCF-DA [ICCV/2019]** [Self-Ensembling with GAN-based Data Augmentation for Domain Adaptation inSemantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Self-Ensembling_With_GAN-Based_Data_Augmentation_for_Domain_Adaptation_in_Semantic_ICCV_2019_paper.pdf): image transfer + mean teacher Datasets: GTA5, SYNTHIA, Cityscapes
- **ADVENT [CVPR/2019]** [ADVENT: Adversarial Entropy Minimization for Domain Adaptationin Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf): Then Adversarial learning based on entropy maps between source and target by a discriminator. 
- **[CVPR/2019]** [Not All Areas Are Equal: Transfer Learning for Semantic Segmentation viaHierarchical Region Selection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Not_All_Areas_Are_Equal_Transfer_Learning_for_Semantic_Segmentation_CVPR_2019_paper.pdf) **Datasets**: GTAV + CITYSCAPES→CITYSCAPES, SYNTHIA +CITYSCAPES→CITYSCAPES and GTAV + SYNTHIA+ CITYSCAPES→CITYSCAPES.
- **SPIGAN [ICLR/2019]** [SPIGAN: Privileged Adversarial Learning from Simulation](https://openreview.net/pdf?id=rkxoNnC5FQ) **Datasets**: SYNTHIA-RAND-CITYSCAPES.
- **AdaptSegNet [CVPR/2018]** [Learning to Adapt Structured Output Space for Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tsai_Learning_to_Adapt_CVPR_2018_paper.pdf): multi-level output adversarial learning. **Datasets**: GTA5, SYNTHIA, Cross-City
- **MCD-DA [CVPR/2018]** [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf): Not good on segmentation tasks. **Datasets**: classification and segmentation
- **CyCADA [ICLR/2018]** [CyCADA: Cycle-Consistent Adversarial Domain Adaptation](http://proceedings.mlr.press/v80/hoffman18a/hoffman18a.pdf): Cycle-GAN + output Adversarial learning
- **[ECCV/2018]** [DCAN: Dual Channel-wise Alignment Networksfor Unsupervised Scene Adaptation](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Zuxuan_Wu_DCAN_Dual_Channel-wise_ECCV_2018_paper.pdf): CycleGAN+alignment based on channel mean and std. **Datasets**: Synthia, Gta5, Cityscapes.
- **[Arxiv/2016]** [Fcns in the wild: Pixel-level adversarial and constraint-basedadaptation](https://arxiv.org/abs/1612.02649) **Datasets**: Cityscapes, SYNTHIA, GTA5, BDDS

### pseudo labels-based UDA - 2D Semantic segmentation
- **[IJCAI/2020]** [Unsupervised Scene Adaptation with Memory Regularizationin vivo](https://arxiv.org/pdf/1912.11164.pdf): auxiliary classifier instead teacher model + pseudo label **Datasets**:GTA5, Cityscapes, SYNTHIA, Oxford RobotCar
- **CBST [ECCV/2018]** [Unsupervised Domain Adaptation for SemanticSegmentation via Class-Balanced Self-Training](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf): pseudo labels+class-balanced self-train **Datasets**:NTHU,Cityscapes, SYNTHIA, GTA5.
- **BDL [CVPR/2019]** [Bidirectional Learning for Domain Adaptation of Semantic Segmentation](https://arxiv.org/pdf/1904.10620.pdf)

### Person Re-identification
- **[ICLR/2020]** [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS): dual mean teacher with contrastive learning.

### Other methods - 2D Semantic segmentation
- **[ICCV/2019]** [Constructing Self-motivated Pyramid Curriculums for Cross-Domain SemanticSegmentation: A Non-Adversarial Approach](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lian_Constructing_Self-Motivated_Pyramid_Curriculums_for_Cross-Domain_Semantic_Segmentation_A_Non-Adversarial_ICCV_2019_paper.pdf) **Datasets**: GTAV, SYNTHIA, Cityscapes.

### Semantic Nighttime Image Segmentation
- **[ICCV/2019]** [Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation forSemantic Nighttime Image Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.pdf) **Datasets**:Dark Zurich

### Domain invariant alignment - Object recognition 
- **[ICML/2020]** [Implicit Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation](https://arxiv.org/abs/2006.04996). MMD+pseudo-labels. **Datasets**: Office-31, Office-Home, andVisDA2017.
- **DADA [AAAI/2020]** [Discriminative Adversarial Domain Adaptation](https://arxiv.org/pdf/1911.12036.pdf): multi-task [category and domainpredictions]. **Datasets**: Office-31, [Syn2Real](https://arxiv.org/abs/1806.09755)
- **[ICML/2019]** [On Learning Invariant Representations for Domain Adaptation](http://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf): Theoretic analysis. **Datasets**: digits
- **BSP [ICML/2019]** [Transferability vs. Discriminability:Batch Spectral Penalization for Adversarial Domain Adaptation](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) **Datasets**: Office-31, Office-Home, VisDA-2017, Digits
- **CADA [CVPR/2019]** [Attending to Discriminative Certainty for Domain Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kurmi_Attending_to_Discriminative_Certainty_for_Domain_Adaptation_CVPR_2019_paper.pdf) **Datasets**: Office-31, Office-Home, ImageCLEF
- **TPN [CVPR/2019]** [Transferrable Prototypical Networks for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1904.11227.pdf): class-level alignment. **Datasets**: digits, VisDA 2017, [GTA5, Synthia, and Cityscapes], VisDA 2018 
- **SWD [CVPR/2019]** [Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1903.04064.pdf): class-level alignment. **Datasets**: digits, VisDA 2017
- **TADA [AAAI/2019]** [Transferable Attention for Domain Adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-attention-aaai19.pdf): Cluster Alignment, similari to Dou. NIPS. **Datasets**: digits, Office-31, ImageCLEF-DA
- **[AAAI/2019]** [Exploiting Local Feature Patterns for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1811.05042.pdf) **Datasets**: Office-31, Office-Home
- **SAFN [ICCV/2019]** [Larger Norm More Transferable: An Adaptive Feature Norm Approach forUnsupervised Domain Adaptation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Larger_Norm_More_Transferable_An_Adaptive_Feature_Norm_Approach_for_ICCV_2019_paper.pdf) **Datasets**: VisDA2017,Office-Home, Office-31, ImageCLEF-DA.
- **CAT [ICCV/2019]** [Cluster Alignment with a Teacher for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Cluster_Alignment_With_a_Teacher_for_Unsupervised_Domain_Adaptation_ICCV_2019_paper.pdf) **Datasets**: Office-31, Office-Home
- **ADR [ICLR/2018]** [Adversarial Dropout Regularization](https://arxiv.org/abs/1711.01575) Tasks: image classification and semantic segmentation. **Datasets**: digits, VisDA 2017, [GTA5, Synthia, and Cityscapes]
- **MSTN [ICML/2018]** [Learning semantic representations for unsupervised domain adaptation](http://proceedings.mlr.press/v80/xie18c/xie18c.pdf): **Datasets**: Office-31, ImageCLEF-DA, digits
- **[CVPR/2018]** [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1712.02560.pdf) **Datasets**: digits, VisDA 2017, [GTA5, Synthia, and Cityscapes]
- **[CVPR/2018]** [Generate To Adapt: Aligning Domains using Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Generate_to_Adapt_CVPR_2018_paper.pdf) **Datasets**: digits, Office-31, CAD synthetic datase, VISDA dataset
- **[ECCV/2018]** [Deep Adversarial Attention Alignment forUnsupervised Domain Adaptation:the Benefit of Target Expectation Maximization](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guoliang_Kang_Deep_Adversarial_Attention_ECCV_2018_paper.pdf) **Datasets**: digits, Office-31
- **iCAN [CVPR/2018]** [Collaborative and Adversarial Network for Unsupervised domain adaptation](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1410.pdf): multi-level GAN. **Datasets**: Office-31, ImageCLEF-DA 
- **SimNet [CVPR/2018]** [Unsupervised Domain Adaptation with Similarity Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pinheiro_Unsupervised_Domain_Adaptation_CVPR_2018_paper.pdf): similarity among C classes learning. **Datasets**: Digits,Office-31, VisDA
- **[NIPS/2018]** [Conditional Adversarial Domain Adaptation](https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf) **Datasets**: Office-31, ImageCLEF-DA, Office-Home, Digits, [VisDA-2017](http://ai.bu.edu/visda-2017/)
- **MADA [AAAI/2018]** [Multi-Adversarial Domain Adaptation](https://arxiv.org/pdf/1809.02176.pdf) **Datasets**: Office-31, ImageCLEF-DA
- **ADDA [CVPR/2017]** [Adversarial Discriminative Domain Adaptation](https://arxiv.org/pdf/1702.05464.pdf) **Tasks**: Digits, NYU depth dataset
- **JAN [ICML/2017]** [Deep Transfer Learning with Joint Adaptation Networks](https://arxiv.org/pdf/1605.06636.pdf): MMD. **Datasets**: Office-31, [ImageCLEF-DA](http://imageclef.org/2014/adaptation)
- **RTN [NIPS/2016]** [Unsupervised Domain Adaptation with Residual Transfer Networks](https://arxiv.org/abs/1602.04433): MMD. **Datasets**: Office-31, Office-Caltech.
- **DANN [JMLR/2016]** [Domain-adversarial training of neural networks](https://arxiv.org/pdf/1505.07818.pdf) **Tasks**: Image classification & Re-Identification
- **DAN [ICML/2015]** [Learning Transferable Features with Deep Adaptation Network](https://arxiv.org/pdf/1502.02791.pdf): MMD. **Datasets**: Office-31, Office-10 + Caltech-10


### Clustering
- **SRDC [CVPR/2020]** [Unsupervised Domain Adaptation via Regularized Conditional Alignment](http://openaccess.thecvf.com/content_CVPR_2020/papers/Tang_Unsupervised_Domain_Adaptation_via_Structurally_Regularized_Deep_Clustering_CVPR_2020_paper.pdf) **Datasets**: Office-31, ImageCLEF-DA, Office-Home
- **CAN [CVPR/2019]** [Contrastive Adaptation Network for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) **Datasets**: Office-31, VisDA-2017
- **DWT-MEC [CVPR/2019]** [Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss](http://openaccess.thecvf.com/content_CVPR_2019/papers/Roy_Unsupervised_Domain_Adaptation_Using_Feature-Whitening_and_Consensus_Loss_CVPR_2019_paper.pdf) **Datasets**: MNIST↔USPS, MNIST↔SVHN,CIFAR-10↔STL,Office-Home
- **SymNets [CVPR/2019]** [Domain-Symmetric Networks for Adversarial Domain Adaptation](https://arxiv.org/abs/1904.04663) **Datasets**: Office-31, ImageCLEF-DA, Office-Home.
- **DIRT-T [ICLR/2018]** [A DIRT-T Approach to Unsupervised Domain Adaptation](https://openreview.net/pdf?id=H1q-TM-AW) **Datasets**: MNIST, MNIST-M,SVHN, SYN DIGITS, SYN SIGNS, GTSRB, CIFAR-10, and STL-10.




### 3D Semantic Segmentation/Point cloud
- **xMUDA [CVPR/2020]** [xMUDA: Cross-Modal Unsupervised Domain Adaptationfor 3D Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jaritz_xMUDA_Cross-Modal_Unsupervised_Domain_Adaptation_for_3D_Semantic_Segmentation_CVPR_2020_paper.pdf):