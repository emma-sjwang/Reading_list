# Unsupervised Learning 

## Domain Generalization (DG)
- **[CVPR/2020]** [Multi-Domain Learning for Accurate and Few-Shot Color Constancy](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xiao_Multi-Domain_Learning_for_Accurate_and_Few-Shot_Color_Constancy_CVPR_2020_paper.pdf) : multi-domain, channel re-weighting
- **[arxiv]** [Domain Generalization via Semi-supervised Meta Learning](https://arxiv.org/pdf/2009.12658.pdf): supervised loss, semi-loss, alignment loss.
- **LDDG [NIPS/2020]** [Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization](https://arxiv.org/abs/2009.12829)
- **EISNet [ECCV/2020]** [Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization](https://github.com/EmmaW8/EISNet)
- **DMG [ECCV/2020]** [Learning to Balance Specificity and Invariance for In and Out of Domain Generalization](https://arxiv.org/pdf/2008.12839.pdf): domain specific masks. similar to attention map **DATA**: PACS, DomainNet
- **[ECCV/2020(oral)]** [Self-Challenging Improves Cross-Domain Generalization](https://arxiv.org/abs/2007.02454) Interesting. mute activate node.	
- **[ECCV/2020]** [Learning to Learn with Variational Information Bottleneck for Domain Generalization](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550205.pdf): meta-learning
- **[ECCV/2020]** [Learning to Generate Novel Domains for Domain Generalization](https://arxiv.org/abs/2007.03304)
- **[ECCV/2020]** [Learning to Optimize Domain Specific Normalization for Domain Generalization](https://arxiv.org/abs/1907.04275)
- **[ECCV/2020]** [HGNet: Hybrid Generative Network for Zero-shot Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720052.pdf)

  
### Medical related
- **[IEEE-TMI/2020]** [MS-Net: Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data](https://github.com/liuquande/MS-Net): **Supervised** domain specific BN. **Task**: Prostate Segmentation
- **[Neurocomputing/2019]** [USE-Net: Incorporating Squeeze-and-Excitation blocks into U-Net for prostate zonal segmentation of multi-institutional MRI datasets](https://www.sciencedirect.com/science/article/pii/S0925231219309245): **Supervised**
- **[MIDL-Abstract/2019]** [A Strong Baseline for Domain Adaptation and Generalization in Medical Imaging](https://openreview.net/forum?id=S1gvm2E-t4)
- **[MICCAI/2018]** [Alifelong  learning  approach  to  brain  mr  segmentation  across  scannersand  protocols](https://arxiv.org/pdf/1805.10170.pdf): DSBN. **Brain** structure segmentation in **MR** images

## Domain Adaptation (DA)

ECCV2020 37 DA related papers
- **[ECCV/2020]** [Instance Adaptive Self-Training for Unsupervised Domain Adaptation](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710409.pdf): adaptation in the instance level
- :star::star::star::star::star::star:[ECCV/2020/Oral]** [Domain-invariant Stereo Matching Newtorks](https://arxiv.org/pdf/1911.13287.pdf) Domain normalisation
- **[ECCV/2020]** [Joint Disentangling and Adaptation for Cross-Domain Person Re-Identification](https://arxiv.org/abs/2007.10315) CycleGAN+disentanglement+self-training
- **[ECCV/2020]** [Unsupervised Domain Adaptation with Noise Resistible Mutual-Training for Person Re-identification](https://zhaoj9014.github.io/pub/1391.pdf)
- **[ECCV/2020]** [Unsupervised Domain Adaptation in the Dissimilarity Space for Person Re-identification](https://arxiv.org/abs/2007.13890)
- **[ECCV/2020]** [Label Propagation with Augmented Anchors: A Simple Semi-Supervised Learning baseline for Unsupervised Domain Adaptation](https://arxiv.org/abs/2007.07695): Transfer DA to semi-supervised learning and performance is quite good.
- **[ECCV/2020]** [Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation](https://arxiv.org/abs/2007.09257): disentangle [domain + category] + 2 GAN **[CODE](https://github.com/VisionLearningGroup/Domain2Vec)**
- **[ECCV/2020]** [Unsupervised Domain Attention Adaptation Network for Caricature Attribute Recognition](https://arxiv.org/abs/2007.09344)
- **[ECCV/2020]** [Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation](https://arxiv.org/abs/2007.08801)
- **[ECCV/2020]** [A Balanced and Uncertainty-aware Approach for Partial Domain Adaptation](https://arxiv.org/abs/2003.02541)
- **[ECCV/2020]** [Class-Incremental Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580052.pdf) source free Good
- **[ECCV/2020]** [Spatial Attention Pyramid Network for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2003.12979.pdf)
- **[ECCV/2020]** [Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590579.pdf)
- **[ECCV/2020]** [Curriculum Manager for Source Selection in Multi-Source Domain Adaptation](https://arxiv.org/pdf/2007.01261.pdf)
- **[ECCV/2020]** [Two-phase Pseudo Label Densification for Self-training based Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580528.pdf)
- **[ECCV/2020]** [Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation](https://arxiv.org/abs/2007.09375)
- **[ECCV/2020]** [Curriculum Manager for Source Selection in Multi-Source Domain Adaptation](https://arxiv.org/pdf/2007.01261.pdf): interesting. Simple method with curriculum learning
- **[ECCV/2020]** [Learning to Detect Open Classes for Universal Domain Adaptation]
- **[ECCV/2020]** [Contextual-Relation Consistent Domain Adaptation for Semantic Segmentation](https://arxiv.org/abs/2007.02424): local alignment
- **[ECCV/2020]** [Partially-Shared Variational Auto-encoders for Unsupervised Domain Adaptation with Target Shift](https://arxiv.org/pdf/2001.07895.pdf) : interesting. Select informative instance to update networks. mutual learning with two networks. Like Yixiao Ge.
- **[ECCV/2020]** [Online Meta-Learning for Multi-Source and Semi-Supervised Domain Adaptation](https://arxiv.org/abs/2004.04398)
- **[ECCV/2020]** [Transferring Domain Shift Across Tasks for Zero-shot Domain adaptation]
- **[ECCV/2020]** [YOLO in the Dark - Domain Adaptation Method for Merging Multiple Models]
- **[ECCV/2020]** [Minimum Class Confusion for Versatile Domain Adaptation](https://arxiv.org/abs/1912.03699)
- **[ECCV/2020]** [Learning from Scale-Invariant Examples for Domain Adaptation in Semantic Segmentation](https://arxiv.org/abs/2007.14449)
- **[ECCV/2020]** [Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery]
- **[ECCV/2020]** [Universal Self-Training for Unsupervised Domain Adaptation]
- **[ECCV/2020]** [Domain Adaptation through Task Distillation]
- **[ECCV/2020]** [Multi-Source Open-Set Deep Adversarial Domain Adaptation](https://dipeshtamboli.github.io/blog/2020/Multi-Source-Open-Set-Deep-Adversarial-Domain-Adaptation/)
- **[ECCV/2020]** [Label-Driven Reconstruction for Domain Adaptation in Semantic Segmentation](https://arxiv.org/abs/2003.04614)
- **[ECCV/2020]** [High Resolution Zero-Shot Domain Adaptation of Synthetically Rendered Face Images](https://arxiv.org/abs/2006.15031)
- **[ECCV/2020]** [Unsupervised Monocular Depth Estimation for Night-time Images using Adversarial Domain Feature Adaptation]
- **[ECCV/2020]** [Dual Mixup Regularized Learning for Adversarial Domain Adaptation](https://arxiv.org/abs/2007.03141)
- **[ECCV/2020]** [Skin Segmentation from NIR Images using Unsupervised Domain Adaptation through Generative Latent Search](https://arxiv.org/abs/2006.08696)


### Medical related

- **[MICCAI/2020]** [Self Domain Adapted Network](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_43): DA at test time on each image. OCT layer segmentation & MRI synthesis. using auto encoder between feature blocks.
- **[TMI/2020]** [Anatomy-Regularized Representation Learning for Cross-Modality Medical Image Segmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9201096): An applicant ion of CV method.
- **ISTN [MICCAI/2020]** [Image-level Harmonization of Multi-Site Datausing Image-and-Spatial Transformer Networks](https://arxiv.org/pdf/2006.16741.pdf): w/o comparison with other DA methods. ????
- **[CVPR/2020]** [Unsupervised Instance Segmentation in Microscopy Images via Panoptic DomainAdaptation and Task Re-weighting](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Unsupervised_Instance_Segmentation_in_Microscopy_Images_via_Panoptic_Domain_Adaptation_CVPR_2020_paper.pdf): CycleGAN + 3 kinds of (GRL) adaptation: image, instance, and semantic  **Tasks**: nuclei segmentation of histopathology patch image
- **[ICCV/2019]** [Semantic-Transferable Weakly-Supervised Endoscopic Lesions Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dong_Semantic-Transferable_Weakly-Supervised_Endoscopic_Lesions_Segmentation_ICCV_2019_paper.pdf): Endoscopic Lesions Segmentation. weakly supervised
- **BEAL [MICCAI/2019]** [Boundary and Entropy-Driven Adversarial Learning for Fundus Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_12): boundary and entropy based adversairl learning. on **Fundus image segmentation**
- **[MICCAI/2019]** [Unsupervised Domain Adaptation via Disentangled Representations: Application to Cross-Modality Liver Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_29). **Tasks**: CT-MR. **Liver** segmentation.
- **SIFA [AAAI/2019]** [Synergistic image and feature adaptation:  Towardscross-modality  domain  adaptation  for  medical  image  seg-mentation](https://arxiv.org/abs/1901.08211): CycleGAN variance. shared encoder for segfmenattion and image generation. **Tasks**: CT-MR 
- **[IEEE-TMI/2019]** [Patch-Based Output Space Adversarial Learning for Joint Optic Disc and Cup Segmentation](https://ieeexplore.ieee.org/abstract/document/8643416): output space adversarial learning. **Fundus image segmentation**
- **SeUDA [MICCAI workshop/2018]** [Semantic-Aware Generative Adversarial Netsfor Unsupervised Domain Adaptationin Chest X-ray Segmentation](https://arxiv.org/pdf/1806.00600.pdf):CycleGAN **Tasks**: different chest X-ray
- **[MICCAI/2018]** [Adversarial Domain Adaptation for Classification of Prostate Histopathology Whole-Slide Images](https://arxiv.org/abs/1806.01357):feature-level adverarial learning. **Task**: whole slide Prostate image classification
- **[MICCAI/2018]** [Tumor-Aware, Adversarial DomainAdaptation from CT to MRI for LungCancer Segmentation](https://link.springer.com/content/pdf/10.1007%2F978-3-030-00934-2_86.pdf): CT<-->MRI. tumor segmentation --> semi-supervised. **Lung cancer**
- **TD-GAN [MICCAI/2018]** [Task Driven Generative Modeling for Unsupervised Domain Adaptation: Application to X-ray Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_67):CycleGAN+Segmentation Net **Task**: CT-X-ray
- **SeUDA [MICCAI-Workshop/2018]** [Semantic-aware  gen-erative  adversarial  nets  for  unsupervised  domain  adaptation  in  chestx-ray  segmentation](https://arxiv.org/abs/1806.00600)
- **[MIDL/2018]** [Domain Adaptation for MRI Organ Segmentation using Reverse Classification Accuracy](https://arxiv.org/abs/1806.00363): supervised domain adaptation:  two-center MR database, organ segmentation
- **[arxiv/2018]** [Unsupervised domain adaptation for medical imaging segmentation with self-ensembling](https://arxiv.org/abs/1811.06042v1): multi-center data.  Spinal Cord Gray Matter 
- **[JBHI/2017]** [Epithelium-Stroma Classification via Convolutional Neural Networks and Unsupervised Domain Adaptation in Histopathological Images ](https://ieeexplore.ieee.org/document/7893702/)
- **[IPMI/2016]** [Unsupervised domain adaptation in brain lesionsegmentation with adversarial networks](https://arxiv.org/pdf/1612.08894.pdf): MR-->MR. **Brain**



### Domain randomization - 2D
- **[CVPR/2020]** [Learning Texture Invariant Representationfor Domain Adaptation of Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Learning_Texture_Invariant_Representation_for_Domain_Adaptation_of_Semantic_Segmentation_CVPR_2020_paper.pdf): like domain randomization, first generate styled images then train with usually GAN. **Datasets**: GTA5, SYNTHIA, Cityscapes
- **[ICCV/2019]** []()

### Adversarial-based UDA - 2D Semantic segmentation
- **[ICIP/2020]**[VARIATIONAL AUTOENCODER BASED UNSUPERVISED DOMAIN ADAPTATION FOR SEMANTIC SEGMENTATION])(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9190973&tag=1)
- **[AAAI/2020]** [Content-Consistent Matching for DomainAdaptive Semantic Segmentation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590426.pdf) layout matching
- **[ECCV/2020]** [Self-Supervised CycleGAN for Object-Preserving Image-to-Image Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650494.pdf): patch-based domain ailgnment based on CycleGAN+domain classificaion network
- **[ECCV/2020]** [Spatial Attention Pyramid Network for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2003.12979.pdf): Attention-based network architecture design.
- **[ECCV/2020]** [Every Pixel Matters: Center-aware Feature Alignment for Domain Adaptive Object Detector](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540698.pdf): Global feature GAN+class-specifc GAN
- **[ECCV/2020]** [CSCL: Critical Semantic-Consistent Learning for Unsupervised Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530732.pdf): RL-based GAN **[CODE](https://github.com/chengchunhsu/EveryPixelMatters)**
- :star::star::star::star::star::star: **SIM [CVPR/2020]** [Differential Treatment for Stuff and Things:A Simple Unsupervised Domain Adaptation Method for Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Differential_Treatment_for_Stuff_and_Things_A_Simple_Unsupervised_Domain_CVPR_2020_paper.pdf): pseudo label; Performance is better than oral paper below, solved GAN training decrease. **Datasets**: Sources[GTA5, SYNTHIA],  Targ et[Cityscapes validation set]
- **[CVPR/2020]** [Unsupervised Intra-domain Adaptation for Semantic Segmentationthrough Self-Supervision](https://arxiv.org/pdf/2004.07703v1.pdf): ADVENT variance, split the dataset into easy and hard subsets. Then adversaril learning between them by a discriminator. Improvement is evident compared with ADVENT but similar to MaxSquare. **Datasets**: Sources[GTA5, SYNTHIA, Synscapes],  Targ et[Cityscapes validation set]
- **[ECCV/2020]** [Two-phase Pseudo Label Densification for Self-training based Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580528.pdf): similar to the above one. The performance even higher than above
- **Dec [AAAI/2020]** [Joint Adversarial Learning for Domain Adaptationin Semantic Segmentation](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ZhangY.4858.pdf) **Datasets**:GTA5, Cityscapes, SYNTHIA
- **[AAAI/2020]** [An Adversarial Perturbation Oriented Domain Adaptation Approach forSemantic Segmentation](https://arxiv.org/pdf/1912.08954v1.pdf)
- :star::star::star::star::star::star:**[NIPS/2019]** [Category Anchor-Guided Unsupervised DomainAdaptation for Semantic Segmentation](https://arxiv.org/pdf/1910.13049.pdf)
- **SIBAN [ICCV/2019]** [Significance-aware Information Bottleneck for Domain AdaptiveSemantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Significance-Aware_Information_Bottleneck_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2019_paper.pdf) **Datasets**: GTA5, Synthiam, Cityscapes.
- **SSF-DAN [ICCV/2019]** [SSF-DAN: Separated Semantic Feature based Domain Adaptation Network forSemantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Du_SSF-DAN_Separated_Semantic_Feature_Based_Domain_Adaptation_Network_for_Semantic_ICCV_2019_paper.pdf): Semantic-wise Separable Discriminator. **Datasets**: GTA5, Synthiam, Cityscapes.
- **MaxSquare [ICCV/2019]** [Domain Adaptation for Semantic Segmentation with Maximum Squares Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Domain_Adaptation_for_Semantic_Segmentation_With_Maximum_Squares_Loss_ICCV_2019_paper.pdf): Change loss function and entropy calculation way with square and margin with multi-scale training. **Datasets**: Office-31, Sources[GTA5, SYNTHIA],  Target[Cityscapes validation set]
- **Patch alignment [ICCV/2019]** [Domain Adaptation for Structured Output via DDiscriminative Patch Representations](https://arxiv.org/pdf/1901.05427.pdf): performance is not so good. **Datasets**: Sources[GTA5, SYNTHIA],  Target[Cityscapes validation set]
- **(TGCF-DA [ICCV/2019]** [Self-Ensembling with GAN-based Data Augmentation for Domain Adaptation inSemantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Self-Ensembling_With_GAN-Based_Data_Augmentation_for_Domain_Adaptation_in_Semantic_ICCV_2019_paper.pdf): image transfer + mean teacher **Datasets**: GTA5, SYNTHIA, Cityscapes
- **ADVENT [CVPR/2019]** [ADVENT: Adversarial Entropy Minimization for Domain Adaptationin Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf): Then Adversarial learning based on entropy maps between source and target by a discriminator. 
- **CrDoCo [CVPR/2019]** [CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_CrDoCo_Pixel-Level_Domain_Transfer_With_Cross-Domain_Consistency_CVPR_2019_paper.pdf): Cycle-GAN+2 segmentators. **Datasets**:GTA5, SYNTHIA, CITYSCAPES
- **[CVPR/2019]** [Bidirectional Learning for Domain Adaptation of Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Bidirectional_Learning_for_Domain_Adaptation_of_Semantic_Segmentation_CVPR_2019_paper.pdf): Cycle-GAN+ sgementatio network train simutenously. **Datasets**:GTA5, SYNTHIA, CITYSCAPES
- **[CVPR/2019]** [Not All Areas Are Equal: Transfer Learning for Semantic Segmentation viaHierarchical Region Selection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Not_All_Areas_Are_Equal_Transfer_Learning_for_Semantic_Segmentation_CVPR_2019_paper.pdf) **Datasets**: GTAV + CITYSCAPES→CITYSCAPES, SYNTHIA +CITYSCAPES→CITYSCAPES and GTAV + SYNTHIA+ CITYSCAPES→CITYSCAPES.
- **SPIGAN [ICLR/2019]** [SPIGAN: Privileged Adversarial Learning from Simulation](https://openreview.net/pdf?id=rkxoNnC5FQ) **Datasets**: SYNTHIA-RAND-CITYSCAPES.
- **AdaptSegNet [CVPR/2018]** [Learning to Adapt Structured Output Space for Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tsai_Learning_to_Adapt_CVPR_2018_paper.pdf): multi-level output adversarial learning. **Datasets**: GTA5, SYNTHIA, Cross-City
- **MCD-DA [CVPR/2018]** [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf): Not good on segmentation tasks. **Datasets**: classification and segmentation
- **CyCADA [ICLR/2018]** [CyCADA: Cycle-Consistent Adversarial Domain Adaptation](http://proceedings.mlr.press/v80/hoffman18a/hoffman18a.pdf): Cycle-GAN + output Adversarial learning
- **[ECCV/2018]** [DCAN: Dual Channel-wise Alignment Networksfor Unsupervised Scene Adaptation](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Zuxuan_Wu_DCAN_Dual_Channel-wise_ECCV_2018_paper.pdf): CycleGAN+alignment based on channel mean and std. **Datasets**: Synthia, Gta5, Cityscapes.
- **[Arxiv/2016]** [Fcns in the wild: Pixel-level adversarial and constraint-basedadaptation](https://arxiv.org/abs/1612.02649) **Datasets**: Cityscapes, SYNTHIA, GTA5, BDDS

### pseudo labels-based UDA - 2D Semantic segmentation
- **[CVPR/2019]** [Domain-Specific Batch Normalization for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1906.03950.pdf)
- **[IJCAI/2020]** [Unsupervised Scene Adaptation with Memory Regularizationin vivo](https://arxiv.org/pdf/1912.11164.pdf): auxiliary classifier instead teacher model + pseudo label **Datasets**:GTA5, Cityscapes, SYNTHIA, Oxford RobotCar
- **CBST [ECCV/2018]** [Unsupervised Domain Adaptation for SemanticSegmentation via Class-Balanced Self-Training](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf): pseudo labels+class-balanced self-train **Datasets**:NTHU,Cityscapes, SYNTHIA, GTA5.
- **BDL [CVPR/2019]** [Bidirectional Learning for Domain Adaptation of Semantic Segmentation](https://arxiv.org/pdf/1904.10620.pdf)

### Person Re-identification
- **[ECCV/2020]** [Deep Credible Metric Learning for Unsupervised Domain Adaptation Person Re-identification](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530630.pdf)
- **[ECCV/2020]** [Generalizing Person Re-Identification by Camera-Aware Invariance Learning and Cross-Domain Mixup](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600222.pdf): mixup variances 
- **[ICLR/2020]** [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS): dual mean teacher with contrastive learning.

### Other methods - 2D Semantic segmentation
- **[ICCV/2019]** [Constructing Self-motivated Pyramid Curriculums for Cross-Domain SemanticSegmentation: A Non-Adversarial Approach](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lian_Constructing_Self-Motivated_Pyramid_Curriculums_for_Cross-Domain_Semantic_Segmentation_A_Non-Adversarial_ICCV_2019_paper.pdf) **Datasets**: GTAV, SYNTHIA, Cityscapes.

### Semantic Nighttime Image Segmentation
- **[ICCV/2019]** [Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation forSemantic Nighttime Image Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.pdf) **Datasets**:Dark Zurich

### Domain invariant alignment - Object recognition 
- [**CVPR/2020**] [Structure Preserving Generative Cross-Domain Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xia_Structure_Preserving_Generative_Cross-Domain_Learning_CVPR_2020_paper.pdf): projection gragh matching
- [Submitted to NIPS/2020] [Domain Adaptation without Source Data](https://arxiv.org/pdf/2007.01524.pdf)
- **[ECCV/2020]** [Mind the Discriminability: Asymmetric Adversarial Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690579.pdf): Auto encoder to reconstruct feature for S and T domains.
- **[ECCV/2020]** [Class-Incremental Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580052.pdf): project Source domain to target domain, align them
- **[ICML/2020]** [Implicit Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation](https://arxiv.org/abs/2006.04996). MMD+pseudo-labels. **Datasets**: Office-31, Office-Home, andVisDA2017.
- **[CVPR/2020]** [Universal Source-Free Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kundu_Universal_Source-Free_Domain_Adaptation_CVPR_2020_paper.pdf)
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
- **[ECCV/2020]** [Monocular 3D Object Detection via Feature Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540018.pdf) point cloud DA
- **xMUDA [CVPR/2020]** [xMUDA: Cross-Modal Unsupervised Domain Adaptationfor 3D Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jaritz_xMUDA_Cross-Modal_Unsupervised_Domain_Adaptation_for_3D_Semantic_Segmentation_CVPR_2020_paper.pdf): Fusion 2D and 3D.

### Video domain adaptation
- **[ECCV/2020]** [Shuffle and Attend: Video Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570664.pdf):
- [**ECCV/2020**] [Omni-sourced Webly-supervised Learning for Video Recognition](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600664.pdf): 滑雪视频 MMLAB， 收集数据啥的
- :star::star: :star:[**CVPR/2020**] [Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Action_Segmentation_With_Joint_Self-Supervised_Temporal_Domain_Adaptation_CVPR_2020_paper.pdf): GRL DA + frame permutation classifier
- [**WACV/2020**] [Action Segmentation with Mixed Temporal Domain Adaptation](https://openaccess.thecvf.com/content_WACV_2020/papers/Chen_Action_Segmentation_with_Mixed_Temporal_Domain_Adaptation_WACV_2020_paper.pdf): local GRL DA+ global DA with domain attention
- [**ICCV/2020**] [Temporal Attentive Alignment for Large-Scale Video Domain Adaptation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Temporal_Attentive_Alignment_for_Large-Scale_Video_Domain_Adaptation_ICCV_2019_paper.pdf): domain attention
- [**AAAI/2020**] [Generative Adversarial Networks for Video-to-Video Domain Adaptation](https://arxiv.org/abs/2004.08058): CycleGAN





## GAN Training



- **[NIPS 2020]** [Differentiable Augmentation for Data-Efficient GAN Training](https://github.com/mit-han-lab/data-efficient-gans): Add Data augmentatio between G and D. results seem amazing.
- 





## Representation Learning

- [] [Self-Supervised Ranking for Representation Learning](https://arxiv.org/pdf/2010.07258.pdf): 换了一种loss
- SimCLR **[ICML/2020]** [A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/google-research/simclr)
- SimCLRv2 **[NIPS/2020]** [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://github.com/google-research/simclr)
- MOCO **[]** [Momentum contrast for unsupervised visual representation learning](https://arxiv.org/abs/1911.05722)
- **C2L** **[MICCAI/2020]** [Comparing to Learn: Surpassing ImageNet  Pretraining on Radiographs by Comparing Image Representations](https://arxiv.org/abs/2007.07423) a version of MOCO update, change the way to select positive and negative pairs.