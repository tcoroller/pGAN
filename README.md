# A Deep Learning Approach to Private Data Sharing of Medical Images Using Conditional GANs

## Publications
* ArXiv: `TBD`

## Project

Investigate application of `GANs` in medical images. Scope of the project include:
1. Generate artificial images of vertebra units (VUs) conditioned on anatomical region.
2. Conduct an extensive evaluation of the dataset behavior and on the trade off between image quality/dataset faithfulness and privacy.

## Related dataset:
* Link to `Zenodo TBD`
An hdf5 database of the synethetic dataset (10000 pairs of images and region, 2.95GB) is shared with the code.<br/>
With some minor tweaking, the synthetic dataset can be used to run training and analysis to validate the code.
(The analysis itself will be far less relevant because comparing privacy on two synthetic dataset is not very useful)

### Code Lifting
Because the original data is not anonimized, it is not shared with the code. The preprocessing is not shared here either to avoid sharing sensitive system information. <br/>
This code cannot be run end to end out of the box.<br/>
Notebooks for analaysis still hold latest state with figures.<br/>

### Scripts, Notebooks and Demos
1. __Training and generating synthetic VUs and corresponding regions__ [src/manuscript/Train](https://github.com/tcoroller/pGAN/tree/master/src/manuscript/Train)
    - `Training`: python code is [src/manuscript/Train/train_region.py](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Train/train_region.py)
    - `Inference`: Generating synthetic samples from [src/manuscript/Train/Generate_image.ipynb.py](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Train/Generate_image.ipynb)
2. __Fidelity - Analysis__ [src/manuscript/Fidelity](https://github.com/tcoroller/pGAN/tree/master/src/manuscript/Fidelity)
    - `Fetching and plotting` real images from different regions, plotting synthetic samples, interpolating between classes: [src/manuscript/Fidelity/1_images_qualitative_inspection.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Fidelity/1_images_qualitative_inspection.ipynb)
3. __Diversity - Analysis__ [src/manuscript/Diversity](https://github.com/tcoroller/pGAN/tree/master/src/manuscript/Diversity)
    - `Preprocessing`, preparing dataset and training UMAP: [1_generate_synth_112_224.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Diversity/1_generate_synth_112_224.ipynb) and [2_train_umap_112_224.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Diversity/2_train_umap_112_224.ipynb)
    - `UMAP` diversity visualization: [3_plot_umap_diversity.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Diversity/3_plot_umap_diversity.ipynb)
    - `Classification` analysis, quantitative diversity evaluation: [4_classifier_analysis.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Diversity/4_classifier_analysis.ipynb)
4. __Privacy - Analysis__ [src/manuscript/Privacy](https://github.com/tcoroller/pGAN/tree/master/src/manuscript/Privacy)
    - `Preprocessing`, computing features and similartiy: [1_prepare_9_64_64_pixel_space.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Privacy/1_prepare_9_64_64_pixel_space.ipynb), [2_UMAP_64_64.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Privacy/2_UMAP_64_64.ipynb) and [3_compute_distances.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Privacy/3_compute_distances.ipynb)
    - `Pairwise and density attack` robustness: [4_plot_pairwise_attacks.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Privacy/4_plot_pairwise_attacks.ipynb) and [5_plot_density_attacks.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Privacy/5_plot_density_attacks.ipynb)
    - `Embedding space` density visualization: [6_density_plot.ipynb](https://github.com/tcoroller/pGAN/blob/master/src/manuscript/Privacy/5_plot_density_attacks.ipynb)

### Structure

```bash
.
├── README.md
├── environment.yml                  # pgan-env
├── synthetic_dataset.h5
└── src
    ├── helper.py                         # utiliy function (current date-time for mlflow/grid for image visualization)
    └── manuscript                        
       ├── Diversity
       │   ├── 1_generate_synth_112_224.ipynb
       │   ├── 2_train_umap_112_224.ipynb
       │   ├── 3_plot_umap_diversity.ipynb
       │   ├── 4_classifier_analysis.ipynb
       │   ├── classifier_logs                 # restricted data (classifier on train might not be private)
       │   │   └── ... 
       │   ├── diversity_saves                 # restricted data (post processed real dataset included)
       │   │   └── ... 
       │   ├── images
       │   │   └── ... 
       │   └── train_classifier.py
       ├── Fidelity
       │   ├── 1_images_qualitative_inspection.ipynb
       │   ├── helpers
       │   │   └── utils.py                    # code for interpolation between regions
       │   └── images
       │       └── ... 
       ├── Privacy
       │   ├── 1_prepare_9_64_64_pixel_space.ipynb
       │   ├── 2_UMAP_64_64.ipynb
       │   ├── 3_compute_distances.ipynb
       │   ├── 4_plot_pairwise_attacks.ipynb
       │   ├── 5_plot_density_attacks.ipynb
       │   ├── 6_density_plot.ipynb
       │   ├── images
       │   │   ├── ...
       │   │   └── supp
       │   │       └── ... 
       │   └── privacy_saves                  # restricted data (post processed real dataset included, UMAP object might
       │       └── ...                        #  not be private)
       └── Train
            ├── batchers.py
            ├── fixed_architecture.py
            ├── Generate_image.ipynb
            ├── training_parser.py
            ├── train_region.py
            ├── restricted                    # restricted data (GAN weights, local machine preprocessing, indexes
            │   └── ...                       # of train/val split)
            └── transforms
                ├── augmentations.py
                └── transforms.py
```
## Team

__Authors:__

- __Hanxi Sun__, _Purdue University, Department of Statistics_
- __Jason Plawinski__, _Novartis_
- __Sajanth Subramaniam__, _Novartis_
- __Amir Jamaludin__, _Oxford Big Data Institute_
- __Timor Kadir__, _Oxford Big Data Institute_
- __Aimee Readie__, _Novartis_
- __Gregory Ligozio__, _Novartis_
- __David Ohlssen__,_Novartis_
- __Mark Baillie__, _Novartis_
- __Thibaud Coroller__, _Novartis_


