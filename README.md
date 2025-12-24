# illusory-flash-mapping

[![DOI](https://zenodo.org/badge/888130091.svg)](https://doi.org/10.5281/zenodo.14167119)  
Data and code accompanying the paper "Did you see the sound? A Bayesian Perspective on Crossmodal Perception in Low Vision" by Ailene Chan, N. R. B. Stiles, C. A. Levitan, A. R. Tanguay, and S. Shimojo.
---
### Getting Started
#### Prerequisites
- Psychtoolbox (Download [here](http://psychtoolbox.org/download))
 
#### Tested on:
- MATLAB R2021a
- Psychtoolbox-3.0.17

#### Hardware information
- Monitor: Dell UltraSharp U2720Q, 3840 x 2160, 60 Hz refresh rate
- Speaker: Bose Companion 2 Series III

#### Installation
1. Clone this repository:  
```
    git clone https://github.com/chanyca/illusory-flash-mapping.git
```
2. Navigate to the project directory in terminal:
```
    cd('illusory-flash-mapping')
```
3. Set up environment
```
    conda env create -f environment.yml
    conda activate illusory-mapping
```
---
### Key functions
`runExpt_vf`: Main script to run Visual Flash Detection Task.  
`runExpt_df`: Main script to run Illusory Double Flash Task.  
`runExpt_ad`: Main script to run Beep Detection Task.  

### Data analysis + plotting
BCI model fitting and analysis: 
- Model fitting: `Data/bci_model_fitting.ipynb`
- Analysis: `Data/bci_analysis.ipynb`

To reproduce each figure:  
- Figure 1:
  - `Data/figure_1.ipynb`
- Figure 2:
  - `Data/figure_2.ipynb`
- Figure 3:
  - `Data/figure_3.ipynb`
- Figure 4:
  - `Data/figure_4.ipynb`
- Figure 5:
  - `Data/figure_5.ipynb`
- Figure S1:
  - `Data/figure_s1.ipynb`
- Figure S2:
  - `Data/figure_s2.ipynb`
- Figure S3:
  - `Data/figure_s3.ipynb`
---
### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
