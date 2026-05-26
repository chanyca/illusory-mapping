# illusory-mapping

[![DOI](https://zenodo.org/badge/1122453263.svg)](https://doi.org/10.5281/zenodo.18049128)  
Data and code accompanying the paper "Did you see the sound? A Bayesian Assessment on Crossmodal Perception in Low Vision" by Ailene Chan, N. R. B. Stiles, C. A. Levitan, A. R. Tanguay, and S. Shimojo.
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
    git clone https://github.com/chanyca/illusory-mapping.git
```
2. Navigate to the project directory in terminal:
```
    cd('illusory-mapping')
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
BCI model fitting: 
- Model fitting: `Data/bci_model_fitting.ipynb`

To reproduce each figure:  
- Figure _N_:
  - `Data/figure_{N}.ipynb`
- Supplementary Figure _N_:
  - `Data/figure_s{N}.ipynb`
---
### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
