## Jupyter Notebooks for implemented methods in
# Blind Hyperspectral Unmixing using Autoencoders: A Critical Comparison

The code uses tensorflow 2.x. The requirements needed to run the code is in the file requirements.txt. The autoencoder methods need the datasets to be in Matlab mat files having the following named variables:

| Variable | Content |
| --- | ----------- |
| Y | Array having dimensions B x P containing the spectra |
| GT | Array having dimensions R x B containing the reference endmembers |
|cols | The number of columns in the hyperspectral image (HSI) |
|rows | The number of rows in the HSI |

Here, R is the number of endmembers, B the number of bands, and P the number of pixels. If you use any of the implemented methods, make sure you reference the papers they implement. The codes might be updated for increased clarity over the coming weeks. Included in this repository are the Urban and Samson HSIs. 

## How to cite
**If you use Method_CNNAEU please use**

B. Palsson, M. O. Ulfarsson and J. R. Sveinsson, "Convolutional Autoencoder for Spectral–Spatial Hyperspectral Unmixing," in IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 1, pp. 535-549, Jan. 2021, doi: 10.1109/TGRS.2020.2992743.

@ARTICLE{9096565,  author={Palsson, Burkni and Ulfarsson, Magnus O. and Sveinsson, Johannes R.},  journal={IEEE Transactions on Geoscience and Remote Sensing},   title={Convolutional Autoencoder for Spectral–Spatial Hyperspectral Unmixing},   year={2021},  volume={59},  number={1},  pages={535-549},  doi={10.1109/TGRS.2020.2992743}}

**If you use code for Method_MTLAEU please use**

B. Palsson, J. R. Sveinsson and M. O. Ulfarsson, "Multitask Learning for Spatial-Spectral Hyperspectral Unmixing," IGARSS 2019 - 2019 IEEE International Geoscience and Remote Sensing Symposium, 2019, pp. 564-567, doi: 10.1109/IGARSS.2019.8900229.

@INPROCEEDINGS{8900229,  author={Palsson, Burkni and Sveinsson, Johannes R. and Ulfarsson, Magnus O.},  booktitle={IGARSS 2019 - 2019 IEEE International Geoscience and Remote Sensing Symposium},   title={Multitask Learning for Spatial-Spectral Hyperspectral Unmixing},   year={2019},  volume={},  number={},  pages={564-567},  doi={10.1109/IGARSS.2019.8900229}}

**If you use code for Method_DAEU please use**

B. Palsson, J. Sigurdsson, J. R. Sveinsson and M. O. Ulfarsson, "Hyperspectral Unmixing Using a Neural Network Autoencoder," in IEEE Access, vol. 6, pp. 25646-25656, 2018, doi: 10.1109/ACCESS.2018.2818280.

@ARTICLE{8322133,  author={Palsson, Burkni and Sigurdsson, Jakob and Sveinsson, Johannes R. and Ulfarsson, Magnus O.},  journal={IEEE Access},   title={Hyperspectral Unmixing Using a Neural Network Autoencoder},   year={2018},  volume={6},  number={},  pages={25646-25656},  doi={10.1109/ACCESS.2018.2818280}}
