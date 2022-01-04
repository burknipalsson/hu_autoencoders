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
