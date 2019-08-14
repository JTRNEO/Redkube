### Resolve conda deps problems

assuming that enfironment will be called `tf` 

```
conda env remove -n tf -y
conda create -n tf -y 
conda activate tf
conda install -c defaults tensorflow-gpu keras-gpu opencv geopandas cython ipython scipy pillow scikit-image
pip install tqdm imgaug
 ```
 
### Install QGIS on ubuntu

1. add `deb https://qgis.org/ubuntugis/ xenial main` to `/etc/apt/sources.list` file
2. sudo apt-get update
3. Resolve gpg key problem if any, using commands from the documetnation: https://www.qgis.org/en/site/forusers/alldownloads.html
4. After first run install "openlayers" plugin


### Changes

1. Change output from shapeFiles to geoJson files. One file for a featureSet and easier to read.
