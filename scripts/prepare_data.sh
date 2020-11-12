mkdir -p data
cd data
gdown "https://drive.google.com/uc?id=1E_Xo_o7kwHh3t5sjIxwMCvNv2_542eGE"
unzip meva_data.zip
rm meva_data.zip
cd ..
mkdir -p $HOME/.torch/models/
mv data/meva_data/yolov3.weights $HOME/.torch/models/