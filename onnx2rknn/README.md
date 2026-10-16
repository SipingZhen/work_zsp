conda create -n rknn  python=3.8

conda activate rknn

pip install rknn-toolkit2

pip install --upgrade pillow

pip uninstall torch torchvision torchaudio -y

pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

python onnx2rknn.py#运行脚本 这里面要改路径 
