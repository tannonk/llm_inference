
# Commands used to setup working environment with HuggingFace, Accelerate and BitsandBytes for faster LLM inference

mkdir installs

ml multigpu anaconda3

conda create -n llm_hf1 -c conda-forge python=3.9 cudatoolkit-dev=11.6 -y
conda activate llm_hf1

# install transformers from source
git clone https://github.com/huggingface/transformers.git installs/transformers
cd installs/transformers
pip install -e .
cd ../..

# install bitsandbytes from source
git clone https://github.com/TimDettmers/bitsandbytes.git installs/bitsandbytes
cd installs/bitsandbytes
CUDA_VERSION=116 make cuda11x
python setup.py install
cd ../..

pip install -r requirements.txt

# check the install and CUDA dependencies
python -m bitsandbytes
