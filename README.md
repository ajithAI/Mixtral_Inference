# Mixtral_Inference

### 1. Prerequisites : 
- Install Docker & Nvidia Docker. Follow [Link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) </br>
- Make sure all 8xNvidia H100 GPUs are visible. </br>
- Check GPUs status with Command : `nvidia-smi`


### 2. Setup TRT-LLM Docker Container : 

###### Replace this with your Work Space Path. Minimum Disk Space Required : 300GB

```
export HOSTSPACE="/mnt/Scratch_space"  
```
```
sudo docker run --runtime=nvidia --name=TensorRT_LLM_8xGPU_CUDA_12.6.0 --gpus=all --entrypoint /bin/bash \
                --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add=SYS_PTRACE \
                --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH --security-opt seccomp=unconfined -it \
                -v ${HOSTSPACE}:/home/user -w /home/user nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
```

### 3. Install Dependencies ( Inside Docker ) : 

```
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs vim
```
```
pip3 install numpy==1.26.4
```
### 4. Install TRT-LLM : 

**Unzip the "Mixtral_Inference.zip" folder to ```/home/user/Mixtral_Inference```**
```
cd /home/user/Mixtral_Inference
git clone --recursive https://github.com/NVIDIA/TensorRT-LLM.git
pip install typing-extensions
pip install "cython<3"
pip install "pyyaml==5.4.1" --no-build-isolation
pip install -r requirements.txt
pip install --upgrade transformers
chmod 777 *.sh 
bash copy.sh 
```

### 5. Check Installation : 

```
python3 -c "import tensorrt_llm"  # < Prints TRT Version >
```
```
[TensorRT-LLM] TensorRT-LLM version: 0.13.0.dev2024081000
```

### 6. Download Mixtral 8x7B Model from HuggingFace : 

```
apt-get install git-lfs 
git lfs install 
git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 /home/user/Mixtral-8x7B-v0.1
```

### 7. Run Quantization & Create Checkpoint :
```
export DIR=/home/user/Mixtral_Inference && cd $DIR && mkdir $DIR/Checkpoints
```
```
cd TensorRT-LLM/examples
python3 quantization/quantize.py --model_dir /home/user/Mixtral-8x7B-v0.1 --dtype float16 \
      --qformat fp8 --kv_cache_dtype fp8 --calib_size 512 --tp_size 8 \
      --output_dir $DIR/Checkpoints/Mixtral_8x7B_v0.1_Checkpoint_FP8_8xGPU_CUDA_12.6_TRT_LLM_0.13
```

### 8. Create TRT Engine :

```
export DIR=/home/user/Mixtral_Inference && cd $DIR && mkdir $DIR/TRT_Engines
```
```
trtllm-build --checkpoint_dir $DIR/Checkpoints/Mixtral_8x7B_v0.1_Checkpoint_FP8_8xGPU_CUDA_12.6_TRT_LLM_0.13 \
             --output_dir $DIR/TRT_Engines/Mixtral_8x7B_v0.1_TRT_Engine_FP8_8xGPU_MaxBatch_8192_MaxSeqLen_4096_CUDA_12.6_TRT_LLM_0.13_TP_8 \
             --gemm_plugin auto --use_fp8_context_fmha enable --workers 8 --max_batch_size 8192 --max_input_len 2048 --max_seq_len 4096 
```

### 9. To Run Benchmark inside Docker : 

```
USAGE : <RUN_SCRIPT> <BATCH_SIZE> <INPUT_LENGTH> <OUTPUT_LENGTH> <WARMUP_ITER> <BENCHMARK_ITER> <OUTPUT_FILENAME>

./run_mixtral_inference.sh 64 2048 2048 25 75 Mixtral_TRT_Batch_64_Input_2048_Output_2048
./run_mixtral_inference.sh 96 2048 128 50 200 Mixtral_TRT_Batch_96_Input_2048_Output_128
./run_mixtral_inference.sh 1024 128 128 50 150 Mixtral_TRT_Batch_1024_Input_128_Output_128
./run_mixtral_inference.sh 1024 128 2048 5 25 Mixtral_TRT_Batch_1024_Input_128_Output_2048

./run_mixtral_inference.sh 64 2048 1 50 250 Mixtral_TRT_Batch_64_Input_2048_Output_1
./run_mixtral_inference.sh 96 2048 1 50 250 Mixtral_TRT_Batch_96_Input_2048_Output_1
./run_mixtral_inference.sh 1024 128 1 50 250 Mixtral_TRT_Batch_1024_Input_128_Output_1
```

### 10. To Run Benchmark from Docker Outside : 

```
cd ${HOSTSPACE}/Mixtral_Inference
bash ./docker_run_benchmark.sh 
```


### 11. Error Handlings : 

###### For Error : CUDA initialization: Unexpected error from cudaGetDeviceCount()

```
sudo systemctl stop nvidia-fabricmanager
sudo systemctl restart nvidia-fabricmanager
sudo systemctl status nvidia-fabricmanager
```
