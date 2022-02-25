sudo bash /var/scripts/docker-login.sh
sudo docker rm libtpu || true
sudo docker create --name libtpu gcr.io/cloud-tpu-v2-images/libtpu:pytorch-1.9 "/bin/bash"
sudo docker cp libtpu:libtpu.so /lib
sudo pip uninstall --yes torch torch_xla torchvision
sudo pip install torch==1.9.0
sudo pip install torchvision==0.10.0
sudo pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.9-cp38-cp38-linux_x86_64.whl
