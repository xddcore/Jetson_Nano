#！/bin/bash

echo -e "\033[1;31m Install basic env\033[0m"

echo -e "\033[1;32m Changing APT.Sources...\033[0m"

sleep 2



echo "# Update APT-Source for aarch64

# 默认注释了源码仓库，如有需要可自行取消注释

deb https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic main restricted universe multiverse

# deb-src https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic main main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-updates main restricted universe multiverse

# deb-src https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-updates main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-backports main restricted universe multiverse

# deb-src https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-backports main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-security main restricted universe multiverse

# deb-src https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-security main restricted universe multiverse



# 预发布软件源，不建议启用

# deb https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-proposed main restricted universe multiverse

# deb-src https://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-proposed main restricted universe multiverse

" > sources.list



echo -e "\033[1;32m\n Updating System \n...\033[0m"

sleep 2



sudo mv /etc/apt/sources.list /etc/apt/sources.list.bak

sudo mv sources.list /etc/apt/sources.list



sleep 2



sudo apt update

sudo apt update

sudo apt -y uprade

sudo apt install -y python3-pip python3.6-dev libcurl3-gnutls=7.47.0-1ubuntu2.14 zsh curl tmux screen tree

sudo apt -y autoremove



echo -e "\033[1;32m\n Export CUDA PATH \n...\033[0m"

echo -e "

export CUDA_HOME=/usr/local/cuda-10.0

export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

export PATH=/usr/local/cuda-10.0/bin:$PATH

" >> ~/.bashrc



echo -e "\033[1;32m\n SpaceVIM \n...\033[0m"

curl -sLf https://spacevim.org/cn/install.sh | bash





echo -e "\033[1;32m\n oh-my-zsh \n...\033[0m"

curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | bash

chsh -s /usr/bin/zsh



echo -e "\033[1;32m\n install TF-2.0 \n...\033[0m"

wget https://developer.download.nvidia.com/compute/redist/jp/v42/tensorflow-gpu/tensorflow_gpu-2.0.0+nv19.11-cp36-cp36m-linux_aarch64.whl



sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev

sudo pip3 install -U pip testresources setuptools



sudo pip3 install -U tensorflow_gpu-2.0.0+nv19.11-cp36-cp36m-linux_aarch64.whl Processing ./tensorflow_gpu-2.0.0+nv19.11-cp36-cp36m-linux_aarch64.whl





# If you don't want to use TF 2.0

# You can try other

#  sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu