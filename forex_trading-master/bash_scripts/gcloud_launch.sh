gcloud compute instances create $MACHINE_NAME \
    --machine-type n1-highmem-8 --zone us-west1-b \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --image=driver2vec-v5 --image-project=cs341driver2vec \
    --maintenance-policy TERMINATE --restart-on-failure \
    --metadata startup-script='#!/bin/bash
    echo "Checking for CUDA and installing."
    # Check for CUDA and try to install.
    if ! dpkg-query -W cuda-10-0; then
      curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
      dpkg -i ./cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
      apt-get update
      apt-get install cuda-10-0 -y
    fi
    source .bashrc
    '
