# At root folder, run
# source bash_scripts/gcloud_helper.sh

# You must change this to your own directory
SCRIPT_DIR=/mnt/c/Users/jingb/GoogleDrive/CurrentCode/CurrentCode/forex_trading/bash_scripts/

version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ $(python --version 2>&1) =~ 2\.7 ]]
then
    # echo "OK. Python version is \"$version\""
    :
else
    # echo "WARNING! GCLOUD is dumb and only works for Python 2.7"
    :
fi

source $SCRIPT_DIR/user_info_no_commit.sh


# SSH onto Google Cloud instance
# Does the same thing as
#   gcloud compute ssh [INSTANCE_NAME]
gc_ssh() {
    echo "Instance Name => $1"
    echo "Additional Args => $2"
    IP_ADDR=$(gcloud compute instances list --filter="name=$1" --format "get(networkInterfaces[0].accessConfigs[0].natIP)")
    echo "ssh -i $SSH_KEY_LOC $GC_USERNAME@$IP_ADDR $2"
    ssh -i $SSH_KEY_LOC $GC_USERNAME@$IP_ADDR $2
}

# List all running virtual machines
gc_info() {
    gcloud compute instances list
}

# rsync files onto Google Cloud
gc_put() {
    echo "Instance Name => $1"
    echo "Source => $2"
    echo "Target => $3"
    IP_ADDR=$(gcloud compute instances list --filter="name=$1" --format "get(networkInterfaces[0].accessConfigs[0].natIP)")
    rsync --exclude='.git/' -Pavz -e "ssh -i $SSH_KEY_LOC" $2 $GC_USERNAME@$IP_ADDR:$3
}

# rsync files from Google Cloud
gc_get() {
    echo "Instance Name => $1"
    echo "Source => $2"
    echo "Target => $3"
    IP_ADDR=$(gcloud compute instances list --filter="name=$1" --format "get(networkInterfaces[0].accessConfigs[0].natIP)")
    rsync --exclude='.git/' -Pavz -e "ssh -i $SSH_KEY_LOC" $GC_USERNAME@$IP_ADDR:$2 $3
}

gc_jupyter() {
    echo "Instance Name => $1"
    echo "Root Directory => $2"
    echo "Port => $3"
    
    AFTER_SSH="source $GC_HOME/.bashrc; conda activate shared_driver; cd $2; jupyter notebook --no-browser --ip=\* --port $3"
    IP_ADDR=$(gcloud compute instances list --filter="name=$1" --format "get(networkInterfaces[0].accessConfigs[0].natIP)")

    echo "Jupyter notebook using will be available at "
    echo "============================================"
    echo -e "\t$IP_ADDR:$3"
    echo "============================================"

    gc_ssh $1 "-tt bash -l -c \"$AFTER_SSH\" "
}