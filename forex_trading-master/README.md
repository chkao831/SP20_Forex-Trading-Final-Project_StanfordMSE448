# Forex Trading for MS&E 448
Code for MS&amp;E 448

# Setting up

Please add a file called `user_info_no_commit.sh` in `bash_scripts` folder.
The following is jingbo's content. Modify as you need (mostly just usuername).

```
# Things that are unique to each user
# SSH_KEY_LOC="/home/jingboyang/.ssh/google_compute_engine"
# SSH_KEY_LOC="/home/jingbo/.ssh/google_compute_engine"

# Path to SSH key location
SSH_KEY_LOC="$HOME/.ssh/google_compute_engine"

# Username on Google Cloud
GC_USERNAME="jingboyang"

# Home directory on Google Cloud
GC_HOME="/home/jingboyang"

# The following are not being used at the moment
# Other things that I don't feel like putting into the repo
GC_DATA="/mnt/shared_data"
GC_JUPYTER="/mnt/shared_data/jupyter_notebooks"
```

Google cloud Python API on your local machine is recommended but not required.

To access VM on GCP, you need to
* Start a VM instance, using the latest of `trading-v[0-9]+` images
* Locally, source `bash_scripts/gcloud_helper.sh`
* Run `gc_ssh [vm name]`

To upload file to GCP VM
* Locally, source `bash_scripts/gcloud_helper.sh`
* Run `gc_ut [vm name] [local path] [remote path]`

See all uses in [gcloud_helper file](bash_scripts/gcloud_helper.sh)


# Data Exploration

See uses of `DataAPI` in [this notebook](notebooks/data_plotting.ipynb)


# Using Deep

For Jingbo and Jon only.

On deep

```
jupyter notebook --no-browser --port=4481
```

On SC for port forwarding


```
ssh -N -L 4481:localhost:4481 jingbo@deep[#]
```

On local (Jingbo only)

```
ssh -N -L 4481:localhost:4481 jingbo@stanford_psc
```

# Using GCP

For rsync to cloud

```
rsync --exclude='.git/' -Pavz -e "ssh -i ~/.ssh/google_compute_engine" jingbo@34.82.149.142:/home/jingbo/forex_trading .
```