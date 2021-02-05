sudo killall tensorboard

cd /mnt/shared_data/experiments
sudo tensorboard --logdir . --port=7000 &

cd /mnt/shared_data/old_experiments
sudo tensorboard --logdir . --port=7001 &

cd /mnt/shared_data/old_old_experiments
sudo tensorboard --logdir . --port=7002 &



screen -d -m bash -c 'tensorboard --logdir=gs://cs224w_sprinkler/experiments --bind_all --port=8964; exec sh'

# /usr/bin/screen
# /opt/anaconda3/bin/tensorboard
# https://timleland.com/how-to-run-a-linux-program-on-startup/
# https://superuser.com/questions/1276775/systemd-service-python-script-inside-screen-on-boot