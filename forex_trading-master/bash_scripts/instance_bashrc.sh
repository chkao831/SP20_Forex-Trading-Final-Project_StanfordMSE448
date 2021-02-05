# source instance_bashrc.sh

export CONDA="/usr/local/cuda/bin:/opt/anaconda3/bin:/opt/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:"

conda_activate(){
	source activate shared_sprinkler
	echo "Sprinkler activated!"
	export PATH=$CONDA
}

conda_activate