1. qsub -I -q k40 -l walltime=12:00:00,nodes=1:ppn=20
    * queue the job

2. singularity shell -B /project,/work --nv /home/admin/singularity/pytorch-1.0.0-dockerhub-v3.simg
    * initializes a singularity shell that will allow pytorch to be run on an os that cannot run pytorch otherwise
    * '-B /directory0, /directory1' binds the directories to the singularity shell, allowing the user to access all files within while inside the shell

3. unset PYTHONPATH
4. unset PYTHONHOME
    * 2 commands that remove any prior python paths so that the virtual environment is used
    * must be done everytime before an environment is created or activated

5. conda create -n env_name python=3.7
    * creates a virtual conda environment named 'pytorch.' can be skipped if an environment has already been made

6. source activate env_name
    * activates the virtual conda environment. depending on the conda version, 'source' may be replaced by 'conda'

7. conda install pytorch torchvision cudatoolkit=9.0 -c env_name
    * installs pytorch and torchvision into the environment.
    * steps 5 and 7 can be skipped if a conda virtual environment has already been created with pytorch installed

8. export LD_LIBRARY_PATH=/usr/local/onnx/onnx:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib64:/.singularity.d/libs
    * fixes an error: "undefined symbol: _ZTIN2at11TypeDefaultE....."

9. cd /work/user/bionoi_autoencoder
    * cd into directory containing the autoencoder.py

10. python autoencoder_general.py -data_dir /work/jfeins1/bae-test-122k/ -model_file ./log/conv1x1-120k-batch512.pt -style conv_1x1 -batch_size 512
    * the command to run the autoencoder file. 
    * python reconstruct.py -data_dir /work/jfeins1/bae-test-122k/ -model /work/jfeins1/bionoi_autoencoder/log/conv1x1-120k.pt -style conv_1x1
    
***You must have a .condarc file in your home directory with the paths to your anaconda environments and packages.  Note that .condarc is a hidden file. Must be formatted as below.
```
envs_dirs:
- /work/user/anaconda3/envs
pkgs_dirs:
- /work/user/anaconda3/pkgs 
```