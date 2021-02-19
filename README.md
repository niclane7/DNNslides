## Installation and usage slide



### Docker
Go to the [Nvidia docker github page](https://github.com/NVIDIA/nvidia-docker) and install nvidia-docker for your system.

First, ensure you're in this repo's directory using `cd <path_to_this_directory>`

Then, if you're using a Mac or Linux, run the following to start the container for this set of slides.
`docker run -it --rm --gpus all -p 8888:8888 -v $(pwd):/home/niclane niclane7/nbslides:latest`

If you're using Windows and running this in the DOS command line instead of WSL, run the following command.
`docker run -it --rm --gpus all -p 8888:8888 -v %CD%:/home/niclane niclane7/nbslides:latest`

If you do not have an Nvidia GPU, please remove `--gpus all` from the above commands.

While in the container, run the following to start the jupyter notebook.
`jupyter notebook --no-browser --allow-root --ip=0.0.0.0 home/niclane/`

You can then open the notebooks via the link the command prints out in your browser.

