## DeepNN Slides

The material here is offered based on two of my lectures that are part of the Deep Neural Networks class at the Computer Science and Technology department at the University of Cambridge: https://mlatcl.github.io/deepnn/

The two lectures in question were: "Convolutional Neural Networks" and "Hardware Ecosystem". Links to these lectures (including a video) are available here:

https://mlatcl.github.io/deepnn/lectures/03-01-hardware.html
https://mlatcl.github.io/deepnn/lectures/04-01-convolutional-neural-networks.html

### Installation and usage

Depending on how you want to view/use these slides you should install them in different ways. I'll describe these three ways below.

#### 1. Simply to view slides

If you only want to locally view the slides on your own machine, and aren't interested in running (and experimenting) with the code that is embedded then it is all very easy.... Simply sync to this repo, and run the two Jupyter Notebooks you find locally. As long as those notebooks have access to to the directory they expect to contain the images they will render fine.

This assumes you have Jupter Notebook software installed locally. 

Then you run from the command line something like: `jupyter notebook --no-browser --allow-root --ip=0.0.0.0 home/niclane/` where you replace the directory with the location where you have placed this repo.


if you want to view the slides just use jupiter notebook. only but you will see some errors.

sync to this repo.

if you want to see the code and interact with it do the below.

if you have cpu. do X

if you GPU do Y.

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

