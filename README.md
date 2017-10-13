# Pilot_online
The Pilot online package is a Tensorflow package used for online training.
The online training happens by interacting with the [simulation-supervised](https://www.github.com/kkelchte/simulation-supervised){:target="_blank"}
package in ROS and Gazebo.

## Dependencies
* [Tensorflow (>1.1)](https://www.tensorflow.org/install/) or [docker image](https://hub.docker.com/r/kkelchte/ros_gazebo_tensorflow/){:target="_blank"} up and running.
* [simulation-supervised](https://www.github.com/kkelchte/simulation-supervised){:target="_blank"}
* log: this directory contains checkpoints of trained models, required to reproduce results.


## Installation
You can use this code from within the [docker image](https://hub.docker.com/r/kkelchte/ros_gazebo_tensorflow/){:target="_blank"} I supply for the [Doshico challenge](http://kkelchte.github.io/doshico){:target="_blank"}.
```bash
$ git clone https://www.github.com/kkelchte/pilot_online
# within a running docker container
$$  python main.py
...
checkpoint: /home/klaas/tensorflow/log/mobilenet_small
('Successfully loaded model from:', '/home/klaas/tensorflow/log/mobilenet_small')
...
```
In order to make it work, you can either adjust some default flag values or adapt the same folder structure.
* summary_dir (main.py): log folder to keep checkpoints and log info: $HOME/tensorflow/log
* checkpoint_path (model.py): log folder from which checkpoints of models are read from: $HOME/tensorflow/log
It is best to download the log folder and save it on the correct relative path.