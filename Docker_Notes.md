# Notes on Docker and Data Science Application



## Docker Introduction



### Installation

Download Docker Desktop for Mac (see the [link](https://docs.docker.com/get-started/)). One can manage Docker images and containers on one's machine by **Docker Dashboard**.

![Docker_dashboard.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/Docker_dashboard.png?raw=true)

 

### Build an Image

A Docker image is a file used to execute code in a Docker container. Docker images act as a set of instructions to build a Docker container, like a template. 

Here is the simplest way to build an image:

```
$ docker build -t IMAGE_NAME:TAG .
```

For example, in the [instruction](https://docs.docker.com/get-started/02_our_app/) to build the "hello world" image by:

```
$ docker build -t getting-started .
```

the `-t` flag tags our image. Think of this simply as a human-readable name for the final image. Since we named the image `getting-started`, we can refer to that image when we run a container.

The `.` at the end of the `docker build` command tells that Docker should look for the `Dockerfile` in the current directory.



### Start a Container



 