# Docker Zuzim-Client


## Getting Started

These instructions will cover usage information and for the docker container 

### Prerequisities


In order to run this container you'll need docker installed.

* [Windows](https://docs.docker.com/windows/started)
* [OS X](https://docs.docker.com/mac/started/)
* [Linux](https://docs.docker.com/linux/started/)


## build
```shell
docker build -t zuzim-client:1.0.0 .
```

## run

```shell
docker run -d -p 3333:3000 --name zuzim_client zuzim-client:1.0.0
```