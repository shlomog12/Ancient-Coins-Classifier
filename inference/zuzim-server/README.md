# Docker Zuzim-Server


## Getting Started

These instructions will cover usage information and for the docker container 

### Prerequisities


In order to run this container you'll need docker installed.

* [Windows](https://docs.docker.com/windows/started)
* [OS X](https://docs.docker.com/mac/started/)
* [Linux](https://docs.docker.com/linux/started/)


## build
```shell
docker build -t zuzim-server:1.0.0 .
```

## run

```shell
docker run -d -p 3555:3555 -v $(pwd)/models:/app/models --name zuzim_server zuzim-server:1.0.0
```
## GET
http://localhost:3555/

## POST
http://localhost:3555/api/run_model/

### parames:<br>
  * image file<br>
### return:<br>
  * results of model