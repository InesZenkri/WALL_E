# WALL_E



## Docker 
1. docker build -t ros2_humble -f .Dockerfile .
2. docker run -it --rm ros2_humble
3. ros2 --help

## Connect to Robot

Static ip of Pi on Robot 192.168.24.82

Start docker

```bash
docker compose build 
docker compose up -d 
docker compose exec -it app /ros_entrypoint.sh bash
```