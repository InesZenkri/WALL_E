#  Docu For WALLE

![alt text](img/WALL_E.jpg)

## Build Backend

To build the FastAPI backend run

```bash
cd brain
python -m venv .venv
source .venv/bin/activate
pip install -e .
touch .env
```

Create a  **.env** file or set the parameters with export 

```bash
OPENAI_API_KEY=<YOUR-KEY-GOES-HERE>
OPENAI_MODEL=openai/gpt-4o
MAX_TOKENS=10000
```

Cool you have build and set up the Backend 🚀 .

Now it is time to get some real work done !

> [!NOTE]  
> If your robot is not using the IP **192.168.24.82** change it in the *brain/settings.py* accordingly

## Start Backend

```bash
python run.py
or
uvicorn brain.routes:app --reload --log-level=critical --host=0.0.0.0 --reload
```

something like this should come up 

```bash
2025-05-11 10:17:47.115 | INFO     | brain.manager:__init__:36 - Initializing Manager
2025-05-11 10:17:47.115 | INFO     | brain.manager:loop:50 - Event loop started
2025-05-11 10:17:47.115 | INFO     | brain.manager:__init__:45 - Manager daemon thread started
2025-05-11 10:17:47.118 | INFO     | brain.fast_api:startup_event:31 - 🚀 FastAPI app is starting up...
2025-05-11 10:17:48.116 | INFO     | brain.manager:work_mode:153 - Executing work_mode
```

get the API docu 

```bash
http://<YOUR-IP>:8000/docs
```

## Send Data to Backend

Use the web interface or simply curl.

This is only useful if you want to send simple commands and get the interpreted by the AI model.

```bash
curl -X 'POST' \
  'http://<YOUR-IP>:8000/command' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "warte das du beladen bist  fahre zur position_home vordefinierten"
} '
```

The robot fill perform a movement based on the interpretation.

We also have a 

**/stop** **/resume** and **/interrupt** command. They have no content.

### /stop 

Will stop the robot on the spot

### /resume

Will resume the movement to the last target position

### /interrupt

Is not used

## Send Data to Backend automatically

Ok so far you started the Backend and have to send some REST API comannds yourself. But you want to be cooler right? Do also this step automatically, right?

So you speek to the robot and it performes its movement automatically.

cd into the main repo folder and install all requirements

> [!NOTE]  
>  Install the camera driver to see the RGB Image from this image. [Link to Website](https://www.baumer.com/de/en/product-overview/industrial-cameras-image-processing/software/baumer-neoapi/c/42528)

```bash
pip install -r gesture/requirements.txt
pip install -r requirements.txt
```

start the Camera

```bash
python gesture/gesture_recognition.py
```

start voice recognition

```bash
python gesture/whisper_mic.py
```

the output should look smth like this if you are in front of the robot and do a stop gesture.

![alt text](img/stop.jpg)

# Next steps

Try more things. You can also speak to the robot. For example, move two meters forward, and the robot will move. If you stop it with a gesture, it will pause and continue its movement when you step out of the way.

Happy hacking! 😉

![alt text](img/WALL_SEW.jpg)
