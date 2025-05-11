#  Docu For WALLE

![alt text](img/WALL_E.jpg)

## Build Backend

to build the FastAPI Backend run

```bash
cd brain
python -m venv .venv
source .venv/bin/activate
pip install -e .
touch .env
```

in the **.venv** file add the following things

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
uvicorn brain.routes:app --reload --log-level=critical --host=0.0.0.0 --reload
```

something like this should come up 

```bash
2025-05-11 10:17:47.115 | INFO     | brain.manager:__init__:36 - Initializing Manager
2025-05-11 10:17:47.115 | INFO     | brain.manager:loop:50 - Event loop started
2025-05-11 10:17:47.115 | INFO     | brain.manager:__init__:45 - Manager daemon thread started
2025-05-11 10:17:47.118 | INFO     | brain.fast_api:startup_event:31 - 🚀 FastAPI app is starting up...
2025-05-11 10:17:48.116 | INFO     | brain.manager:work_mode:153 - Executing work_mode
2025-05-11 10:17:48.116 | INFO     | brain.manager:gotopoint:242 - Going to point: position_shelf (type: predefined)
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

MAX?

## Send Data to Backend automatically

Ok so far you started the Backend and have to send some REST API comannds yourself. But you want to be cooler right? Do also this step automatically, right?

So you speek to the robot and it performes its movement automatically.
