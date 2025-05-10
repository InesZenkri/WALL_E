from doctest import FAIL_FAST, TestResults
import threading
import queue
from collections import deque
import time
import requests
from brain.settings import get_settings
from loguru import logger
import math

# Configure loguru
logger.add("manager.log", rotation="10 MB", level="DEBUG")



class Manager:
    def __init__(self):
        # Create an event queue
        self.event_queue = queue.Queue()
        self.interrupt_queue = queue.Queue()
        self.job_done_queue = queue.Queue()
        self.interrupt_state = False
        self.mode = deque(maxlen=20)
        self.mode.append("work_mode")
        self.mode.append("work_mode")
        self.mode.append("work_mode")
        self.positions = {}
        self.positions["position_storage"] = {"x": 1, "y": 0}
        self.positions["position_shelf"] = {"x": 2, "y": 0} 
        self.positions["position_home"] = {"x": 0, "y": 0}
        self.storage_positions = {}
        self.first = False

        logger.info("Initializing Manager")
        logger.debug(f"Initial mode queue: {list(self.mode)}")

        # Create and configure the daemon thread
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.running = True

        # Start the daemon thread
        self.thread.start()
        logger.info("Manager daemon thread started")

        self._new_event = False

    def loop(self):
        logger.info("Event loop started")
        while self.running:
            try:
                # Get event from queue with timeout
                event = None
                try:
                    event = self.event_queue.get(timeout=1)
                except queue.Empty:
                    pass
                    
                self.interrupt_state = event == "interrupt"
                self._new_event = False
                if event is None or self.interrupt_state:
                    event = self.mode[-1]
                    logger.debug(f"No event in queue, using default mode: {event}")
                else:
                    self.first = True
                    self._new_event = True
                    self.mode.append(event)
                    logger.debug(f"New event appended to mode queue: {list(self.mode)}")
                if event == "stop_event":
                    if self._new_event:
                        self.stop_event()
                    continue
                if event == "resume_from_stop":
                    if self.mode[-2] == "stop_event":
                        self.mode.pop(-2)
                        if self.mode[-2] == "move":
                            self.send_move()
                    self.mode.pop(-1)
                    
                split_event = event.split(";")
                method = getattr(self, split_event[0])

                # Call the method with parameters if they exist
                if len(split_event) > 1:
                    #logger.info(
                    #    f"Executing {split_event[0]} with parameters: {split_event[1:]}"
                    #)
                    output = method(*split_event[1:])
                else:
                    #logger.info(f"Executing {split_event[0]} with no parameters")
                    output = method()
                if output and self.job_done_queue.empty() and self.first:
                    self.first = False
                    self.job_done_queue.put({"status": True})
                if not self.event_queue.empty():
                    self.event_queue.task_done()

            
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                logger.exception("Full exception details:")


    def stop_event(self):
        response = requests.get(f"{get_settings().URL}/api/ros/sew_bot/pose")
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        current_pose = response.json()
        x = current_pose["transform"]["translation"]["x"]
        y = current_pose["transform"]["translation"]["y"]
        w = current_pose["transform"]["rotation"]["w"]
        z = current_pose["transform"]["rotation"]["z"]
        y = current_pose["transform"]["rotation"]["y"]
        x = current_pose["transform"]["rotation"]["x"]



        navigate_payload = {
                    "pose": {
                        "orientation": {
                            "w": w,
                            "x": x,
                            "y": y,
                            "z": z
                        },
                        "position": {
                            "x": x,
                            "y": y,
                            "z": 0
                        }
                    }
                }

        logger.info(f"Sending navigation command: {navigate_payload}")
        
        _ = requests.post(
            f"{get_settings().URL}/api/ros/sew_bot/goal_pose",
            json=navigate_payload,
            headers={"Content-Type": "application/json"}
)

    def resume_from_stop(self):
        pass

    def work_mode(self):
        logger.info("Executing work_mode")
        return True

    def standby(self):
        logger.info("Executing standby")
        return True

    def unstuck_position(self):
        logger.info("Executing unstuck_position")
        return True

    def resume(self):
        logger.info("Executing resume")
        logger.debug(f"Mode queue before resume: {list(self.mode)}")
        self.mode.popleft()
        self.mode.popleft()
        logger.debug(f"Mode queue after resume: {list(self.mode)}")
        return True

    def halt(self):
        logger.info("Executing halt")
        return True

    def wait_until(self):
        logger.info("Executing wait until")
        if self.interrupt_queue.empty():
            return False
        self.interrupt_queue.get() 
        return True

    def get_current_position(self):
        response = requests.get(f"{get_settings().URL}/api/ros/sew_bot/pose")
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        current_pose = response.json()
        x = current_pose["transform"]["translation"]["x"]
        y = current_pose["transform"]["translation"]["y"]
        return x, y 

    def move(self, x, y):
        try:
            # Fetch the current pose from the API
            current_x, current_y = self.get_current_position()
            if self._new_event:
                self.target_x = round(float(x) + current_x,1)
                self.target_y = round(float(y) + current_y,1)
                # send move 

                
            

            # Log the current and target positions
            #logger.info(f"Current pose: x={current_x}, y={current_y}")
            #logger.info(f"Target pose: x={self.target_x}, y={self.target_y}")

            # Calculate the differences
            diff_x = ((current_x - self.target_x)**2)
            diff_y = ((current_y - self.target_y)**2)

            # Check if the differences are within the threshold
            return math.sqrt(diff_x + diff_y) < 0.1
               
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch current pose: {e}")
            return False
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            return False
    def send_move(self):
        logger.info(f"Moving to target position: x={self.target_x}, y={self.target_y}")
                
        navigate_payload = {
            "pose": {
                "orientation": {
                    "w": 1.0,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0
                },
                "position": {
                    "x": self.target_x,
                    "y": self.target_y,
                    "z": 0
                }
            }
        }

        logger.info(f"Sending navigation command: {navigate_payload}")
        
        _ = requests.post(
            f"{get_settings().URL}/api/ros/sew_bot/goal_pose",
            json=navigate_payload,
            headers={"Content-Type": "application/json"}
        )

    def gotopoint(self, location_name, location_type):
        if self._new_event:
            logger.info(f"Going to point: {location_name} (type: {location_type})")
            if location_type == "predefined":
                position = self.positions[location_name]
            else:
                position = self.storage_positions[location_name]
            current_x, current_y = self.get_current_position()
            target_x = position["x"]
            target_y = position["y"]
            
            return self.move(target_x- current_x, target_y- current_y)
        return self.move(0, 0)

    def sleep(self, seconds):
        seconds = float(seconds)
        if self._new_event:
            logger.info(f"Sleeping for {seconds} seconds")
            self.target_time =  time.time() + seconds
        return self.target_time < time.time()

    def save_position(self, location_name):
        logger.info(f"Saving position with name: {location_name}")
        x, y = self.get_current_position()
        self.storage_positions[location_name] = { "x": x, "y": y }
        return True

    def stop(self):
        logger.info("Stopping manager")
        self.running = False
        self.thread.join()
        logger.info("Manager stopped successfully")
        return True

    def interrupt(self):
        self.interrupt_queue.put("interrupt")
        self.event_queue.put("interrupt")
        
