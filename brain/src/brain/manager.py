from doctest import FAIL_FAST, TestResults
import threading
import queue
from collections import deque
import time
from brain.routes import interrupt
from loguru import logger

# Configure loguru
logger.add("manager.log", rotation="10 MB", level="DEBUG")


class Manager:
    def __init__(self):
        # Create an event queue
        self.event_queue = queue.Queue()
        self.interrupt_queue = queue.Queue()
        self.job_done_queue = queue.Queue()ä
        self.interrupt_state = False
        self.mode = deque(maxlen=20)
        self.mode.append("work_mode")
        self.mode.append("work_mode")
        self.mode.append("work_mode")

        logger.info("Initializing Manager")
        logger.debug(f"Initial mode queue: {list(self.mode)}")

        # Create and configure the daemon thread
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.running = True

        # Start the daemon thread
        self.thread.start()
        logger.info("Manager daemon thread started")

    def loop(self):
        logger.info("Event loop started")
        while self.running:
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1)
                self.interrupt_state = event == "interrupt"
                if event is None or self.interrupt_state:
                    event = self.mode[0]
                    logger.debug(f"No event in queue, using default mode: {event}")
                else:
                    self.mode.append(event)
                    split_event = event.split(";")
                    logger.debug(f"New event appended to mode queue: {list(self.mode)}")

                method = getattr(self, split_event[0])

                # Call the method with parameters if they exist
                if len(split_event) > 1:
                    logger.info(
                        f"Executing {split_event[0]} with parameters: {split_event[1:]}"
                    )
                    output = method(*split_event[1:])
                else:
                    logger.info(f"Executing {split_event[0]} with no parameters")
                    output = method()
                if output:
                    self.job_done_queue.put({"status": True})
                self.event_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                logger.exception("Full exception details:")

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

    def move(self, x, y):
        logger.info(f"Moving to coordinates: x={x}, y={y}")
        return True

    def gotopoint(self, location_name, location_type):
        logger.info(f"Going to point: {location_name} (type: {location_type})")
        return True

    def sleep(self, seconds):
        logger.info(f"Sleeping for {seconds} seconds")
        return True

    def save_position(self, location_name):
        logger.info(f"Saving position with name: {location_name}")
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
        
