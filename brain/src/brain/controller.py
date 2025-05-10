from brain.manager import Manager


class Controller:
    def __init__(self):
        self.manager = Manager()

    def work_mode(self):
        """Activates the default working behavior"""
        self.manager.event_queue.put("work_mode")
        return self.manager.job_done_queue.get()
    def interrupt(self):
        self.manager.interrupt()

    def standby(self):
        """Sets robot to standby mode"""
        self.manager.event_queue.put("standby")
        return self.manager.job_done_queue.get()

    def halt(self):
        """Stops the robot and maintains current position"""
        self.manager.event_queue.put("halt")
        return self.manager.job_done_queue.get()


    def move(self, x: float, y: float):
        """Moves the robot in X-Y coordinates"""
        event = f"move;{x};{y}"
        self.manager.event_queue.put(event)
        return self.manager.job_done_queue.get()

    def gotopoint(self, location_name: str, location_type: str):
        """Drives the robot to a specific location"""
        event = f"gotopoint;{location_name};{location_type}"
        self.manager.event_queue.put(event)
        return self.manager.job_done_queue.get()

    def sleep(self, seconds: float):
        """Makes the robot wait"""
        event = f"sleep;{seconds}"
        self.manager.event_queue.put(event)
        return self.manager.job_done_queue.get()

    def wait_until(self):
        event = f"wait_until"
        self.manager.event_queue.put(event)
        return self.manager.job_done_queue.get()
    
    def stop_event(self):
        self.manager.stop_event()
    def resume_from_stop(self):
        self.manager.resume_from_stop()

    def save_position(self, location_name: str):
        """Saves the current position"""
        event = f"save_position;{location_name}"
        self.manager.event_queue.put(event)
        return self.manager.job_done_queue.get()

    def get_tool_info(self, tool_name):
        """Get information about a specific tool"""
        if tool_name not in self.create_tool_mapping():
            raise ValueError(f"Unknown tool: {tool_name}")
        return self.tools[tool_name]

    def list_available_tools(self):
        """List all available tools and their descriptions"""
        tools = self.create_tool_mapping()
        return {
            name: {"description": tool["description"], "parameters": tool["parameters"]}
            for name, tool in tools.items()
        }

    def create_tool_mapping(self):
        """Creates a mapping between tool names and controller methods"""
        return {
            "work_mode": {
                "function": self.work_mode,
                "parameters": {},
                "description": "Activates the default working behavior",
            },
            "standby": {
                "function": self.standby,
                "parameters": {},
                "description": "Sets robot to standby mode",
            },
            "halt": {
                "function": self.halt,
                "parameters": {},
                "description": "Stops the robot and maintains current position",
            },

            "move": {
                "function": self.move,
                "parameters": {
                    "x": {
                        "type": "number",
                        "description": "Distance in meters on X-axis",
                    },
                    "y": {
                        "type": "number",
                        "description": "Distance in meters on Y-axis",
                    },
                },
                "description": "Moves the robot in X-Y coordinates",
            },
            "gotopoint": {
                "function": self.gotopoint,
                "parameters": {
                    "location_name": {
                        "type": "string",
                        "description": "Name of the target location",
                    },
                    "location_type": {
                        "type": "string",
                        "enum": ["predefined", "saved"],
                        "description": "Type of the location",
                    },
                },
                "description": "Drives the robot to a specific location",
            },
            "sleep": {
                "function": self.sleep,
                "parameters": {
                    "seconds": {
                        "type": "number",
                        "description": "Number of seconds to wait",
                    }
                },
                "description": "Makes the robot wait",
            },
            "wait_until": {
                "function": self.wait_until,
                "parameters": {
                },
                "description": "Makes the robot wait until a specified external interrupt.",
            },
            "save_position": {
                "function": self.save_position,
                "parameters": {
                    "location_name": {
                        "type": "string",
                        "description": "Name for the new location",
                    }
                },
                "description": "Saves the current position",
            },
        }


controller = Controller()
