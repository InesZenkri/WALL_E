tools = [
    {
        "type": "function",
        "function": {
            "name": "work_mode",
            "description": "Activates the default working behavior. Robot continuously monitors for tasks and maintains optimal readiness.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "standby",
            "description": "Sets robot to standby mode. Minimal activity to conserve energy while waiting for new instructions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unstuck_position",
            "description": "Use if the robotor is stuck.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "halt",
            "description": "Stops the robot and maintains current position until further instructions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume",
            "description": "Resumes previous task or operation from last known state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Moves the robot in X-Y coordinates. X: Forward(+)/Backward(-), Y: Left(+)/Right(-). Minimum movement 1 meter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "Distance in meters on X-axis. Positive = forward, Negative = backward",
                    },
                    "y": {
                        "type": "number",
                        "description": "Distance in meters on Y-axis. Positive = left, Negative = right",
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gotopoint",
            "description": "Drives the robot to a specific predefined or saved location in the factory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": "Name of the target location. Can be a predefined location (position_a, position_b, position_c) or a previously saved custom location.",
                    },
                    "location_type": {
                        "type": "string",
                        "enum": ["predefined", "saved"],
                        "description": "Type of the location - either 'predefined' for factory default positions or 'saved' for custom saved positions",
                    },
                },
                "required": ["location_name", "location_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sleep",
            "description": "Makes the robot wait for specified number of seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "Number of seconds to wait",
                        "minimum": 0,
                    }
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_until",
            "description": "Makes the robot wait until a specified external interrupt.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_position",
            "description": "Saves the current position as a new named location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": "Name for the new location",
                    }
                },
                "required": ["location_name"],
            },
        },
    },
]
