import time
from torchvision import transforms
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO  # For pedestrian detection (YOLOv8)
import torch
import requests
import sys
import neoapi


class GestureRecognizer:
    def __init__(self):
        # flag for sending a stop command
        self.stop_flag = False

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Load YOLOv8 model for pedestrian detection
        self.yolo_model = YOLO("yolo11m.pt")  # Use YOLOv8 nano for real-time performance
        self.yolo_model.to("cuda")

        # Load STOP sign image
        self.stop_sign = cv2.imread("stop_sign.png", cv2.IMREAD_UNCHANGED)
        # If stop sign image is not found, create a simple one
        if self.stop_sign is None:
            self.create_stop_sign()

        # Track hand movement for "Come" gesture
        self.prev_hand_positions = {}  # Dictionary to store previous hand positions
        self.hand_movement_history = {}  # Dictionary to track movement direction
        self.gesture_mode = {}  # Dictionary to track if a hand is in "Stop" or "Come" mode

    def create_stop_sign(self):
        """Create a simple STOP sign if image is not available"""
        # Create a 200x200 transparent image
        self.stop_sign = np.zeros((200, 200, 4), dtype=np.uint8)

        # Draw a red octagon
        center = (100, 100)
        radius = 90
        points = []
        for i in range(8):
            angle = i * (2 * np.pi / 8)
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(self.stop_sign, [points], (0, 0, 255, 220))  # Red with some transparency

        # Add "STOP" text
        cv2.putText(
            self.stop_sign,
            "STOP",
            (40, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255, 255),  # White
            3,
            cv2.LINE_AA
        )


    def detect_hand_movement(self, hand_id, hand_landmarks, frame_width, frame_height):
        """
        Detect if the hand is moving in a beckoning motion (for "Come" gesture)
        or stationary (for "Stop" gesture).
        """
        # Get the center of the palm
        palm_center_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame_width
        palm_center_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame_height

        current_position = (palm_center_x, palm_center_y)

        # Initialize if this is a new hand
        if hand_id not in self.prev_hand_positions:
            self.prev_hand_positions[hand_id] = current_position
            self.hand_movement_history[hand_id] = []
            self.gesture_mode[hand_id] = "Stop"  # Default to Stop
            return "Stop"

        # Calculate movement
        prev_x, prev_y = self.prev_hand_positions[hand_id]
        dx = current_position[0] - prev_x
        dy = current_position[1] - prev_y

        # Update previous position
        self.prev_hand_positions[hand_id] = current_position

        # Determine movement direction (simplify to horizontal movement)
        # Ignore small movements (noise)
        movement_threshold = 5.0

        if abs(dx) > movement_threshold:
            # Record horizontal movement direction
            direction = "right" if dx > 0 else "left"
            self.hand_movement_history[hand_id].append(direction)

            # Keep only the last 10 movements
            if len(self.hand_movement_history[hand_id]) > 10:
                self.hand_movement_history[hand_id].pop(0)

            # Check for beckoning pattern (alternating left-right movements)
            if len(self.hand_movement_history[hand_id]) >= 4:
                # Check if there's a pattern of alternating directions
                alternating = True
                for i in range(1, len(self.hand_movement_history[hand_id])):
                    if self.hand_movement_history[hand_id][i] == self.hand_movement_history[hand_id][i-1]:
                        alternating = False
                        break

                if alternating:
                    self.gesture_mode[hand_id] = "Come"
                    return "Come"

        # If no significant movement or no beckoning pattern, use the current mode
        return self.gesture_mode[hand_id]

    def overlay_stop_sign(self, frame, hand_landmarks):
        """Overlay a STOP sign on an open hand with robust error handling"""
        try:
            # Get the center of the palm
            palm_center_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame.shape[1])
            palm_center_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame.shape[0])

            # Calculate the size of the stop sign based on hand size
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate hand size
            hand_height = abs(middle_tip.y - wrist.y) * frame.shape[0]
            stop_size = int(hand_height * 1.5)  # Make stop sign proportional to hand size

            if stop_size < 30:  # Minimum size
                stop_size = 30

            # Resize stop sign
            resized_stop = cv2.resize(self.stop_sign, (stop_size, stop_size), interpolation=cv2.INTER_AREA)

            # Calculate position to center the stop sign on the palm
            x_offset = palm_center_x - stop_size // 2
            y_offset = palm_center_y - stop_size // 2

            # Ensure the stop sign stays within frame boundaries
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)

            # Calculate the actual width and height that will fit in the frame
            actual_width = min(stop_size, frame.shape[1] - x_offset)
            actual_height = min(stop_size, frame.shape[0] - y_offset)

            # If the sign would be too small, don't overlay it
            if actual_width <= 0 or actual_height <= 0:
                return frame

            # Crop the resized stop sign to fit within the frame
            cropped_stop = resized_stop[:actual_height, :actual_width]

            # Get the region of the frame where we'll overlay the sign
            roi = frame[y_offset:y_offset+actual_height, x_offset:x_offset+actual_width]

            # Ensure ROI and cropped_stop have the same dimensions
            if roi.shape[:2] != cropped_stop.shape[:2]:
                # If they don't match, resize cropped_stop to match roi
                cropped_stop = cv2.resize(cropped_stop, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)

            # Create a mask from the alpha channel
            alpha_mask = cropped_stop[:, :, 3] / 255.0

            # Apply the mask to each channel
            for c in range(3):  # RGB channels
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + cropped_stop[:, :, c] * alpha_mask

            # Update the frame with the modified ROI
            frame[y_offset:y_offset+actual_height, x_offset:x_offset+actual_width] = roi

            return frame
        except Exception as e:
            # If any error occurs, just return the original frame
            print(f"Error in overlay_stop_sign: {e}")
            return frame

    def overlay_come_sign(self, frame, hand_landmarks):
        """Overlay a COME sign on an open hand with robust error handling"""
        try:
            # Get the center of the palm
            palm_center_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame.shape[1])
            palm_center_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame.shape[0])

            # Calculate the size of the come sign based on hand size
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate hand size
            hand_height = abs(middle_tip.y - wrist.y) * frame.shape[0]
            come_size = int(hand_height * 1.5)  # Make come sign proportional to hand size

            if come_size < 30:  # Minimum size
                come_size = 30

            # Resize come sign
            resized_come = cv2.resize(self.come_sign, (come_size, come_size), interpolation=cv2.INTER_AREA)

            # Calculate position to center the come sign on the palm
            x_offset = palm_center_x - come_size // 2
            y_offset = palm_center_y - come_size // 2

            # Ensure the come sign stays within frame boundaries
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)

            # Calculate the actual width and height that will fit in the frame
            actual_width = min(come_size, frame.shape[1] - x_offset)
            actual_height = min(come_size, frame.shape[0] - y_offset)

            # If the sign would be too small, don't overlay it
            if actual_width <= 0 or actual_height <= 0:
                return frame

            # Crop the resized come sign to fit within the frame
            cropped_come = resized_come[:actual_height, :actual_width]

            # Get the region of the frame where we'll overlay the sign
            roi = frame[y_offset:y_offset+actual_height, x_offset:x_offset+actual_width]

            # Ensure ROI and cropped_come have the same dimensions
            if roi.shape[:2] != cropped_come.shape[:2]:
                # If they don't match, resize cropped_come to match roi
                cropped_come = cv2.resize(cropped_come, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)

            # Create a mask from the alpha channel
            alpha_mask = cropped_come[:, :, 3] / 255.0

            # Apply the mask to each channel
            for c in range(3):  # RGB channels
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + cropped_come[:, :, c] * alpha_mask

            # Update the frame with the modified ROI
            frame[y_offset:y_offset+actual_height, x_offset:x_offset+actual_width] = roi

            return frame
        except Exception as e:
            # If any error occurs, just return the original frame
            print(f"Error in overlay_come_sign: {e}")
            return frame

    def calculate_palm_orientation(self, hand_landmarks, handedness):
        """
        Determine if the palm is facing the camera using multiple methods for robustness.
        """
        # Method 1: Cross product of vectors
        # Get key landmarks
        wrist = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].z
        ])

        index_mcp = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].z
        ])

        pinky_mcp = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].z
        ])

        # Calculate vectors
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        # Calculate cross product
        cross_product = np.cross(v1, v2)

        # Normalize the cross product
        normal = cross_product / np.linalg.norm(cross_product)

        # Method 2: Compare z-coordinates of knuckles vs fingertips
        # This helps with back of hand detection
        index_tip = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].z
        ])

        middle_tip = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
        ])

        # For back of hand, fingertips are typically further away (larger z) than knuckles
        fingertip_avg_z = (index_tip[2] + middle_tip[2]) / 2
        knuckle_avg_z = (index_mcp[2] + pinky_mcp[2]) / 2

        # Determine if fingertips are further away than knuckles
        fingertips_further = fingertip_avg_z > knuckle_avg_z

        # Method 3: Check relative positions of thumb and pinky
        thumb_tip = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y
        ])

        pinky_tip = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y
        ])

        # For right hand with palm facing camera: thumb is to the left of pinky
        # For left hand with palm facing camera: thumb is to the right of pinky
        thumb_pinky_check = (handedness == "Right" and thumb_tip[0] < pinky_tip[0]) or \
                           (handedness == "Left" and thumb_tip[0] > pinky_tip[0])

        # Combine methods for more robust detection
        # The z-component of the normal vector indicates palm orientation
        # For right hand: positive z means palm facing camera
        # For left hand: negative z means palm facing camera
        cross_product_check = (normal[2] > 0 and handedness == "Right") or (normal[2] < 0 and handedness == "Left")

        # Weight the methods (can be adjusted based on performance)
        # If at least 2 of 3 methods agree, we consider it palm facing
        methods_agree = sum([cross_product_check, not fingertips_further, thumb_pinky_check]) >= 2

        return "Palm Facing Camera" if methods_agree else "Back of Hand Facing Camera"

    def check_fingers_extended(self, hand_landmarks, handedness):
        """
        Check if fingers are extended using a more robust method.
        Returns an array of booleans for each finger and the thumb.
        """
        # Get fingertip landmarks
        fingertips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]

        # Get MCP landmarks
        mcps = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        ]

        # Get PIP landmarks
        pips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        ]

        # Get DIP landmarks
        dips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP]
        ]

        # Get wrist landmark
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Determine palm orientation for context
        palm_orientation = self.calculate_palm_orientation(hand_landmarks, handedness)
        is_palm_facing = palm_orientation == "Palm Facing Camera"

        # Check if each finger is extended
        fingers_extended = []
        for i in range(4):  # For each finger (excluding thumb)
            # Calculate distances
            tip_to_mcp_distance = np.sqrt(
                (fingertips[i].x - mcps[i].x)**2 +
                (fingertips[i].y - mcps[i].y)**2
            )

            pip_to_mcp_distance = np.sqrt(
                (pips[i].x - mcps[i].x)**2 +
                (pips[i].y - mcps[i].y)**2
            )

            # Calculate angles between finger segments
            # Vector from PIP to MCP
            pip_to_mcp = np.array([mcps[i].x - pips[i].x, mcps[i].y - pips[i].y])
            # Vector from PIP to DIP
            pip_to_dip = np.array([dips[i].x - pips[i].x, dips[i].y - pips[i].y])

            # Normalize vectors
            if np.linalg.norm(pip_to_mcp) > 0 and np.linalg.norm(pip_to_dip) > 0:
                pip_to_mcp = pip_to_mcp / np.linalg.norm(pip_to_mcp)
                pip_to_dip = pip_to_dip / np.linalg.norm(pip_to_dip)

                # Calculate dot product and angle
                dot_product = np.dot(pip_to_mcp, pip_to_dip)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
            else:
                angle = 0

            # Different criteria based on palm orientation
            if is_palm_facing:
                # For palm facing camera: A finger is extended if the fingertip is further from the MCP than the PIP
                # and the fingertip is above the PIP (y-coordinate is smaller)
                is_extended = (tip_to_mcp_distance > pip_to_mcp_distance * 1.2) and (fingertips[i].y < pips[i].y)

                # Additional check: fingertip should be higher than wrist for a truly extended finger
                is_extended = is_extended and (fingertips[i].y < wrist.y)
            else:
                # For back of hand: Use angle between finger segments and check if finger is relatively straight
                is_extended = angle > 160 or (tip_to_mcp_distance > pip_to_mcp_distance * 1.2)

                # For back of hand, we don't strictly require the fingertip to be above the wrist
                # as the hand might be in various orientations

            fingers_extended.append(is_extended)

        # Check thumb separately - this is more complex due to thumb's different orientation
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        thumb_cmc = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]

        # Calculate distances
        tip_to_mcp_distance = np.sqrt(
            (thumb_tip.x - thumb_mcp.x)**2 +
            (thumb_tip.y - thumb_mcp.y)**2
        )

        ip_to_mcp_distance = np.sqrt(
            (thumb_ip.x - thumb_mcp.x)**2 +
            (thumb_ip.y - thumb_mcp.y)**2
        )

        # Different criteria based on handedness and palm orientation
        if is_palm_facing:
            # For palm facing camera
            if handedness == "Right":
                thumb_extended = (thumb_tip.x < thumb_ip.x) and (tip_to_mcp_distance > ip_to_mcp_distance)
            else:  # Left hand
                thumb_extended = (thumb_tip.x > thumb_ip.x) and (tip_to_mcp_distance > ip_to_mcp_distance)
        else:
            # For back of hand, the thumb is often visible and extended when the hand is open
            # Use a simpler distance-based check
            thumb_extended = tip_to_mcp_distance > ip_to_mcp_distance * 1.2

        return fingers_extended, thumb_extended

    def recognize_gestures(self, hand_landmarks, handedness, hand_id, frame_width, frame_height):
        """
        Recognize gestures based on hand landmarks and handedness.
        Returns the name of the gesture and whether it's a stop/come gesture.
        """
        # Check if fingers are extended using the robust method
        fingers_extended, thumb_extended = self.check_fingers_extended(hand_landmarks, handedness)

        # Determine palm orientation
        palm_orientation = self.calculate_palm_orientation(hand_landmarks, handedness)

        # Recognize gestures
        if all(fingers_extended) and thumb_extended:
            if palm_orientation == "Palm Facing Camera":
                # Detect if it's a "Come" or "Stop" gesture based on movement
                # gesture_mode = self.detect_hand_movement(hand_id, hand_landmarks, frame_width, frame_height)

                # if gesture_mode == "Come":
                #     return "Come", "Come"
                # else:
                return "Open Hand (Palm)", "Stop"
            else:
                return "Open Hand (Back)", None
        # elif not any(fingers_extended) and not thumb_extended:
        #     return "Fist", None
        # elif fingers_extended[0] and not any(fingers_extended[1:]) and not thumb_extended:
        #     return "Pointing", None
        # elif fingers_extended[0] and fingers_extended[1] and not any(fingers_extended[2:]):
        #     return "Peace Sign", None
        # elif thumb_extended and not any(fingers_extended):
        #     return "Thumbs Up", None
        # elif not thumb_extended and not fingers_extended[0] and all(fingers_extended[1:]):
        #     return "Rock On", None
        # elif fingers_extended[0] and fingers_extended[3] and not fingers_extended[1] and not fingers_extended[2]:
        #     return "Spider-Man", None
        # elif all(fingers_extended) and not thumb_extended:
        #     return "Four Fingers", None
        # elif fingers_extended[0] and fingers_extended[1] and fingers_extended[2] and not fingers_extended[3]:
        #     return "Three Fingers", None
        # elif not thumb_extended and not fingers_extended[0] and not fingers_extended[1] and fingers_extended[2] and fingers_extended[3]:
        #     return "Horns", None
        # elif thumb_extended and fingers_extended[0] and not any(fingers_extended[1:]):
        #     return "Gun", None
        # elif thumb_extended and fingers_extended[3] and not any([fingers_extended[0], fingers_extended[1], fingers_extended[2]]):
        #     return "Hang Loose", None
        else:
            return "Unknown Gesture", None

    def draw_debug_info(self, frame, hand_landmarks, handedness):
        """
        Draw detailed debugging information for hand orientation and finger extension.
        """
        # Get key landmarks
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        # Calculate palm orientation
        palm_orientation = self.calculate_palm_orientation(hand_landmarks, handedness)

        # Check finger extension
        fingers_extended, thumb_extended = self.check_fingers_extended(hand_landmarks, handedness)

        # Draw palm orientation text
        cv2.putText(
            frame,
            f"Palm: {palm_orientation}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        # Draw finger extension status
        finger_names = ["Index", "Middle", "Ring", "Pinky"]
        for i, (name, extended) in enumerate(zip(finger_names, fingers_extended)):
            color = (0, 255, 0) if extended else (0, 0, 255)
            cv2.putText(
                frame,
                f"{name}: {'Extended' if extended else 'Not Extended'}",
                (10, 90 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        # Draw thumb extension status
        thumb_color = (0, 255, 0) if thumb_extended else (0, 0, 255)
        cv2.putText(
            frame,
            f"Thumb: {'Extended' if thumb_extended else 'Not Extended'}",
            (10, 90 + len(finger_names) * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            thumb_color,
            2,
            cv2.LINE_AA
        )

    def detect_pedestrians(self, frame):
        """
        Detect pedestrians using YOLOv8.
        """
        # Get original dimensions
        height, width = frame.shape[:2]
        
        # Calculate border width (e.g., 20% of the frame width on each side)
        border_width = int(width * 0.2)
        
        # Create a new frame with borders
        bordered_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Copy the center portion of the original frame
        bordered_frame[:, border_width:width-border_width] = frame[:, border_width:width-border_width]
        
        # Apply histogram equalization to improve contrast
        gray = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        equalized_color = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        
        # Let YOLO handle the tensor conversion and GPU placement
        results = self.yolo_model.track(equalized_color, persist=True)
        
        # Process results if they exist
        if results and len(results) > 0:
            detections = results[0].boxes.data.cpu().numpy()  # Get bounding boxes

            for box in detections:
                x1, y1, x2, y2, id, conf, cls = box
                if int(cls) == 0:  # Class 0 is "person" in YOLO
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Pedestrian N:{int(id)} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
                    self.send_stop()
        
        return frame

    def process_frame(self, frame):
        """
        Process a single frame for gesture recognition and debugging.
        """
        start = time.time()
        # # Convert the BGR image to RGB for MediaPipe
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # Process the frame for hand detection
        # hand_results = self.hands.process(rgb_frame)

        # Process the frame for pedestrian detection
        frame = self.detect_pedestrians(frame)

        # # Draw hand landmarks and debugging information
        # if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        #     for i, (hand_landmarks, handedness) in enumerate(zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
        #         # Draw the hand landmarks
        #         self.mp_drawing.draw_landmarks(
        #             frame,
        #             hand_landmarks,
        #             self.mp_hands.HAND_CONNECTIONS,
        #             self.mp_drawing_styles.get_default_hand_landmarks_style(),
        #             self.mp_drawing_styles.get_default_hand_connections_style()
        #         )
        #
        #         # Get handedness (Left or Right)
        #         hand_label = handedness.classification[0].label
        #
        #         # Create a unique ID for this hand
        #         hand_id = f"{hand_label}_{i}"
        #
        #         # Recognize and display the gesture
        #         gesture, action_type = self.recognize_gestures(
        #             hand_landmarks,
        #             hand_label,
        #             hand_id,
        #             frame.shape[1],
        #             frame.shape[0]
        #         )
        #
        #         # Display gesture name
        #         cv2.putText(
        #             frame,
        #             f"Gesture: {gesture}",
        #             (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,
        #             (0, 255, 0),
        #             2,
        #             cv2.LINE_AA
        #         )
        #
        #         # If it's a stop or come gesture, overlay the appropriate sign
        #         if action_type == "Stop":
        #             frame = self.overlay_stop_sign(frame, hand_landmarks)
        #         elif action_type == "Come":
        #             frame = self.overlay_come_sign(frame, hand_landmarks)
        #
        #         # Draw debugging information
        #         self.draw_debug_info(frame, hand_landmarks, hand_label)
        frame_rate = 1 / (time.time() - start)
        cv2.putText(frame, f'{frame_rate} FPS' , (350, 30), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2, cv2.LINE_AA)
        return frame

    def send_stop(self):
        if not self.stop_flag:
            # Send transcription as JSON with the correct format
            try:
                print(f"Attempting to send to http://192.168.1.104:8000/command...")
                response = requests.post(
                    'http://192.168.1.104:8000/stop',
                    timeout=1000
                )
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")
                self.stop_flag = True
            except requests.exceptions.ConnectionError as e:
                print(
                    f"Connection Error: Could not connect to the server. Please check if the server is running at 192.168.1.104:8000")
                print(f"Detailed error: {str(e)}")
            except requests.exceptions.Timeout as e:
                print(f"Timeout Error: The server took too long to respond")
            except requests.exceptions.RequestException as e:
                print(f"Request Error: {str(e)}")

def main():
    # Initialize the webcam
    #cap = cv2.VideoCapture(0)
    camera = neoapi.Cam()
    camera.Connect()

    # Check if the webcam is opened correctly
    #if not cap.isOpened():
    #    print("Error: Could not open webcam.")
    #    return

    # Initialize the gesture recognizer
    recognizer = GestureRecognizer()

    print("Gesture Recognition with STOP/COME Sign Overlay Started. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam

        frame = camera.GetImage().GetNPArray()
        #print("frame222222", type(frame))
        success = True
        if not success:
            print("Error: Failed to read frame from webcam.")
            break

        # Process the frame for hand gesture recognition and debugging
        processed_frame = recognizer.process_frame(frame)

        # Display the processed frame
        cv2.imshow('Gesture Recognition with STOP/COME Sign', processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()