#!/bin/bash
# filepath: /home/lemmi25/Documents/Hackathon_Black_Forest/WALL_E/start-vnc.sh

# Set the VNC password
echo "$VNC_PASSWORD" | vncpasswd -f > /root/.vnc/passwd
chmod 600 /root/.vnc/passwd

# Start the VNC server
vncserver $DISPLAY -depth $VNC_COL_DEPTH -geometry $VNC_RESOLUTION

# Start the XFCE desktop environment
startxfce4 &
tail -f /dev/null
