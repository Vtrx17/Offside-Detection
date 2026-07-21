# Offside-Detection

A computer vision offside-detection tool for football (soccer), built on top of a player/ball tracking pipeline. Given a broadcast-style frame or video, it tracks players, identifies teams by jersey color, and draws the offside line based on the 2nd-last defender.

## How it works

1. **Tracking** — YOLO-based detection and tracking of players, referees, and the ball
2. **Team assignment** — K-means clustering on jersey colors to split players into two teams
3. **Perspective setup** — click 4 points on the field to compute a homography, so player positions can be measured on a flat top-down view instead of raw image pixels
4. **Offside line** — finds the 2nd-last defender (accounting for goalkeeper), compares against the ball position, and flags any attacker ahead of that line
5. **Visualization** — draws the offside line and marks attackers ON/OFF on the original frame

## Built on

This started from a base football-tracking pipeline (player/ball tracking, camera movement compensation, speed & distance estimation) — the offside logic (`offside_detector.py`, `detect_offside_image.py`) is my own addition on top of it.

## Requirements

- Python 3.x
- ultralytics (YOLO)
- OpenCV
- NumPy

Model weights and input videos aren't included in the repo (too large for git) — drop your own YOLO weights in `models/` and video/image in `input_videos/` before running.

## Status

Still a work-in-progress / learning project — offside logic works on single frames with manual perspective setup; not yet automated for full match video.
