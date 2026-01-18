"""
Offside Detection Module
Vertical offside line using last/2nd-last defender X position (image space).
"""

import cv2
import numpy as np


class OffsideDetector:
    def __init__(self, frame_width=1920, frame_height=1080):
        self.frame_width = frame_width
        self.frame_height = frame_height

    # --- DEFENDER LINE (X‑AXIS) ---

    def get_second_last_defender_x(self, defending_players, direction="right"):
        """
        Find 2nd-last defender X in image coords.
        direction="right": attackers go left->right, so 'deeper' defender has larger x.
        direction="left" : attackers go right->left, so 'deeper' defender has smaller x.
        """
        if not defending_players:
            return None

        xs = []
        for p in defending_players.values():
            bbox = p.get("bbox")
            pos = p.get("position_img") or p.get("x_img", [0, 0])
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                cx = (x1 + x2) / 2.0
            else:
                cx = pos[0]
            xs.append(cx)

        if len(xs) == 0:
            return None
        if len(xs) == 1:
            return xs[0]

        xs.sort()
        # 2nd deepest
        if direction == "right":
            return xs[-2]
        else:  # "left"
            return xs[1]

    # --- OFFSIDE LOGIC ---

    def check_offside_event(
        self,
        attacking_players,
        defending_players,
        ball_pos,
        frame_idx,
        direction="right",
    ):
        """
        Simple FIFA offside:
        - attacker must be closer to opponent goal line than BOTH ball and 2nd-last defender.
        - Here we work only along X‑axis in image space.
        """
        line_x = self.get_second_last_defender_x(defending_players, direction)
        if line_x is None:
            return {}

        if ball_pos is not None:
            ball_x = ball_pos[0]
        else:
            ball_x = line_x

        offside_status = {}
        for pid, p in attacking_players.items():
            bbox = p.get("bbox")
            pos = p.get("position_img") or p.get("x_img", [0, 0])
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                px = (x1 + x2) / 2.0
            else:
                px = pos[0]

            if direction == "right":
                nearer_goal = px > line_x and px > ball_x
            else:  # "left"
                nearer_goal = px < line_x and px < ball_x

            offside_status[pid] = "OFF" if nearer_goal else "ON"

        return offside_status

    # --- DRAWING ---

    def draw_offside_visualization(
        self,
        frame,
        attacking_players,
        defending_players,
        offside_status,
        ball_pos,
        direction="right",
    ):
        """
        Draw:
        - vertical yellow offside line at defender line_x
        - attackers: green / red circles (ON/OFF)
        - defenders: blue rectangles
        """
        h, w = frame.shape[:2]
        vis = frame.copy()

        line_x = self.get_second_last_defender_x(defending_players, direction)
        if line_x is not None:
            xi = int(line_x)
            if 0 <= xi < w:
                cv2.line(vis, (xi, 0), (xi, h), (0, 255, 255), 3)

        # attackers
        for pid, p in attacking_players.items():
            bbox = p.get("bbox")
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                status = offside_status.get(pid, "ON")
                color = (0, 0, 255) if status == "OFF" else (0, 255, 0)
                cv2.circle(vis, (cx, cy), 15, color, -1)
                cv2.putText(
                    vis,
                    f"ID:{pid} {status}",
                    (cx - 40, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # defenders
        for pid, p in defending_players.items():
            bbox = p.get("bbox")
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(
                    vis,
                    f"ID:{pid} DEF",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        return vis
