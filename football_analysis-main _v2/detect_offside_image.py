from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import argparse
import sys


# ====================================================
# CLASS: Perspective Transformer (Auto-Sort)
# ====================================================
class ViewTransformer:
    def __init__(self, source_pts, target_width=680, target_height=1050):
        self.source_pts = self.order_points(source_pts)
        self.target_width = target_width
        self.target_height = target_height

        target_pts = np.float32(
            [
                [0, 0],
                [target_width, 0],
                [target_width, target_height],
                [0, target_height],
            ]
        )

        self.M = cv2.getPerspectiveTransform(self.source_pts, target_pts)
        self.M_inv = cv2.getPerspectiveTransform(target_pts, self.source_pts)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        return rect

    def transform_points(self, points):
        if len(points) == 0:
            return []
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.M)
        return transformed.reshape(-1, 2)

    def inverse_transform_points(self, points):
        if len(points) == 0:
            return []
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.M_inv)
        return transformed.reshape(-1, 2)


# ====================================================
# UI HELPERS
# ====================================================
def select_perspective_points(image):
    print("\n[STEP 1] Click 4 corners of a rectangle on the field (e.g. mowing lines).")
    points = []
    display_img = image.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(
                    display_img, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2
                )
            if len(points) == 4:
                cv2.line(
                    display_img, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2
                )
            cv2.imshow("Step 1: Select Perspective", display_img)

    cv2.imshow("Step 1: Select Perspective", display_img)
    cv2.setMouseCallback("Step 1: Select Perspective", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or len(points) == 4:
            cv2.waitKey(500)
            break
    cv2.destroyWindow("Step 1: Select Perspective")

    if len(points) != 4:
        h, w = image.shape[:2]
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return np.float32(points)


def select_defending_team(image, players_dict):
    """Asks user to click on ONE defending player to identify the team."""
    print("\n[STEP 2] Click on ANY PLAYER from the DEFENDING TEAM.")
    display_img = image.copy()
    selected_team_id = -1

    # Draw boxes to help user click
    for pid, data in players_dict.items():
        bbox = data["bbox"]
        cv2.rectangle(
            display_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (255, 255, 0),
            2,
        )

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_team_id
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check which box was clicked
            for pid, data in players_dict.items():
                bx = data["bbox"]
                if bx[0] < x < bx[2] and bx[1] < y < bx[3]:
                    selected_team_id = data.get(
                        "team_id_internal", -1
                    )  # Use internal team ID
                    print(f"✓ Selected Player {pid}. Team ID: {selected_team_id}")
                    cv2.destroyWindow("Step 2: Click Defending Player")
                    return

    cv2.imshow("Step 2: Click Defending Player", display_img)
    cv2.setMouseCallback("Step 2: Click Defending Player", mouse_callback)

    while selected_team_id == -1:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if (
        cv2.getWindowProperty("Step 2: Click Defending Player", cv2.WND_PROP_VISIBLE)
        >= 1
    ):
        cv2.destroyWindow("Step 2: Click Defending Player")

    return selected_team_id


# ====================================================
# MAIN LOGIC
# ====================================================
def detect_offside_image(image_path, output_path):
    # 1. Load Image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error reading {image_path}")
        return

    # 2. Perspective Setup
    src_points = select_perspective_points(frame)
    view_transformer = ViewTransformer(src_points)

    # 3. Detect Players
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks([frame], read_from_stub=False)
    tracker.add_position_to_tracks(tracks)
    if not tracks.get("players") or not tracks["players"][0]:
        print("No players.")
        return

    # 4. Assign Colors (needed for internal team logic)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame, tracks["players"][0])

    # Store internal team IDs in the tracks dict for the UI helper
    for pid, data in tracks["players"][0].items():
        tid = team_assigner.get_player_team(frame, data["bbox"], pid)
        tracks["players"][0][pid]["team_id_internal"] = tid

    # 5. USER INPUT: Identify Defending Team
    defending_team_id = select_defending_team(frame, tracks["players"][0])
    if defending_team_id == -1:
        print("Selection cancelled.")
        return
    print(f"✓ Defending Team ID set to: {defending_team_id}")

    # 6. Process Players & Transform
    players = []
    for pid, data in tracks["players"][0].items():
        bbox = data["bbox"]
        foot_pos = np.array([[(bbox[0] + bbox[2]) / 2, bbox[3]]])
        field_pos = view_transformer.transform_points(foot_pos)[0]

        # Determine team based on ID match
        # Note: If player was GK and color was diff, this might be tricky.
        # But we will FIX GK assignment below.
        p_team = data.get("team_id_internal")

        players.append(
            {
                "id": pid,
                "team": p_team,
                "bbox": bbox,
                "img_pos": foot_pos[0],
                "field_x": field_pos[0],
            }
        )

    # 7. Determine Direction (Defenders vs Attackers X Position)
    defenders = [p for p in players if p["team"] == defending_team_id]
    attackers = [p for p in players if p["team"] != defending_team_id]

    if not defenders or not attackers:
        print("Error: Teams not separated.")
        return

    def_avg_x = np.median([p["field_x"] for p in defenders])
    att_avg_x = np.median([p["field_x"] for p in attackers])

    # If Defenders are to the Right of Attackers, Goal is RIGHT.
    goal_is_right = def_avg_x > att_avg_x
    print(f"✓ Goal Direction: {'RIGHT (>>>)' if goal_is_right else 'LEFT (<<<)'}")

    # 8. FIX: Force Goalkeeper to Defending Team
    # Find player closest to the goal edge
    if goal_is_right:
        gk_player = max(players, key=lambda p: p["field_x"])
    else:
        gk_player = min(players, key=lambda p: p["field_x"])

    # Force GK to defending team
    if gk_player["team"] != defending_team_id:
        print(f"✓ Fixing Goalkeeper (ID {gk_player['id']}) to Defending Team.")
        gk_player["team"] = defending_team_id

    # 9. Find Offside Line (2nd Last Defender)
    # Refresh defenders list
    defenders = [p for p in players if p["team"] == defending_team_id]

    if len(defenders) < 2:
        offside_line_x = defenders[0]["field_x"]
        offside_player = defenders[0]
    else:
        # Sort: Deepest first
        defenders_sorted = sorted(
            defenders, key=lambda x: x["field_x"], reverse=goal_is_right
        )
        offside_player = defenders_sorted[1]  # 2nd last
        offside_line_x = offside_player["field_x"]

    # 10. Ball Line Rule
    ball_bbox = None
    if tracks.get("ball") and tracks["ball"][0]:
        for k, v in tracks["ball"][0].items():
            if "bbox" in v:
                ball_bbox = v["bbox"]
                break

    if ball_bbox:
        bx, by = (ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2
        ball_x = view_transformer.transform_points(np.array([[bx, by]]))[0][0]

        # If ball is closer to goal, move line
        if goal_is_right:
            if ball_x > offside_line_x:
                offside_line_x = ball_x
        else:
            if ball_x < offside_line_x:
                offside_line_x = ball_x

    # 11. Draw Results
    annotated_frame = frame.copy()

    # Line
    line_pts_field = np.array(
        [[offside_line_x, 0], [offside_line_x, view_transformer.target_height]]
    )
    line_pts_img = view_transformer.inverse_transform_points(line_pts_field)
    pt1, pt2 = line_pts_img[0].flatten(), line_pts_img[1].flatten()
    vec = pt2 - pt1
    pt1_ext, pt2_ext = pt1 - vec * 0.5, pt2 + vec * 0.5
    cv2.line(
        annotated_frame,
        (int(pt1_ext[0]), int(pt1_ext[1])),
        (int(pt2_ext[0]), int(pt2_ext[1])),
        (0, 255, 255),
        3,
    )

    if offside_player:
        cv2.putText(
            annotated_frame,
            "OFFSIDE LINE",
            (int(offside_player["bbox"][0]), int(offside_player["bbox"][1]) - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # Attackers Check
    offside_count = 0
    attackers = [p for p in players if p["team"] != defending_team_id]
    for p in attackers:
        is_offside = False
        if goal_is_right:
            if p["field_x"] > offside_line_x:
                is_offside = True
        else:
            if p["field_x"] < offside_line_x:
                is_offside = True

        if is_offside:
            offside_count += 1
            cx, cy = int(p["img_pos"][0]), int(p["img_pos"][1])
            cv2.circle(annotated_frame, (cx, cy), 15, (0, 0, 255), 3)
            cv2.putText(
                annotated_frame,
                "OFFSIDE",
                (cx - 30, cy - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    cv2.imwrite(output_path, annotated_frame)
    print(f"✓ Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=r"C:\Users\ROG\Desktop\VTRX\Academics_IIUM\sem7_2026\MV\VSCODE\football_analysis-main\football_analysis-main\input_videos\offside_img4.png",
    )
    parser.add_argument("--output", type=str, default="offside4.jpg")
    args = parser.parse_args()
    detect_offside_image(args.input, args.output)
