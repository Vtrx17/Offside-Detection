from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from offside_detector import OffsideDetector  # vertical-line version


# <<< SET THESE ONCE FOR YOUR VIDEO >>>
ATTACKING_TEAM_ID = 1  # team that should be attackers
DEFENDING_TEAM_ID = 2  # team that should be defenders
# If they are still swapped in the video, just swap the numbers above.


def main():
    # Read Video
    video_frames = read_video("input_videos/Football.mp4")
    if not video_frames:
        raise ValueError("No frames loaded from video. Check input_videos/Football.mp4")

    frame_height, frame_width = video_frames[0].shape[:2]

    # Initialize Tracker
    tracker = Tracker(
        r"C:\Users\ROG\Desktop\VTRX\Academics_IIUM\sem7_2026\MV\VSCODE\football_analysis-main\football_analysis-main\models\best.pt"
    )

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=(
            r"C:\Users\ROG\Desktop\VTRX\Academics_IIUM\sem7_2026\MV\VSCODE"
            r"\football_analysis-main\football_analysis-main\stubs\track_stubs.pkl"
        ),
    )
    tracker.add_position_to_tracks(tracks)

    # Camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=(
            r"C:\Users\ROG\Desktop\VTRX\Academics_IIUM\sem7_2026\MV\VSCODE"
            r"\football_analysis-main\football_analysis-main\stubs\camera_movement_stub.pkl"
        ),
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed & distance
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Team assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    # Ball acquisition & team ball control (kept for stats, but not used to decide roles)
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        if frame_num < len(tracks["ball"]) and 1 in tracks["ball"][frame_num]:
            ball_bbox = tracks["ball"][frame_num][1]["bbox"]
            assigned_player = player_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )
            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(
                    tracks["players"][frame_num][assigned_player]["team"]
                )
            else:
                team_ball_control.append(
                    team_ball_control[-1] if team_ball_control else 0
                )
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    # Offside detector (vertical line)
    offside_detector = OffsideDetector(frame_width, frame_height)

    # Draw base annotations
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # ---------- OFFSIDE PER FRAME ----------
    for frame_num in range(len(output_video_frames)):
        if frame_num >= len(tracks["players"]):
            continue

        player_track = tracks["players"][frame_num]

        # Ball position (image coords)
        ball_pos_img = None
        if frame_num < len(tracks["ball"]) and 1 in tracks["ball"][frame_num]:
            ball_dict = tracks["ball"][frame_num][1]
            if "position_img" in ball_dict:
                ball_pos_img = ball_dict["position_img"]
            elif "bbox" in ball_dict:
                x1, y1, x2, y2 = ball_dict["bbox"]
                ball_pos_img = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        # FIXED attacker/defender mapping by team ID
        attacking_players = {}
        defending_players = {}
        for pid, player in player_track.items():
            team_id = player.get("team", 0)
            if team_id == ATTACKING_TEAM_ID:
                attacking_players[pid] = player
            elif team_id == DEFENDING_TEAM_ID:
                defending_players[pid] = player

        if not attacking_players or not defending_players:
            continue

        # Attack direction along X (left -> right).
        # If the attacking team actually goes right -> left, change to "left".
        attacking_direction = "right"

        offside_status = offside_detector.check_offside_event(
            attacking_players,
            defending_players,
            ball_pos_img,
            frame_num,
            attacking_direction,
        )

        output_video_frames[frame_num] = offside_detector.draw_offside_visualization(
            output_video_frames[frame_num],
            attacking_players,
            defending_players,
            offside_status,
            ball_pos_img,
            attacking_direction,
        )
    # ---------------------------------------

    save_video(output_video_frames, "output_videos/output_video_offside.avi")


if __name__ == "__main__":
    main()
