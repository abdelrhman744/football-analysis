"""
Team Classification Service
Classifies detected players into teams based on kit colors.

TODO: Replace placeholder with a real team classification model.
      Approaches:
      - K-Means clustering on dominant jersey colors (simple, no training needed)
      - CNN-based kit classifier trained on football jersey images
      - HSV color histogram comparison

Example color-based clustering:
    from sklearn.cluster import KMeans
    import cv2

    def get_dominant_color(player_crop):
        pixels = player_crop.reshape(-1, 3).astype(float)
        kmeans = KMeans(n_clusters=3).fit(pixels)
        dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        return dominant
"""

import random
from models.schemas import (
    TeamClassificationResult,
    TeamPlayer,
    AnalysisStatus,
)


def classify_teams(video_id: str, detection_result) -> TeamClassificationResult:
    """
    Classify detected players into two teams based on jersey appearance.

    TODO: Replace this function body with your actual classification model.

    Steps for real implementation:
    1. Crop each detected player bounding box from video frames
    2. Extract dominant color(s) from the torso region of each crop
    3. Cluster players into 2 groups using K-Means on color features
    4. Label groups as Team A / Team B
    5. Handle referees as a separate class (black/yellow kit usually)
    6. Optionally use a pre-trained CNN for higher accuracy

    Args:
        video_id: Unique identifier for the video
        detection_result: Output from cv_detection_service.detect_players_and_ball()

    Returns:
        TeamClassificationResult with per-player team assignments
    """

    # ── PLACEHOLDER: Assign fake team labels ──────────────────────────────────
    num_players = 22
    team_a_players = []
    team_b_players = []

    colors_a = ["#FFFFFF", "#F0F0FF", "#E8E8E8"]  # White kit placeholder
    colors_b = ["#003366", "#002244", "#001133"]  # Dark blue kit placeholder

    for pid in range(num_players):
        if pid < 11:
            team_a_players.append(
                TeamPlayer(
                    player_id=pid,
                    team_label="Team A",
                    confidence=round(random.uniform(0.80, 0.98), 3),
                    dominant_color=random.choice(colors_a),
                )
            )
        else:
            team_b_players.append(
                TeamPlayer(
                    player_id=pid,
                    team_label="Team B",
                    confidence=round(random.uniform(0.80, 0.98), 3),
                    dominant_color=random.choice(colors_b),
                )
            )
    # ── END PLACEHOLDER ───────────────────────────────────────────────────────

    return TeamClassificationResult(
        video_id=video_id,
        team_a_label="Team A",
        team_b_label="Team B",
        team_a_players=team_a_players,
        team_b_players=team_b_players,
        team_a_color="#FFFFFF",
        team_b_color="#003366",
        status=AnalysisStatus.COMPLETED,
    )
