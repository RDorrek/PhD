import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# MediaPipe Pose
mp_pose = mp.solutions.pose

# -----------------------------------------------
# Datei-Auswahl (öffnet Recording-Ordner)
# -----------------------------------------------
def choose_file():
    root = Tk()
    root.withdraw()

    bag_path = askopenfilename(
        title="Wähle eine RealSense .bag-Datei",
        initialdir=r"C:\Users\rodorrek\Seafile\Lehre_Forschung\Research_Unanticipated_Movements\Code\Recordings",
        filetypes=[("RealSense Bag Files", "*.bag"), ("Alle Dateien", "*.*")]
    )

    root.destroy()
    return bag_path


# -----------------------------------------------
# Hilfsfunktionen für Winkel & Zeichnen
# -----------------------------------------------
def compute_angle(a, b, c):
    """
    Winkel an Punkt b (in Grad) zwischen Punkten a-b-c.
    a, b, c sind Arrays [x, y].
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return None
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def draw_text_at_joint(img, coord, text):
    x, y = int(coord[0]), int(coord[1])
    cv2.putText(
        img,
        text,
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def draw_skeleton(img, coords, vis, min_vis=0.5):
    """Einfaches Skelett mit Linien zwischen MediaPipe-Verbindungen."""
    if coords is None or vis is None:
        return
    for i, j in mp_pose.POSE_CONNECTIONS:
        if vis[i] > min_vis and vis[j] > min_vis:
            p1 = tuple(coords[i].astype(int))
            p2 = tuple(coords[j].astype(int))
            cv2.line(img, p1, p2, (0, 255, 0), 2)

    for idx in range(coords.shape[0]):
        if vis[idx] > min_vis:
            p = tuple(coords[idx].astype(int))
            cv2.circle(img, p, 3, (0, 255, 255), -1)


def draw_angles(img, coords, vis, flags, min_vis=0.5):
    """
    Zeichnet Winkel entsprechend der Flags.
    flags: dict mit Booleans, z.B. flags['elbow_right']
    """
    if coords is None or vis is None:
        return

    # Indizes laut MediaPipe-Doku :contentReference[oaicite:0]{index=0}
    L_SH = 11
    R_SH = 12
    L_EL = 13
    R_EL = 14
    L_WR = 15
    R_WR = 16
    L_HIP = 23
    R_HIP = 24
    L_KNEE = 25
    R_KNEE = 26
    L_ANK = 27
    R_ANK = 28
    L_FI = 31   # left foot index
    R_FI = 32   # right foot index

    def ok(idx):
        return vis[idx] > min_vis

    # --- Ellenbogen ---
    if flags["elbow_right"] and ok(R_SH) and ok(R_EL) and ok(R_WR):
        ang = compute_angle(coords[R_SH], coords[R_EL], coords[R_WR])
        if ang is not None:
            draw_text_at_joint(img, coords[R_EL], f"R Elb {ang:.1f}")

    if flags["elbow_left"] and ok(L_SH) and ok(L_EL) and ok(L_WR):
        ang = compute_angle(coords[L_SH], coords[L_EL], coords[L_WR])
        if ang is not None:
            draw_text_at_joint(img, coords[L_EL], f"L Elb {ang:.1f}")

    # --- Schulter (zwischen Oberkörper und Oberarm) ---
    if flags["shoulder_right"] and ok(R_HIP) and ok(R_SH) and ok(R_EL):
        ang = compute_angle(coords[R_HIP], coords[R_SH], coords[R_EL])
        if ang is not None:
            draw_text_at_joint(img, coords[R_SH], f"R Sh {ang:.1f}")

    if flags["shoulder_left"] and ok(L_HIP) and ok(L_SH) and ok(L_EL):
        ang = compute_angle(coords[L_HIP], coords[L_SH], coords[L_EL])
        if ang is not None:
            draw_text_at_joint(img, coords[L_SH], f"L Sh {ang:.1f}")

    # --- Hüfte (zwischen Oberkörper und Oberschenkel) ---
    if flags["hip_right"] and ok(R_SH) and ok(R_HIP) and ok(R_KNEE):
        ang = compute_angle(coords[R_SH], coords[R_HIP], coords[R_KNEE])
        if ang is not None:
            draw_text_at_joint(img, coords[R_HIP], f"R Hip {ang:.1f}")

    if flags["hip_left"] and ok(L_SH) and ok(L_HIP) and ok(L_KNEE):
        ang = compute_angle(coords[L_SH], coords[L_HIP], coords[L_KNEE])
        if ang is not None:
            draw_text_at_joint(img, coords[L_HIP], f"L Hip {ang:.1f}")

    # --- Knie ---
    if flags["knee_right"] and ok(R_HIP) and ok(R_KNEE) and ok(R_ANK):
        ang = compute_angle(coords[R_HIP], coords[R_KNEE], coords[R_ANK])
        if ang is not None:
            draw_text_at_joint(img, coords[R_KNEE], f"R Knee {ang:.1f}")

    if flags["knee_left"] and ok(L_HIP) and ok(L_KNEE) and ok(L_ANK):
        ang = compute_angle(coords[L_HIP], coords[L_KNEE], coords[L_ANK])
        if ang is not None:
            draw_text_at_joint(img, coords[L_KNEE], f"L Knee {ang:.1f}")

    # --- Sprunggelenk (z.B. Knie-Ankle-FootIndex) ---
    if flags["ankle_right"] and ok(R_KNEE) and ok(R_ANK) and ok(R_FI):
        ang = compute_angle(coords[R_KNEE], coords[R_ANK], coords[R_FI])
        if ang is not None:
            draw_text_at_joint(img, coords[R_ANK], f"R Ank {ang:.1f}")

    if flags["ankle_left"] and ok(L_KNEE) and ok(L_ANK) and ok(L_FI):
        ang = compute_angle(coords[L_KNEE], coords[L_ANK], coords[L_FI])
        if ang is not None:
            draw_text_at_joint(img, coords[L_ANK], f"L Ank {ang:.1f}")


# -----------------------------------------------
# Eine Bag-Datei laden + Slider-Viewer
# -----------------------------------------------
def process_bag_file(bag_path):

    mp_drawing = mp.solutions.drawing_utils  # nur falls du später noch brauchst

    pipeline = rs.pipeline()
    config = rs.config()

    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
    config.enable_all_streams()
    pipeline_profile = pipeline.start(config)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_color = []
    landmarks_list = []  # pro Frame: (coords (33x2), visibility (33,))

    print("\nLese Frames aus der .bag und berechne Pose...")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                print("Ende der Datei.")
                break

            color_frame = frames.get_color_frame()
            if not color_frame:
                break

            img_bgr = np.asanyarray(color_frame.get_data())
            h, w, _ = img_bgr.shape

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                coords = np.zeros((33, 2), dtype=np.float32)
                vis = np.zeros(33, dtype=np.float32)
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    coords[i, 0] = lm.x * w
                    coords[i, 1] = lm.y * h
                    vis[i] = lm.visibility
                landmarks_list.append((coords, vis))
            else:
                landmarks_list.append((None, None))

            frames_color.append(img_bgr)

    finally:
        pipeline.stop()
        pose.close()

    print(f"Frames geladen: {len(frames_color)}")

    if len(frames_color) == 0:
        print("Keine Frames – zurück zum Auswahlfenster.")
        return True

    # ----------------------------------
    # Interaktiver Viewer
    # ----------------------------------
    window_name = "Pose Viewer"
    cv2.namedWindow(window_name)

    current_idx = 0

    # Toggle-Flags
    show_skeleton = True
    show_flags = {
        "elbow_right": False,
        "elbow_left": False,
        "shoulder_right": False,
        "shoulder_left": False,
        "hip_right": False,
        "hip_left": False,
        "knee_right": False,
        "knee_left": False,
        "ankle_right": False,
        "ankle_left": False,
    }

    def show_frame(idx):
        idx = max(0, min(idx, len(frames_color) - 1))
        img = frames_color[idx].copy()
        coords, vis = landmarks_list[idx]

        # Skelett
        if show_skeleton and coords is not None:
            draw_skeleton(img, coords, vis)

        # Winkel
        draw_angles(img, coords, vis, show_flags)

        cv2.imshow(window_name, img)

    def on_trackbar(val):
        nonlocal current_idx
        current_idx = val
        show_frame(current_idx)

    cv2.createTrackbar(
        "Frame",
        window_name,
        0,
        len(frames_color) - 1,
        on_trackbar
    )

    show_frame(0)

    print("\nSteuerung:")
    print("  a / d   = vorheriger / nächster Frame")
    print("  h       = Skelett ein/aus")
    print("  e / E   = Ellenbogen rechts / links")
    print("  r / R   = Schulter rechts / links")
    print("  t / T   = Hüfte rechts / links")
    print("  z / Z   = Knie rechts / links")
    print("  u / U   = Sprunggelenk rechts / links")
    print("  w       = Recording wechseln")
    print("  q oder Fenster schließen = Script beenden\n")

    # ----------------------------------
    # Viewer Loop
    # ----------------------------------
    while True:
        key = cv2.waitKey(50) & 0xFF

        # Fenster manuell geschlossen -> Script beenden
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return False

        if key == ord('q'):
            cv2.destroyAllWindows()
            return False

        elif key == ord('d'):
            if current_idx < len(frames_color) - 1:
                current_idx += 1
                cv2.setTrackbarPos("Frame", window_name, current_idx)
                show_frame(current_idx)

        elif key == ord('a'):
            if current_idx > 0:
                current_idx -= 1
                cv2.setTrackbarPos("Frame", window_name, current_idx)
                show_frame(current_idx)

        # Recording wechseln
        elif key == ord('w'):
            cv2.destroyAllWindows()
            return True

        # Skelett an/aus
        elif key == ord('h'):
            show_skeleton = not show_skeleton
            show_frame(current_idx)

        # Winkel-Toggles: rechts = klein, links = groß
        elif key == ord('e'):   # right elbow
            show_flags["elbow_right"] = not show_flags["elbow_right"]
            show_frame(current_idx)
        elif key == ord('E'):   # left elbow
            show_flags["elbow_left"] = not show_flags["elbow_left"]
            show_frame(current_idx)

        elif key == ord('r'):
            show_flags["shoulder_right"] = not show_flags["shoulder_right"]
            show_frame(current_idx)
        elif key == ord('R'):
            show_flags["shoulder_left"] = not show_flags["shoulder_left"]
            show_frame(current_idx)

        elif key == ord('t'):
            show_flags["hip_right"] = not show_flags["hip_right"]
            show_frame(current_idx)
        elif key == ord('T'):
            show_flags["hip_left"] = not show_flags["hip_left"]
            show_frame(current_idx)

        elif key == ord('z'):
            show_flags["knee_right"] = not show_flags["knee_right"]
            show_frame(current_idx)
        elif key == ord('Z'):
            show_flags["knee_left"] = not show_flags["knee_left"]
            show_frame(current_idx)

        elif key == ord('u'):
            show_flags["ankle_right"] = not show_flags["ankle_right"]
            show_frame(current_idx)
        elif key == ord('U'):
            show_flags["ankle_left"] = not show_flags["ankle_left"]
            show_frame(current_idx)


# -----------------------------------------------
# MAIN LOOP: Recording wählen / wechseln
# -----------------------------------------------
while True:
    bag_file = choose_file()
    if not bag_file:
        print("Keine Datei ausgewählt. Script beendet.")
        break

    keep_running = process_bag_file(bag_file)

    if not keep_running:
        print("Script beendet.")
        break

print("Bye!")
