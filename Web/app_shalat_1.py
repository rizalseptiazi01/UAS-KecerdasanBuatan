import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================================================================
# KONFIGURASI HALAMAN
# ================================================================
st.set_page_config(page_title="Pemonitor Shalat", layout="wide", page_icon="🕋")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background-color: #F4FBF7 !important; }

[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #D6EEE3 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #1A2E24 !important; }

[data-testid="stSelectbox"] > div > div {
    background-color: #EBF8F1 !important;
    border: 1px solid #C4E8D5 !important;
    border-radius: 10px !important;
    color: #1A2E24 !important;
}

div.stButton > button {
    background-color: #FFFFFF !important;
    color: #1A8F55 !important;
    border: 1.5px solid #C4E8D5 !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.2rem !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    width: 100% !important;
}
div.stButton > button:hover {
    background-color: #EBF8F1 !important;
    border-color: #34C47A !important;
    color: #0D5C36 !important;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6EEE3 !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] p {
    color: #7BAA90 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    color: #0D5C36 !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
}

[data-testid="stCheckbox"] label {
    color: #1A2E24 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

h1 { color: #1A2E24 !important; font-size: 1.5rem !important; font-weight: 600 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }

[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
}
[data-testid="stImage"] img {
    border-radius: 12px !important;
    max-height: 62vh !important;
    width: auto !important;
    max-width: 100% !important;
    object-fit: contain !important;
}

hr { border-color: #D6EEE3 !important; margin: 0.8rem 0 !important; }

.sidebar-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7BAA90;
    margin-bottom: 0.35rem;
    margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# AUDIO SYNTHESIZER BEAT (SUARA)
# ================================================================
def play_warning_sound():
    st.components.v1.html("""<script>
    (function(){
        var ctx=new(window.AudioContext||window.webkitAudioContext)();
        function b(f,s,d,v){var o=ctx.createOscillator(),g=ctx.createGain();
        o.connect(g);g.connect(ctx.destination);o.frequency.value=f;o.type='sine';
        g.gain.setValueAtTime(v,ctx.currentTime+s);
        g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+s+d);
        o.start(ctx.currentTime+s);o.stop(ctx.currentTime+s+d+0.05);}
        b(880,0,.18,.6);b(660,.22,.18,.6);b(880,.44,.18,.6);
    })();
    </script>""", height=0)

def play_rakaat_sound():
    st.components.v1.html("""<script>
    (function(){
        var ctx=new(window.AudioContext||window.webkitAudioContext)();
        var o=ctx.createOscillator(),g=ctx.createGain();
        o.connect(g);g.connect(ctx.destination);
        o.frequency.value=523;o.type='sine';
        g.gain.setValueAtTime(0.4,ctx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+0.35);
        o.start(ctx.currentTime);o.stop(ctx.currentTime+0.4);
    })();
    </script>""", height=0)

def play_success_sound():
    st.components.v1.html("""<script>
    (function(){
        var ctx=new(window.AudioContext||window.webkitAudioContext)();
        function b(f,s,d,v){var o=ctx.createOscillator(),g=ctx.createGain();
        o.connect(g);g.connect(ctx.destination);o.frequency.value=f;o.type='sine';
        g.gain.setValueAtTime(v,ctx.currentTime+s);
        g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+s+d);
        o.start(ctx.currentTime+s);o.stop(ctx.currentTime+s+d+0.05);}
        b(523,0,.15,.5);b(659,.15,.15,.5);b(784,.30,.3,.5);
    })();
    </script>""", height=0)

# ================================================================
# LOAD MODEL & DETECTOR
# ================================================================
@st.cache_resource
def load_skeleton_model():
    return load_model("model_skeleton_shalat.h5")

@st.cache_resource
def init_mediapipe_detector():
    import urllib.request, os
    model_path = "pose_landmarker_full.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(options)

try:
    model = load_skeleton_model()
    model_ok = True
except:
    model_ok = False

try:
    detector = init_mediapipe_detector()
except:
    pass

# ================================================================
# INITIALIZE SESSION STATE
# ================================================================
CLASS_NAMES = ['Jalsa', 'Qiyam_Recitation', 'Ruku', 'Salam_Left', 'Salam_Right', 'Sujud', 'Takbir']

if 'initialized' not in st.session_state:
    st.session_state.initialized        = True
    st.session_state.rakaat_count       = 1
    st.session_state.sujud_count        = 0   
    st.session_state.sujud_phase        = 0
    st.session_state.in_sujud           = False
    st.session_state.warning_played     = False
    st.session_state.success_played     = False
    st.session_state.shalat_selesai     = False
    st.session_state.last_rakaat_sound  = 0
    st.session_state.non_sujud_counter  = 0 

# ================================================================
# SIDEBAR UI
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.2rem;">
        <div style="width:40px;height:40px;background:#1A8F55;border-radius:11px;
                    display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">
            🕋
        </div>
        <div>
            <div style="font-size:0.92rem;font-weight:600;color:#1A2E24;line-height:1.2;">Pemonitor Shalat</div>
            <div style="font-size:0.68rem;color:#7BAA90;">Skeleton Pose Tracking</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Jenis Shalat</div>', unsafe_allow_html=True)
    jenis_shalat = st.selectbox(
        label="jenis",
        options=["Subuh — 2 Rakaat", "Dzuhur — 4 Rakaat", "Ashar — 4 Rakaat", "Maghrib — 3 Rakaat", "Isya — 4 Rakaat"],
        label_visibility="collapsed"
    )
    if   "Subuh"   in jenis_shalat: max_rakaat = 2
    elif "Maghrib" in jenis_shalat: max_rakaat = 3
    else:                            max_rakaat = 4

    # TOTAL SUJUD DIHITUNG BERDASARKAN RAKAAT (1 Rakaat = 2 Sujud)
    max_sujud = max_rakaat * 2

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Progress Shalat</div>', unsafe_allow_html=True)

    metric_slot   = st.empty()
    progress_slot = st.empty()

    def render_metrics():
        pct = min(st.session_state.rakaat_count / max_rakaat, 1.0) * 100
        with metric_slot:
            c1, c2 = st.columns(2)
            with c1: st.metric("Rakaat", f"{st.session_state.rakaat_count} / {max_rakaat}")
            with c2: st.metric("Total Sujud",  f"{st.session_state.sujud_count} / {max_sujud}")
        progress_slot.markdown(f"""
        <div style="margin:0.5rem 0 0.8rem;background:#D6EEE3;border-radius:99px;height:6px;">
            <div style="width:{pct:.0f}%;background:#34C47A;height:6px;border-radius:99px;"></div>
        </div>
        <div style="font-size:0.68rem;color:#7BAA90;text-align:right;margin-top:-0.5rem;">{pct:.0f}% selesai</div>
        """, unsafe_allow_html=True)

    render_metrics()

    st.markdown("---")
    if st.button("↺  Reset Shalat"):
        st.session_state.rakaat_count       = 1
        st.session_state.sujud_count        = 0
        st.session_state.sujud_phase        = 0
        st.session_state.in_sujud           = False
        st.session_state.warning_played     = False
        st.session_state.success_played     = False
        st.session_state.shalat_selesai     = False
        st.session_state.last_rakaat_sound  = 0
        st.session_state.non_sujud_counter  = 0
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Status Sistem</div>', unsafe_allow_html=True)
    if model_ok:
        st.success("Model .h5 aktif", icon="✅")
    else:
        st.error("Model gagal dimuat", icon="❌")

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Pose Terdeteksi</div>', unsafe_allow_html=True)
    pose_slot = st.empty()
    pose_slot.markdown("""
    <div style="background:#EBF8F1;border:1px solid #C4E8D5;border-radius:10px;padding:10px 12px;">
        <div style="font-size:0.72rem;color:#7BAA90;">Pose</div>
        <div style="font-size:1rem;font-weight:600;color:#0D5C36;">— menunggu —</div>
        <div style="font-size:0.72rem;color:#7BAA90;margin-top:2px;">Keyakinan: 0.0%</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Fase Sujud Rakaat Ini</div>', unsafe_allow_html=True)
    phase_slot = st.empty()
    phase_slot.markdown("""
    <div style="background:#EBF8F1;border:1px solid #C4E8D5;border-radius:10px;padding:10px 12px;">
        <div style="font-size:1.2rem;letter-spacing:4px;">⬜ ⬜</div>
        <div style="font-size:0.72rem;color:#3D6B52;margin-top:4px;">Menunggu sujud pertama</div>
    </div>""", unsafe_allow_html=True)

# ================================================================
# MAIN KONTEN
# ================================================================
st.markdown("""
<div style="margin-bottom:1rem;">
    <h1 style="margin:0 0 4px;">Pemantau Gerakan Shalat</h1>
    <p style="color:#7BAA90;font-size:0.82rem;margin:0;">
        Deteksi pose otomatis berbasis koordinat skeleton
    </p>
</div>
""", unsafe_allow_html=True)

warning_banner = st.empty()
sound_slot     = st.empty()
run_cam = st.checkbox("📷  Aktifkan Kamera Pemantau", value=False)

ORNAMEN_KIRI = """
<svg width="60" height="430" viewBox="0 0 60 430" xmlns="http://www.w3.org/2000/svg">
  <line x1="30" y1="20" x2="30" y2="410" stroke="#C4E8D5" stroke-width="1.5" stroke-dasharray="4 6"/>
  <path d="M10,10 L50,10 L50,50 M10,10 L10,50" fill="none" stroke="#34C47A" stroke-width="2" stroke-linecap="round"/>
  <circle cx="30" cy="30" r="4" fill="#34C47A" opacity="0.5"/>
  <path d="M10,420 L50,420 L50,380 M10,420 L10,380" fill="none" stroke="#34C47A" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

ORNAMEN_KANAN = """
<svg width="60" height="430" viewBox="0 0 60 430" xmlns="http://www.w3.org/2000/svg">
  <line x1="30" y1="20" x2="30" y2="410" stroke="#C4E8D5" stroke-width="1.5" stroke-dasharray="4 6"/>
  <path d="M50,10 L10,10 L10,50 M50,10 L50,50" fill="none" stroke="#34C47A" stroke-width="2" stroke-linecap="round"/>
  <circle cx="30" cy="30" r="4" fill="#34C47A" opacity="0.5"/>
  <path d="M50,420 L10,420 L10,380 M50,420 L50,380" fill="none" stroke="#34C47A" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

if not run_cam:
    col_l, col_c, col_r = st.columns([1, 8, 1])
    with col_l: st.markdown(ORNAMEN_KIRI, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div style="background:#0D3D22;border-radius:16px;border:1.5px solid #1A5C35;
                    height:430px;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:14px;">
            <div style="width:68px;height:68px;border-radius:50%;
                        border:2px dashed rgba(52,196,122,0.35);
                        display:flex;align-items:center;justify-content:center;font-size:28px;">📷</div>
            <div style="color:rgba(168,232,197,0.75);font-size:0.88rem;font-weight:500;">Kamera belum aktif</div>
            <div style="color:rgba(52,196,122,0.4);font-size:0.73rem;text-align:center;max-width:220px;">
                Aktifkan checkbox di atas untuk mulai memantau gerakan shalat</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r: st.markdown(ORNAMEN_KANAN, unsafe_allow_html=True)

else:
    col_l, col_c, col_r = st.columns([1, 8, 1])
    with col_l: st.markdown(ORNAMEN_KIRI, unsafe_allow_html=True)
    with col_c: FRAME_WINDOW = st.empty()
    with col_r: st.markdown(ORNAMEN_KANAN, unsafe_allow_html=True)

    camera          = cv2.VideoCapture(0)
    blink_state     = True
    last_blink_time = time.time()

    while run_cam:
        ret, frame = camera.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h0, w0 = frame.shape[:2]
        if w0 > 700: frame = cv2.resize(frame, (700, int(h0 * 700 / w0)))

        h, w, _ = frame.shape
        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image         = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)

        current_pose = "Mencari tubuh..."
        confidence   = 0.0

        if detection_result.pose_landmarks:
            row = []
            for lm in detection_result.pose_landmarks[0]:
                v = getattr(lm, 'visibility', 1.0)
                row.extend([lm.x, lm.y, lm.z, v])
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (52, 196, 122), -1)

            preds        = model.predict(np.array([row]), verbose=0)
            idx          = np.argmax(preds[0])
            current_pose = CLASS_NAMES[idx]
            confidence   = preds[0][idx]

            # ---- INTERSEPSI UTAMA: KELEBIHAN RAKAAT PASCA-SALAM ----
            if st.session_state.shalat_selesai and current_pose == 'Qiyam_Recitation' and confidence > 0.65:
                st.session_state.shalat_selesai = False       
                st.session_state.rakaat_count = max_rakaat + 1  
                st.session_state.sujud_count = max_sujud  # Di-set penuh biar sinkron melompati batas
                st.session_state.sujud_phase = 0

            # ---- EVALUASI ENGINE STATE UTAMA ----
            if confidence > 0.80:
                phase = st.session_state.sujud_phase

                # 1. Pengecekan Gerakan Salam (Wajib sudah menyelesaikan Sujud terakhir di rakaat terakhir)
                if st.session_state.rakaat_count == max_rakaat and phase == 4 and current_pose in ['Salam_Right', 'Salam_Left']:
                    st.session_state.shalat_selesai = True

                # 2. Logika Utama Deteksi Sujud
                if current_pose == 'Sujud':
                    st.session_state.non_sujud_counter = 0  
                    
                    if not st.session_state.in_sujud:
                        st.session_state.in_sujud = True
                        if phase == 0:                     
                            # Sujud ke-1 pada rakaat ini (sujud bertambah akumulatif)
                            st.session_state.sujud_count += 1
                            st.session_state.sujud_phase = 1
                        elif phase == 2:                   
                            # Sujud ke-2 pada rakaat ini (sujud bertambah akumulatif)
                            st.session_state.sujud_count += 1
                            st.session_state.sujud_phase = 3
                else:
                    # JIKA DETEKSI BUKAN SUJUD (Jalsa / Qiyam / Ruku dll)
                    st.session_state.in_sujud = False
                    
                    # Cek bangun dari sujud pertama (Phase 1 -> Phase 2)
                    if phase == 1:
                        st.session_state.non_sujud_counter += 1
                        if st.session_state.non_sujud_counter >= 5:
                            st.session_state.sujud_phase = 2
                            st.session_state.non_sujud_counter = 0
                            
                    # Cek bangkit dari sujud kedua (Phase 3 -> Masuk Rakaat Baru / Tahiyyat Akhir)
                    elif phase == 3:
                        st.session_state.non_sujud_counter += 1
                        if st.session_state.non_sujud_counter >= 5:
                            if st.session_state.rakaat_count < max_rakaat:
                                st.session_state.rakaat_count += 1
                                st.session_state.sujud_phase  = 0
                                st.session_state.last_rakaat_sound = time.time()
                            else:
                                # Jika sudah di rakaat terakhir dan selesai sujud ke-2
                                st.session_state.sujud_phase = 4
                            st.session_state.non_sujud_counter = 0

                # 3. INTERSEPSI KEDUA: Jika sudah sujud terakhir, bukannya salam tapi malah langsung berdiri lagi
                if st.session_state.sujud_phase == 4 and current_pose == 'Qiyam_Recitation':
                    st.session_state.rakaat_count = max_rakaat + 1
                    st.session_state.sujud_phase = 0

        # Update real-time sidebar metric & progress bar
        render_metrics()

        pose_slot.markdown(f"""
        <div style="background:#EBF8F1;border:1px solid #C4E8D5;border-radius:10px;padding:10px 12px;">
            <div style="font-size:0.72rem;color:#7BAA90;">Pose</div>
            <div style="font-size:1rem;font-weight:600;color:#0D5C36;">{current_pose}</div>
            <div style="font-size:0.72rem;color:#7BAA90;margin-top:2px;">Keyakinan: {confidence*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        ph = st.session_state.sujud_phase
        phase_info = {
            0: ("⬜ ⬜", "Menunggu sujud pertama rakaat ini"),
            1: ("🟩 ⬜", "Sedang Sujud 1..."),
            2: ("🟩 ⬜", "Duduk di antara dua sujud (Siap Sujud ke-2)"),
            3: ("🟩 🟩", "Sedang Sujud 2..."),
            4: ("🟩 🟩", "Sujud Selesai — Menuju Tahiyyat Akhir & Salam"),
        }
        dots, desc = phase_info.get(ph, ("⬜ ⬜", ""))
        phase_slot.markdown(f"""
        <div style="background:#EBF8F1;border:1px solid #C4E8D5;border-radius:10px;padding:10px 12px;">
            <div style="font-size:1.2rem;letter-spacing:4px;">{dots}</div>
            <div style="font-size:0.72rem;color:#3D6B52;margin-top:4px;">{desc}</div>
        </div>""", unsafe_allow_html=True)

        # ====== LOGIKA BANNER ALERTS & AUDIO EFFECTS ======
        if st.session_state.rakaat_count > max_rakaat:
            warning_banner.error(f"⚠️ Rakaat melebihi {max_rakaat}! Segera akhiri shalat.", icon="🚨")
            if time.time() - last_blink_time > 0.4:
                blink_state = not blink_state
                last_blink_time = time.time()
            if blink_state:
                cv2.rectangle(frame, (0,0), (w,h), (50,50,220), 22)
                cv2.putText(frame, "RAKAAT BERLEBIH!", (int(w*.1), int(h*.1)), cv2.FONT_HERSHEY_DUPLEX, 0.85, (50,80,255), 2, cv2.LINE_AA)
            if not st.session_state.warning_played:
                with sound_slot: play_warning_sound()
                st.session_state.warning_played = True
                
        elif st.session_state.shalat_selesai:
            warning_banner.success("🎉 Alhamdulillah, Shalat Selesai! Semoga ibadah Anda diterima.", icon="✅")
            if not st.session_state.success_played:
                with sound_slot: play_success_sound()
                st.session_state.success_played = True
        else:
            warning_banner.empty()
            st.session_state.warning_played = False

        if st.session_state.last_rakaat_sound > 0 and time.time() - st.session_state.last_rakaat_sound < 0.5:
            with sound_slot: play_rakaat_sound()
            st.session_state.last_rakaat_sound = 0

        # HUD Overlay Rendering
        ov = frame.copy()
        cv2.rectangle(ov, (0, h-58), (w, h), (13,61,34), -1)
        frame = cv2.addWeighted(ov, 0.72, frame, 0.28, 0)
        cv2.putText(frame, f"Pose: {current_pose}", (14, h-36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (168,232,197), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Keyakinan: {confidence*100:.1f}%", (14, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (52,196,122), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Sujud {st.session_state.sujud_count}/{max_sujud}  |  Rakaat {st.session_state.rakaat_count}/{max_rakaat}", (w-285, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (52,196,122), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    camera.release()