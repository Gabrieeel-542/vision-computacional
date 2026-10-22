import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Obtener distancia entre dos puntos de la mano
def distancia(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def analizar_gesto(lm):
    # Puntas e índices de articulaciones de los dedos
    dedos = [
        (8, 6),    # Índice
        (12, 10),  # Medio
        (16, 14),  # Anular
        (20, 18)   # Meñique
    ]

    # Un dedo está levantado si su punta está por
    # encima de la articulación intermedia.
    levantados = sum(
        1 for punta, articulacion in dedos
        if lm[punta].y < lm[articulacion].y
    )

    pulgar_arriba = lm[4].y < lm[3].y - 0.04

    # Detectar si el pulgar está extendido lateralmente.
    # Se compara su punta y su articulación con la base
    # del dedo índice.
    pulgar_extend = (
        distancia(lm[4], lm[5])
        > distancia(lm[3], lm[5]) * 1.25
    )

    # Reconocer LIKE: pulgar arriba y los otros dedos
    # recogidos.
    if pulgar_arriba and levantados == 0:
        return "LIKE 👍", 0

    # Contar los dedos, incluyendo el pulgar.
    total = levantados + int(pulgar_extend)

    if total == 0:
        return "MANO CERRADA ✊", 0

    return f"DEDOS: {total}", total


# Abrir la cámara web
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    raise RuntimeError(
        "No se pudo abrir la cámara. "
        "Comprueba que esté conectada y disponible."
    )

print("Reconocimiento de gestos iniciado.")
print("Presiona Q para salir.")

try:
    while True:
        correcto, frame = camara.read()

        if not correcto:
            print("No se pudo leer el fotograma.")
            break

        # Efecto espejo para que sea más natural
        frame = cv2.flip(frame, 1)

        # MediaPipe requiere una imagen RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        texto = "MUESTRA TU MANO"
        color = (255, 255, 255)

        if resultado.multi_hand_landmarks:
            mano = resultado.multi_hand_landmarks[0]

            # Dibujar puntos y conexiones de la mano
            mp_drawing.draw_landmarks(
                frame,
                mano,
                mp_hands.HAND_CONNECTIONS
            )

            texto, cantidad = analizar_gesto(
                mano.landmark
            )

            if texto.startswith("LIKE"):
                color = (0, 255, 0)
            elif cantidad == 0:
                color = (0, 0, 255)
            else:
                color = (255, 200, 0)

        # Panel visual
        cv2.rectangle(frame, (10, 10), (460, 100),
                      (30, 30, 30), -1)

        cv2.putText(
            frame, texto, (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, color, 2, cv2.LINE_AA
        )

        cv2.putText(
            frame, "Q = SALIR", (25, 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (220, 220, 220), 1
        )

        cv2.imshow("AI - Reconocimiento de Gestos", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camara.release()
    hands.close()
    cv2.destroyAllWindows()
