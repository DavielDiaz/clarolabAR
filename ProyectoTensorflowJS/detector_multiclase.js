let modelo;

// Resolución fija para capturar el video (rendimiento)
const VIDEO_WIDTH = 320;
const VIDEO_HEIGHT = 240;

// Intervalo entre inferencias (milisegundos)
const INTERVALO_MS = 750;

// Cantidad de detecciones iguales necesarias para confirmar
const CONFIRMACIONES_REQUERIDAS = 3;

// Probabilidad mínima para aceptar una clase como válida
const UMBRAL_PROB = 0.95;

// Historial de clases detectadas recientemente
let historial = [];
let deteccionConfirmada = false;
let claseDetectada = null;

// Clases que puede detectar el modelo
const clases = ["AtariAsteroids", "GeniusOldMouse", "MagicBrainCalculator"];

// Carga el modelo entrenado (formato GraphModel)
async function cargarModelo() {
  modelo = await tf.loadGraphModel("ModeloMulticlase/model.json");
  console.log("✅ Modelo cargado");
}

// Inicia la cámara del usuario (trasera en móviles si es posible)
async function iniciarCamara(videoElement) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "environment",
      width: VIDEO_WIDTH,
      height: VIDEO_HEIGHT
    },
    audio: false
  });

  videoElement.srcObject = stream;

  // Espera a que el video esté listo
  return new Promise(resolve => {
    videoElement.onloadedmetadata = () => {
      videoElement.play();
      resolve();
    };
  });
}

// Captura un frame del video y lo convierte a tensor 4D (1, 128, 128, 3)
function capturarFrame(videoElement) {
  const canvas = document.createElement("canvas");
  canvas.width = VIDEO_WIDTH;
  canvas.height = VIDEO_HEIGHT;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoElement, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
  const imageData = ctx.getImageData(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

  // Crea tensor normalizado en rango [0, 1]
  return tf.tidy(() => {
    return tf.browser
      .fromPixels(imageData)   // Convierte imagen a tensor
      .resizeBilinear([128, 128]) // Redimensiona para que coincida con la entrada del modelo
      .expandDims(0)           // Añade dimensión batch
      .toFloat()
      .div(255.0);             // Normaliza
  });
}

// Procesa la salida del modelo para determinar clase y bounding box más confiable
function procesarPrediccion(prediccion) {
  const datos = prediccion.dataSync();  // Extrae datos del tensor
  let mejorClase = null;
  let mejorProb = 0;
  let mejorBox = null;

  for (let i = 0; i < clases.length; i++) {
    const prob = datos[i * 5]; // Probabilidad de presencia
    if (prob > mejorProb) {
      mejorProb = prob;
      mejorClase = clases[i];
      mejorBox = datos.slice(i * 5 + 1, i * 5 + 5); // Coordenadas [cx, cy, w, h]
    }
  }

  if (mejorProb >= UMBRAL_PROB) {
    return { clase: mejorClase, box: mejorBox };
  }

  return null;
}

// Verifica si una clase ha sido detectada varias veces seguidas (evita falsos positivos)
function verificarDeteccion(clase) {
  if (!clase) {
    historial = [];
    deteccionConfirmada = false;
    claseDetectada = null;
    return;
  }

  historial.push(clase);
  if (historial.length > CONFIRMACIONES_REQUERIDAS) {
    historial.shift(); // Mantiene solo las últimas detecciones
  }

  const todasIguales = historial.every(c => c === clase);
  if (todasIguales && historial.length === CONFIRMACIONES_REQUERIDAS) {
    deteccionConfirmada = true;
    claseDetectada = clase;
    console.log("✅ Objeto confirmado:", clase);
  } else {
    deteccionConfirmada = false;
    claseDetectada = null;
  }
}

// Dibuja la caja sobre el canvas y posiciona el contenido AR
function dibujarBoundingBox(canvas, box, etiqueta) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const arContent = document.getElementById("ar-content");
  arContent.style.display = "none";

  if (!box) return;

  const [cx, cy, w, h] = box;

  // Conversión de coordenadas normalizadas (0-1) a píxeles
  const renderWidth = canvas.width;
  const renderHeight = canvas.height;

  const x = (cx - w / 2) * renderWidth;
  const y = (cy - h / 2) * renderHeight;
  const width = w * renderWidth;
  const height = h * renderHeight;

  // Fondo transparente negro
  ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
  ctx.fillRect(x, y, width, height);

  // Cuadro delimitador
  ctx.strokeStyle = "#00FF00";
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, width, height);

  // Etiqueta de clase
  ctx.fillStyle = "#00FF00";
  ctx.font = "bold 16px Arial";
  ctx.fillText(etiqueta, x + 4, y - 6);

  // Posiciona imagen de contenido AR en la caja detectada
  arContent.style.left = `${x}px`;
  arContent.style.top = `${y}px`;
  arContent.style.width = `${width}px`;
  arContent.style.height = `${height}px`;
  arContent.style.display = "block";
}

// Función principal que corre periódicamente: captura, predice, y procesa
async function iniciarDeteccion(videoElement, canvasElement) {
  await cargarModelo();
  await iniciarCamara(videoElement);

  setInterval(() => {
    const imagen = capturarFrame(videoElement);
    try {
      const pred = modelo.execute(imagen);  // Ejecuta modelo sobre imagen
      const resultado = procesarPrediccion(pred);

      if (resultado) {
        verificarDeteccion(resultado.clase);
        dibujarBoundingBox(canvasElement, resultado.box, resultado.clase);
      } else {
        verificarDeteccion(null);
        dibujarBoundingBox(canvasElement, null, "");
      }

      tf.dispose([imagen, pred]); // Libera memoria
    } catch (error) {
      console.error("Error en ejecución del modelo:", error);
      tf.dispose(imagen);
    }
  }, INTERVALO_MS);
}

// Exporta funciones para ser usadas desde el HTML
window.estadoDeteccion = () => ({
  confirmado: deteccionConfirmada,
  clase: claseDetectada
});

window.iniciarEscaneo = iniciarDeteccion;
