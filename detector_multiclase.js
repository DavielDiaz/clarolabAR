let modelo;
const VIDEO_WIDTH = 320;
const VIDEO_HEIGHT = 240;
const INTERVALO_MS = 750;
const CONFIRMACIONES_REQUERIDAS = 3;
const UMBRAL_PROB = 0.95;

let historial = [];
let deteccionConfirmada = false;
let claseDetectada = null;

const clases = ["AtariAsteroids", "GeniusOldMouse", "MagicBrainCalculator"];

async function cargarModelo() {
  modelo = await tf.loadGraphModel("ModeloMulticlase/model.json");
  console.log("✅ Modelo cargado");
}

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
  return new Promise(resolve => {
    videoElement.onloadedmetadata = () => {
      videoElement.play();
      resolve();
    };
  });
}

function capturarFrame(videoElement) {
  const canvas = document.createElement("canvas");
  canvas.width = VIDEO_WIDTH;
  canvas.height = VIDEO_HEIGHT;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoElement, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
  const imageData = ctx.getImageData(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

  return tf.tidy(() => {
    return tf.browser
      .fromPixels(imageData)
      .resizeBilinear([128, 128])
      .expandDims(0)
      .toFloat()
      .div(255.0);
  });
}

function procesarPrediccion(prediccion) {
  const datos = prediccion.dataSync();
  let mejorClase = null;
  let mejorProb = 0;
  let mejorBox = null;

  for (let i = 0; i < clases.length; i++) {
    const prob = datos[i * 5];
    if (prob > mejorProb) {
      mejorProb = prob;
      mejorClase = clases[i];
      mejorBox = datos.slice(i * 5 + 1, i * 5 + 5); // [cx, cy, w, h]
    }
  }

  if (mejorProb >= UMBRAL_PROB) {
    return { clase: mejorClase, box: mejorBox };
  }

  return null;
}

function verificarDeteccion(clase) {
  if (!clase) {
    historial = [];
    deteccionConfirmada = false;
    claseDetectada = null;
    return;
  }

  historial.push(clase);
  if (historial.length > CONFIRMACIONES_REQUERIDAS) {
    historial.shift();
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

function dibujarBoundingBox(canvas, box, etiqueta) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const arContent = document.getElementById("ar-content");
  arContent.style.display = "none";

  if (!box) return;

  const [cx, cy, w, h] = box;
  const renderWidth = canvas.width;
  const renderHeight = canvas.height;

  const x = (cx - w / 2) * renderWidth;
  const y = (cy - h / 2) * renderHeight;
  const width = w * renderWidth;
  const height = h * renderHeight;

  // Fondo oscuro detrás del cuadro
  ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
  ctx.fillRect(x, y, width, height);

  // Cuadro verde
  ctx.strokeStyle = "#00FF00";
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, width, height);

  // Etiqueta
  ctx.fillStyle = "#00FF00";
  ctx.font = "bold 16px Arial";
  ctx.fillText(etiqueta, x + 4, y - 6);

  // Mostrar contenido AR en la caja
  arContent.style.left = `${x}px`;
  arContent.style.top = `${y}px`;
  arContent.style.width = `${width}px`;
  arContent.style.height = `${height}px`;
  arContent.style.display = "block";
}


async function iniciarDeteccion(videoElement, canvasElement) {
  await cargarModelo();
  await iniciarCamara(videoElement);

  setInterval(() => {
    const imagen = capturarFrame(videoElement);
    try {
      const pred = modelo.execute(imagen);
      const resultado = procesarPrediccion(pred);

      if (resultado) {
        verificarDeteccion(resultado.clase);
        dibujarBoundingBox(canvasElement, resultado.box, resultado.clase);
      } else {
        verificarDeteccion(null);
        dibujarBoundingBox(canvasElement, null, "");
      }

      tf.dispose([imagen, pred]);
    } catch (error) {
      console.error("Error en ejecución del modelo:", error);
      tf.dispose(imagen);
    }
  }, INTERVALO_MS);
}

// Exporta el estado y función de inicio para usarse desde el HTML
window.estadoDeteccion = () => ({
  confirmado: deteccionConfirmada,
  clase: claseDetectada
});

window.iniciarEscaneo = iniciarDeteccion;
