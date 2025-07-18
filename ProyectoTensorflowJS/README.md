# TensorFlow.js + AR — Clasificación de Objetos

Este proyecto es una prueba para una aplicación web que utiliza TensorFlow.js para detectar objetos reales desde la cámara,
clasificarlos y, en el futuro, mostrar contenido de realidad aumentada (AR) relacionado.

Actualmente, el objetivo cumplido es detectar y clasificar objetos en tiempo real usando un modelo entrenado desde cero.

## Qué hace este proyecto

- Entrena un modelo CNN en Python con TensorFlow.
- Clasifica imágenes en tres clases distintas más una clase negativa.
- Exporta el modelo a TensorFlow.js (archivos model.json y varios .bin).
- Utiliza la cámara del navegador para detectar qué objeto está frente a la cámara.
- Muestra el nombre del objeto si lo reconoce con alta certeza.

## Estructura del proyecto

/TensorflowApp/
├── Dataset/                      # Imágenes para entrenar, divididas por clases  
│ ├── AtariAsteroids/  
│ ├── GeniusOldMouse/  
│ ├── MagicBrainCalculator/  
│ └── negativos/  
├── ModeloMulticlase/             # Modelo exportado para usar en la web  
│ ├── model.json  
│ ├── group1-shard1of5.bin  
│ ├── group1-shard2of5.bin  
│ ├── group1-shard3of5.bin  
│ ├── group1-shard4of5.bin  
│ └── group1-shard5of5.bin  
├── detector_multiclase.js      # Código JS que carga el modelo y hace detección con la cámara  
├── entrenamiento_multiclase.js # Script Python para entrenar y exportar el modelo  
├── index.html                  # Página web principal  
└── style.css                   # Estilos CSS  

## Cómo entrenar el modelo

1. Asegurate de tener instalado Python 3 y las librerías `tensorflow`, `numpy`, `PIL`, `scikit-learn`, `matplotlib`.
2. Organiza tus imágenes en la carpeta `/Dataset/` siguiendo la estructura de clases.
3. Corre el script `entrenamiento_multiclase.py`:

```bash
python entrenamiento_multiclase.py
```
## Repositorio y despliegue

El código está alojado en GitHub y desplegado con GitHub Pages para acceso rápido vía web.

### Clonar el repositorio

Para clonar el proyecto en tu notebook, necesitas tener instalado Git. Luego puedes usar el siguiente comando:

```bash
git clone https://github.com/DavielDiaz/clarolabAR.git
```

## Abrir la app web localmente

Después de clonar, puedes abrir index.html directamente en un navegador,
pero para evitar problemas de permisos con la cámara y scripts,
es recomendable servirlo con un servidor local simple.
Si tienes Python instalado, puedes correr desde la carpeta raíz:

# Python 3.x
```bash
python -m http.server 8000
```
Luego abre en tu navegador

```bash
http://localhost:8000/index.html
```
## Acceder a la app online (GitHub Pages)

El proyecto también está publicado en GitHub Pages y puede accederse en:

https://davieldiaz.github.io/clarolabAR/

Podés usar este enlace para probar la aplicación sin necesidad de clonar nada.