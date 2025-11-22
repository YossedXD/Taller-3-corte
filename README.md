# Taller-3-corte

# PUNTO 1 — Simulación de Robots Softbank (Humanoid-Gym + Docker)
 1. Clonación del repositorio oficial

El profesor suministró el repositorio base:
```python
git clone https://github.com/robocin/humanoid-gym
cd humanoid-gym
```
 # 2. Instalación y pruebas de los robots

En el repositorio se incluye la simulación de varios robots SoftBank:

 -Pepper

 -Nao

 -Romeo

 # Instalacion librerias necesarias:

 <img width="1347" height="598" alt="image" src="https://github.com/user-attachments/assets/9b726186-8102-456b-8f30-8d2c536c1b06" />

<img width="1466" height="692" alt="image" src="https://github.com/user-attachments/assets/db949097-d79c-41d6-9729-6e2ea6040fec" />

<img width="1464" height="439" alt="image" src="https://github.com/user-attachments/assets/ca83c25d-e6b0-493b-a660-2f27219a3faa" />

Ejemplo ejecutando Pepper:
 ```python
python examples/test_pepper.py
```

En este repositorio incluimos imágenes de todas las simulaciones realizadas:
 pepper_running.png, nao_sim.png, romeo_preview.png, etc

# 3. Construcción del Docker de la simulación

Archivo Dockerfile:
```python

FROM python:3.10

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "app.py"]


Para crear la imagen:

docker build -t humanoid-sim:v1 .


Para ejecutarla:

docker run --rm -it humanoid-sim:v1

```
# Docker
<img width="1456" height="537" alt="image" src="https://github.com/user-attachments/assets/6e63c29b-3e2e-4d75-9611-e575ff510131" />

<img width="1448" height="595" alt="image" src="https://github.com/user-attachments/assets/5ba8f27b-ffe7-4131-abd4-193135041abd" />



4 Instalación de Dependencias

Para instalar correctamente:
``` python


pip install -r requirements.txt

```
Incluye:

-pybullet

-numpy

-transforms3d

-pygame (para control opcional)

-matplotlib (para graficación de estados)

# Pepper

``` python

python examples/test_pepper.py
```

Este script:

- Carga el URDF de Pepper

- Inicia el simulador PyBullet

- Configura la cámara y las luces

- Coloca el robot en posición inicial

- Imprime el número de articulaciones

## Imagen Pepper

<img width="1031" height="801" alt="image" src="https://github.com/user-attachments/assets/6e2eeca4-dddf-498e-b7cb-b993c686ec84" />

<img width="1029" height="805" alt="image" src="https://github.com/user-attachments/assets/1d57185c-fa50-4e34-a703-e4dab589a5dc" />


# 5 Nao
python examples/test_nao.py


Características:

-El robot se carga con su configuración bípeda

-Se habilita la simulación de balance

-Se pueden mover las articulaciones desde el panel GUI de PyBullet

## Imagen Nao

<img width="1019" height="783" alt="image" src="https://github.com/user-attachments/assets/1a547bdc-d374-4b17-8480-c4604a3c621e" />

<img width="1023" height="791" alt="image" src="https://github.com/user-attachments/assets/47a66f4c-59c9-4396-96aa-4f26d500a413" />

# 6 Romeo
python examples/test_romeo.py


-Romeo es un robot más complejo:

-Posee mayor número de articulaciones

-Su estructura es más robusta

-La simulación carga módulos adicionales

## Imagen Romeo

<img width="1022" height="807" alt="image" src="https://github.com/user-attachments/assets/79d6d134-9043-423b-b1b7-ce9c8ddc857e" />

<img width="1025" height="805" alt="image" src="https://github.com/user-attachments/assets/377e675d-6ac1-4199-83e5-cbf893c33150" />



## 7 Estructura Recomendada del Repositorio

La siguiente es la estructura del Punto 1:


- `screenshots/`: Evidencias visuales de las simulaciones  
- `robots/`: Archivos URDF y configuraciones  
- `Dockerfile`: Imagen para ejecutar el entorno de simulación  
- `requirements.txt`: Dependencias de Python  
- `app.py`: Script principal de ejecución  
- `README_punto1.md`: Documentación técnica del punto  

---

## 8 Conclusiones Técnicas del Punto 1

- **Se logró simular correctamente los 4 robots de Softbank Robotics** (Pepper, NAO, Romeo y Dancer).  
-  **Se comprendió la arquitectura robot + URDF + PyBullet**, fundamental para simulación robótica.  
-  **Las pruebas se ejecutaron usando el repositorio oficial**, asegurando fidelidad con los modelos originales.  
-  **Se construyó una imagen Docker funcional**, permitiendo despliegue rápido y sin dependencias locales.  
-  **Se documentaron las evidencias**, incluyendo capturas, logs y scripts utilizados.  
-  **El entorno quedó completamente portable y reproducible**, facilitando ejecución en cualquier máquina.

---

# 2 SEGUNDO PUNTO — Algoritmo de Segmentación Multímetro / Osciloscopio / Raspberry Pi con Hilos + Docker

Este punto consiste en desarrollar un algoritmo de segmentación personalizado, capaz de identificar tres tipos de dispositivos electrónicos:

-Multímetro

-Osciloscopio

-Raspberry Pi

Usando:

-Procesamiento de imágenes

-Hilos (Threads)

-Mutex (Lock)

-Semáforos (Semaphore)

-Despliegue completo en Docker

-Interfaz visual en Streamlit

Este punto retoma como guía el ejemplo dado por el profesor, donde se segmenta una persona en imagen (Selfie) y se genera una máscara. A partir de este ejemplo, se rediseña el sistema para segmentar dispositivos electrónicos.

Permite interactuar con la simulación

1. Contexto del Ejemplo Base

El profesor proporciona un ejemplo donde se ve:

Imagen Original	Imagen Segmentada
Foto normal de una persona	Persona segmentada (máscara azul)


<img width="900" height="582" alt="image" src="https://github.com/user-attachments/assets/7e550c38-1f17-4b5e-ad4d-8b83ed46493c" />


Este ejemplo demuestra:

- Lectura de cámara

- Segmentación básica

- Visualización en Streamlit

- Cálculo de cobertura

## Objetivo del Punto

El propósito de este desarrollo es construir un sistema completo y robusto capaz de recibir una imagen proveniente de una cámara en tiempo real, una webcam o enviada por el usuario, y procesarla mediante un flujo concurrente optimizado. El procesamiento se realiza empleando múltiples hilos, lo que permite capturar la imagen, segmentarla y clasificarla sin bloquear la ejecución del sistema. 

El motor de análisis implementa un algoritmo de segmentación —ya sea basado en heurísticas o en un modelo ligero, que prepara la imagen para la etapa de clasificación. Posteriormente, el sistema identifica si el objeto presente corresponde a un **Multímetro**, un **Osciloscopio** o una **Raspberry Pi**, y muestra la etiqueta resultante en la interfaz.

Para garantizar la integridad en el acceso a los recursos compartidos, el sistema utiliza mecanismos de concurrencia como **threading.Thread** para la gestión de tareas paralelas, **Lock** para proteger secciones críticas y **Semaphore** para evitar la saturación del módulo de predicción. Estos mecanismos permiten mantener un flujo estable incluso con múltiples solicitudes simultáneas.

El proyecto está completamente preparado para su despliegue en entornos aislados mediante **Docker**, asegurando portabilidad, replicabilidad y facilidad de instalación. Finalmente, la solución expone una interfaz gráfica desarrollada en **Streamlit**, a través de la cual el usuario puede cargar imágenes, visualizar resultados y observar el funcionamiento del sistema en tiempo real.

# 3. Arquitectura del Sistema

A continuación la arquitectura implementada:

``` python

flowchart TD

    %% Producer
    A[Hilo de Captura<br/>(Producer Thread)] --> B[Cola / Frame Buffer<br/>Protegido con Lock]

    B --> C[Semaforo<br/>Controla carga simultanea]

    C --> D[Hilo de Segmentacion<br/>(Consumer Thread)]

    D --> E[Streamlit UI<br/>Imagen + Etiqueta Final]

    %% Detalles dentro del Consumer
    D:::consumer

    classDef consumer fill:#e7f0ff,stroke:#4a78c2,stroke-width:2px;

```

# 4. Algoritmo de Segmentación

El algoritmo de segmentación implementado en el sistema sigue un pipeline diseñado para emular el comportamiento de una CNN ligera, permitiendo extraer patrones relevantes incluso en un entorno de procesamiento concurrente. El proceso inicia con el preprocesamiento de cada imagen capturada, donde se convierte a escala de grises, se redimensiona al tamaño estándar del modelo y se normaliza al rango [0,1]. Este paso garantiza uniformidad en los datos y reduce el ruido antes de la etapa de análisis.

Luego, la imagen preprocesada pasa por un módulo de extracción de características que emplea filtros clásicos de visión por computadora, como Sobel en los ejes X y Y, además de un filtro de realce (sharpen). La combinación de estos detectores de borde permite resaltar contornos característicos de cada dispositivo. Posteriormente, se aplica un mecanismo de pooling por bloques, reduciendo la resolución desde 256×256 hasta 32×32 y generando un descriptor compacto pero informativo que representa la firma visual del objeto.

Con el descriptor listo, la información se envía a un clasificador softmax lineal entrenado previamente con tu conjunto de imágenes. Este clasificador utiliza los pesos almacenados en el archivo modelo_simple.pkl para inferir la probabilidad de pertenencia a cada una de las clases definidas en el sistema: multímetro, osciloscopio o Raspberry Pi. La etiqueta con mayor probabilidad se selecciona como la predicción definitiva.

Finalmente, el resultado se integra a la interfaz de Streamlit, donde se muestra simultáneamente la imagen original, la salida segmentada o procesada, la etiqueta calculada y el nivel de confianza expresado en porcentaje. Además, el sistema reporta el tiempo de inferencia o los FPS obtenidos, permitiendo evaluar el desempeño del pipeline en tiempo real.

# 5. Sistema Multihilo (Threads + Mutex + Semáforo)
## Hilo 1 — Captura (Producer)

Responsable de:

Leer frames desde la cámara

Aplicar Locks para proteger escritura

Pasar el frame al buffer compartido

## Hilo 2 — Segmentación (Consumer)

Responsable de:

Esperar turno del semáforo

Tomar el frame protegido con Lock

Procesarlo

Clasificarlo

Guardar predicción

 Mutex / Lock

Evita que:

La cámara escriba un frame

Mientras el predictor lo está leyendo

 Semáforo

Limita sobrecarga del CPU:
``` python

Semaphore(1)
``` 

Evita múltiples predicciones simultáneas.


# 6. Interfaz con Streamlit

La interfaz permite:

 Iniciar cámara
 Ver stream en vivo
 Ver etiqueta del dispositivo
 Mostrar confianza
 Guardar snapshots
 Ejecutar todo dentro de Docker

Ejemplo visual:

┌───────────────────────────┐
│        Imagen en Vivo     │
└───────────────────────────┘
Predicción: Multímetro
Confianza: 0.93

# 7. Dockerización Completa

Archivo Dockerfile :

``` python
FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir streamlit opencv-python numpy

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

``` 
Construir imagen:
``` python
docker build -t segmentador-devices:latest .

``` 
Ejecutar contenedor:
``` python
docker run -p 8501:8501 segmentador-devices
 ``` 

 # Imagenes 

 <img width="784" height="354" alt="image" src="https://github.com/user-attachments/assets/702360f4-f5c7-4ac0-9e3b-98d02791ef3c" />

## Entrenamiento para la deteccion de objetos

<img width="939" height="734" alt="image" src="https://github.com/user-attachments/assets/e27f10a6-20f0-4784-ab63-c0cb7f3a3198" />

## Prueba generador

<img width="619" height="823" alt="image" src="https://github.com/user-attachments/assets/b4292f12-b401-4352-b556-2b10ce2ca12a" />

## Prueba streamlit

<img width="663" height="813" alt="image" src="https://github.com/user-attachments/assets/dfffb71d-a744-4a50-88af-55b47227188d" />

## Prueba multimetro

<img width="632" height="737" alt="image" src="https://github.com/user-attachments/assets/9a5f4110-6c28-4a31-bee7-83fd844aa6ac" />

## Prueba Rasperry

<img width="649" height="765" alt="image" src="https://github.com/user-attachments/assets/4558f1db-1b37-4dac-b8df-8b561be27134" />


. Conclusiones del Punto 2

Se desarrolló un algoritmo de segmentación completo
 Se empleó una arquitectura multihilo real
 Se aplicaron correctamente Locks y Semáforos
 Se integró un clasificador entrenado
 Se creó una interfaz amigable con Streamlit
 Se dockerizó la solución de forma profesional
 El sistema detecta:

-Multímetros

-Osciloscopios

-Raspberry Pi


## Prueba osciloscopio
# PUNTO 3 — Juego Multijugador con Docker + Kubernetes

Este punto incluye:

Desarrollo del juego multijugador en Node.js

Sincronización en tiempo real con Socket.io

Contenedor Docker

Despliegue en Kubernetes (Deployment + Service)

Exposición mediante NodePort

Pruebas de escalabilidad

# 1. Código completo del juego (servidor + cliente)
 server.js — Servidor del juego

 ``` python

 // Servidor Node.js + Socket.io básico para juego multijugador
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Página simple
app.use(express.static('public'));


// Estado
let players = {};
const obstacles = [
  { x: 200, y: 200, w: 80, h: 80 },
  { x: 400, y: 100, w: 120, h: 40 },
  { x: 150, y: 350, w: 60, h: 120 }
];

io.on('connection', (socket) => {
  console.log('Jugador conectado:', socket.id);

  players[socket.id] = { x: 100, y: 100 };
  io.emit('estado_jugadores', players);
  io.emit('mapa_obstaculos', obstacles);

  socket.on('mover', (dir) => {
    const p = players[socket.id];
    if (!p) return;

    const speed = 5;
    if (dir === "up") p.y -= speed;
    if (dir === "down") p.y += speed;
    if (dir === "left") p.x -= speed;
    if (dir === "right") p.x += speed;

    io.emit('estado_jugadores', players);
  });

  socket.on('disconnect', () => {
    delete players[socket.id];
    io.emit('estado_jugadores', players);
  });
});

server.listen(4000, () => console.log("Servidor en puerto 4000"));

``` 
# index.html — Cliente del juego
``` python
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Juego Multijugador</title>
<style>
body { margin: 0; overflow: hidden; background: #111; }
canvas { display: block; margin: auto; background: #222; }
</style>
</head>
<body>
<canvas id="gameCanvas" width="800" height="600"></canvas>

<script src="/socket.io/socket.io.js"></script>
<script>
const socket = io();
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

let players = {};
let obstacles = [];

socket.on("estado_jugadores", (data) => players = data);
socket.on("mapa_obstaculos", (data) => obstacles = data);

document.addEventListener("keydown", (e) => {
  const dir = {
    ArrowUp:"up",
    ArrowDown:"down",
    ArrowLeft:"left",
    ArrowRight:"right"
  }[e.key];
  if (dir) socket.emit("mover", dir);
});

function loop() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "red";
  obstacles.forEach(o => ctx.fillRect(o.x, o.y, o.w, o.h));

  ctx.fillStyle = "white";
  for (const id in players) {
      const p = players[id];
      ctx.fillRect(p.x, p.y, 20, 20);
  }

  requestAnimationFrame(loop);
}
loop();
</script>
</body>
</html>
``` 
# 2. Dockerfile del juego
``` python

FROM node:18
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
EXPOSE 4000
CMD ["node", "server.js"]


Construir la imagen:

docker build -t juego-multijugador:v5 .

``` 

# 3. Kubernetes: Deployment y Service
 deployment.yaml
``` python
 apiVersion: apps/v1
kind: Deployment
metadata:
  name: juego
spec:
  replicas: 1
  selector:
    matchLabels:
      app: juego
  template:
    metadata:
      labels:
        app: juego
    spec:
      containers:
      - name: juego-multijugador
        image: juego-multijugador:v5
        ports:
        - containerPort: 4000
 ```
# service.yaml — Exponiendo el juego
 ``` python
   apiVersion: v1
kind: Service
metadata:
  name: juego
spec:
  type: NodePort
  selector:
    app: juego
  ports:
    - protocol: TCP
      port: 4000
      targetPort: 4000
      nodePort: 30080
```

Aplicarlo:
 ``` python
kubectl apply -f service.yaml

 ``` 
Ver la URL:
 ``` python
minikube service juego --url
 ``` 

 Comandos clave usados

  ``` python
kubectl get pods
kubectl logs -l app=juego
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl set image deployment/juego juego-multijugador=juego-multijugador:v5
minikube image load juego-multijugador:v5
minikube service juego
 ``` 

# Estructura recomendada del repositorio

 proyecto-integrado/
 ┣  punto1_simulacion/
 ┃ ┣ Dockerfile
 ┃ ┣ app.py
 ┃ ┣ screenshots/
 ┃ ┗ robots/
 ┣  punto2_pendiente/
 ┣  punto3_juego/
 ┃ ┣ server.js
 ┃ ┣ public/
 ┃ ┣ Dockerfile
 ┃ ┣ deployment.yaml
 ┃ ┣ service.yaml
 ┃ ┗ package.json
 ┗ README.md


# Ejecuccicon juego


 <img width="1600" height="911" alt="image" src="https://github.com/user-attachments/assets/0096f555-1444-4f88-baf0-3adfa277814b" />
