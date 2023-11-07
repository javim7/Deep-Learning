# Descripción General

Este repositorio contiene los laboratorios completados como parte del curso de Deep Learning. Cada laboratorio aborda una serie de conceptos y técnicas fundamentales en el campo de deep learning y proporciona una experiencia práctica que permite a los estudiantes adquirir habilidades y conocimientos esenciales. Los cinco laboratorios abordan diversos temas, desde la construcción de redes neuronales simples hasta sistemas de recomendación y generación de imágenes con GANs.

## Contenido del Repositorio

### Laboratorio 1: Inicialización y Aprendizaje en una Red Neuronal Simple
En este laboratorio, exploramos conceptos clave en una red neuronal simple, como la inicialización de parámetros y la tasa de aprendizaje. Aprendimos cómo estos factores influyen en la convergencia y el rendimiento de la red.

### Laboratorio 2: Implementación de Redes Neuronales para Predicción de Índice de Aprobación de Películas
En este laboratorio, construimos modelos de redes neuronales para predecir el índice de aprobación de películas. Evaluamos varias arquitecturas y técnicas de regularización para determinar el modelo óptimo.

### Laboratorio 3: Implementación de Redes Neuronales en Series Temporales
En este laboratorio, exploramos la aplicación de redes neuronales en series temporales utilizando un conjunto de datos de manchas solares. Comparamos redes Feed Forward, RNN y LSTM para determinar la mejor arquitectura para el problema.

### Laboratorio 5: Generación de Imágenes MNIST con GAN
En este laboratorio, construimos una Red Generativa Adversaria (GAN) para generar imágenes que se asemejan al conjunto de datos MNIST. Exploramos la estructura del generador y el discriminador, así como el proceso de entrenamiento.

### Laboratorio 6: Sistemas de Recomendación para Libros
Este laboratorio se centró en la construcción de sistemas de recomendación para libros. Implementamos dos tipos de sistemas: basados en contenido y basados en filtros colaborativos, y evaluamos su rendimiento y eficacia.

## Autores
- Javier Mombiela ([GitHub](https://github.com/javim7))
- Roberto Ríos ([GitHub](https://github.com/robertoriosm))

# Laboratorios

## Laboratorio 1: Inicialización y Aprendizaje en una Red Neuronal Simple

En este primer laboratorio exploramos los conceptos fundamentales de una red neuronal simple. Utilizamos el código proporcionado en el archivo `NNsimple.ipynb` para abordar tres preguntas clave:

1. **Inicialización de Parámetros**: Investigamos si existe una diferencia significativa en la convergencia de los parámetros (pesos y sesgos) cuando se inicializan en 0 en comparación con la inicialización aleatoria. Descubrimos que la inicialización aleatoria permite la optimización del costo y el ajuste de pesos y sesgos, mientras que la inicialización en 0 resulta en neuronas "muertas" y predicciones incorrectas.

2. **Tasa de Aprendizaje (Learning Rate)**: Exploramos cómo la tasa de aprendizaje afecta la convergencia de la función de costo y los parámetros. Probamos tres tasas de aprendizaje: 0.01, 0.1 y 0.5. Observamos que una tasa de aprendizaje de 0.5 es la más eficaz en este ejemplo, ya que permite un aprendizaje rápido sin riesgo de saltarse mínimos.

3. **Función de Costo MSE**: Implementamos el Error Cuadrático Medio (MSE) como función de costo y adaptamos el código en consecuencia. Este cambio nos permite evaluar cómo la elección de la función de costo afecta el rendimiento de la red.

Este laboratorio nos brindó una comprensión sólida de los aspectos iniciales de la construcción de redes neuronales y cómo los parámetros, la inicialización y la tasa de aprendizaje influyen en el proceso de entrenamiento. Además, resaltó la importancia de seleccionar la función de costo adecuada para tareas específicas.

La exploración de estos conceptos sienta las bases para el estudio posterior de la optimización y el ajuste de hiperparámetros en redes neuronales.

**Código y Resultados**:
- El código necesario para este laboratorio se encuentra en el archivo `Laboratorio1/NNsimple-1.ipynb`.
- Se proporcionan respuestas a las preguntas planteadas junto con capturas de pantalla de los resultados en el archivo `Laboratorio1/lab1.pdf`.

## Laboratorio 2: Implementación de Redes Neuronales para Predicción de Índice de Aprobación de Películas

Para el segundo laboratorio se nos dio la libertad de definir una implementación de redes neuronales para predecir el índice de aprobación (approval_index) de películas. Elegimos un enfoque de regresión y seleccionamos un subconjunto de variables del conjunto de datos 'movie_statistic_dataset.csv' que mostraron una correlación significativa con el approval_index, a saber, 'movie_averageRating' y 'movie_numberOfVotes'.

El propósito de este laboratorio fue explorar diferentes arquitecturas de redes neuronales y técnicas de regularización para evaluar su rendimiento en la predicción del índice de aprobación de películas.

### Redes Neuronales Evaluadas

1. **Primer Modelo**:
   - 6 capas con función de activación ReLu.
   - Batch normalization.
   - Función de pérdida: Mean Squared Error (MSE).
   - Entrenado durante 75 épocas.
   
   Resultados:
   - MAE (Mean Absolute Error): 0.0927
   - MSE (Mean Squared Error): 0.0157
   - R2 (Coeficiente de determinación): 0.9897

2. **Segundo Modelo**:
   - 3 capas con función de activación Sigmoide.
   - Técnica de regularización: Dropout (0.5).
   - Función de pérdida: MSE.
   - Entrenado durante 75 épocas.
   
   Resultados:
   - MAE: 0.3042
   - MSE: 0.1812
   - R2: 0.8650

3. **Tercer Modelo**:
   - 9 capas con función de activación Tangente Hiperbólica.
   - Técnica de regularización: L2 (0.01).
   - Función de pérdida: MSE.
   - Entrenado durante 75 épocas.
   
   Resultados:
   - MAE: 1.0329
   - MSE: 1.7214
   - R2: -7570654406824.367 (negativo, indicando un pobre rendimiento)

### Discusión y Selección de Modelo Óptimo

El primer modelo con 6 capas, función de activación ReLu y Batch Normalization presenta el mejor rendimiento con un MAE y MSE bajos y un alto R2, lo que sugiere que es el más adecuado para predecir los índices de aprobación de películas. La diferencia de rendimiento puede atribuirse a la selección de componentes, como el número de capas, la función de activación y las técnicas utilizadas. La Batch Normalization ayudó a mitigar el problema del desvanecimiento del gradiente y acelerar el entrenamiento. La regularización L2 en el tercer modelo no fue beneficiosa en este contexto.

Concluimos que el primer modelo es el óptimo para este problema, ya que ofrece resultados más precisos en la predicción del índice de aprobación de películas.

Este laboratorio nos proporcionó una valiosa experiencia en la construcción y evaluación de redes neuronales para tareas de regresión, destacando la importancia de seleccionar cuidadosamente la arquitectura y las técnicas de regularización.


## Laboratorio 3: Implementación de Redes Neuronales en Series Temporales

En nuestro tercer laboratorio, nos enfrentamos al desafío de implementar redes neuronales para resolver un problema de regresión en una serie temporal, específicamente utilizando el conjunto de datos Monthly Sunspots, 'sunspots.csv'. Nuestra tarea era implementar tres arquitecturas de redes neuronales diferentes: Feed Forward NN (Red Simple), Recurrent Neural Network (RNN) y Long Short-Term Memory (LSTM).

### Arquitecturas de Redes Neuronales

1. **Feed Forward NN (Red Simple)**:
   - Esta red es sencilla y adecuada para problemas de regresión más simples.
   - Pros:
     - Fácil de implementar y entender.
     - Puede funcionar bien en casos de regresión simples.
   - Contras:
     - No modela adecuadamente las relaciones temporales en series temporales.
     - No puede recordar eventos pasados ni patrones a largo plazo en los datos.

2. **Recurrent Neural Network (RNN)**:
   - Esta red tiene conexiones recurrentes para capturar dependencias secuenciales en los datos.
   - Pros:
     - Capaz de capturar relaciones temporales y patrones a corto plazo en los datos.
     - Permite que la información fluya a través del tiempo.
   - Contras:
     - Dificultad para capturar relaciones a largo plazo.
     - Problemas para capturar patrones en secuencias muy largas.

3. **Long Short-Term Memory (LSTM)**:
   - Diseñada para superar las limitaciones de las RNN en la captura de dependencias a largo y corto plazo.
   - Pros:
     - Capaz de capturar patrones a largo plazo en los datos de series temporales.
     - Conserva información durante períodos prolongados.
   - Contras:
     - Puede ser más difícil de entender e implementar.
     - Posibles problemas de sobreajuste si no se configura adecuadamente.

### Resultados y Elección del Mejor Modelo

Tras implementar estas tres arquitecturas, evaluamos su rendimiento en el conjunto de prueba y calculamos el Root Mean Squared Error (RMSE) para cada modelo. Los resultados fueron los siguientes:

- RMSE en el conjunto de prueba:
  - FFNN: 0.1392
  - RNN: 0.1030
  - LSTM: 0.0980

Basándonos en los resultados, el modelo LSTM obtuvo el RMSE más bajo en el conjunto de prueba en la mayoría de las iteraciones, seguido por el modelo RNN. El modelo FFNN siempre tuvo el peor rendimiento. Por lo tanto, concluimos que la mejor opción para resolver este problema de regresión en una serie temporal es la arquitectura **Long Short-Term Memory (LSTM)**.

Dado que el conjunto de datos 'sunspots.csv' implica la necesidad de capturar tanto conexiones temporales a corto como a largo plazo, la elección de la red neuronal LSTM se justifica. Las LSTM tienen la capacidad de modelar patrones en datos de series temporales con una gran riqueza y pueden capturar tanto relaciones temporales de corta como de larga duración. Además, ofrecen una ventaja clave en la predicción de series temporales en comparación con las otras arquitecturas evaluadas. Su capacidad para conservar información a lo largo de períodos prolongados las hace ideales para resolver este tipo de problema.

El Laboratorio 3 nos permitio explorar y entender las ventajas y limitaciones de diferentes arquitecturas de redes neuronales en el contexto de series temporales, y llegamos conclusión de que, en este caso, la red LSTM es la elección óptima para obtener predicciones precisas en el conjunto de datos 'sunspots.csv'.

## Laboratorio 5: Generación de Imágenes MNIST con GAN

En este laboratorio, tuvimos construir una Red Generativa Adversaria (GAN) capaz de generar imágenes similares al conjunto de datos MNIST. Una GAN consta de dos componentes principales: el generador (G) y el discriminador (D), que compiten entre sí en un proceso de entrenamiento adversarial.

### Estructura de G(x) (Generador)

El generador toma un vector de ruido aleatorio de 100 dimensiones como entrada y genera imágenes que se asemejan a los dígitos MNIST. La estructura del generador es la siguiente:

- Capa de Entrada: La entrada es un vector de ruido de dimensión 100.
- Capa Densa (Fully Connected): Una capa densa con 256 neuronas y activación LeakyReLU (para permitir un pequeño gradiente negativo).
- Capa Densa: Otra capa densa con 512 neuronas y activación LeakyReLU.
- Capa Densa: Una capa densa con 1024 neuronas y activación LeakyReLU.
- Capa de Salida: La capa de salida es una capa densa con 784 neuronas (28x28 píxeles) y utiliza la activación 'tanh' para escalar los valores generados al rango entre -1 y 1, que es válido para imágenes en escala de grises.

El generador crea imágenes a partir del ruido aleatorio proporcionado como entrada, y la activación 'tanh' asegura que los píxeles generados estén en el rango correcto.

### Estructura de D(x) (Discriminador)

El discriminador es una red neuronal que tiene como objetivo distinguir entre imágenes reales (del conjunto de datos MNIST) y falsas (generadas por el generador). La estructura del discriminador es la siguiente:

- Capa de Entrada: La entrada de la red es una imagen de 28x28 píxeles en escala de grises.
- Capa de Aplanamiento: La primera capa aplana la imagen en un vector unidimensional para que pueda ser procesado por capas densas.
- Capa de Dropout: Se utiliza una capa de dropout para regularizar la red y evitar el sobreajuste.
- Capa Densa (Fully Connected): Una capa densa con 1024 neuronas y activación LeakyReLU.
- Capa de Dropout: Otra capa de dropout para regularización.
- Capa Densa: Una capa densa con 512 neuronas y activación LeakyReLU.
- Capa de Dropout: Capa de dropout adicional.
- Capa Densa: Otra capa densa con 512 neuronas y activación LeakyReLU.
- Capa de Salida: Una capa densa con 1 neurona y activación sigmoide. Esta neurona de salida produce una probabilidad que indica si la imagen de entrada es real o falsa.

El discriminador tiene una estructura profunda con capas densas intercaladas con capas de dropout para evitar el sobreajuste. La activación LeakyReLU se utiliza para permitir gradientes negativos. La capa de salida utiliza la activación sigmoide para producir una probabilidad de clasificación binaria.

### Resultados antes y después del entrenamiento

Durante el proceso de entrenamiento, se realizaron 100 épocas, y se observaron las imágenes generadas después de 1, 25, 50, 75 y 100 épocas completas, respectivamente. Los resultados se describen a continuación:

- 1ª Época: Las imágenes son ruido aleatorio, ya que el generador no ha aprendido patrones significativos.
- 25ª Época: Las imágenes comienzan a mostrar formas vagamente similares a los dígitos MNIST, aunque carecen de detalles.
- 50ª Época: La calidad de las imágenes mejora notablemente, con formas más definidas pero posibles imperfecciones.
- 75ª Época: Las imágenes se asemejan más a los dígitos, y se observa un cambio significativo, pero aún no son perfectas.
- 100ª Época: Las imágenes muestran una pequeña mejora en comparación con la 75ª época, pero no se observa una diferencia significativa.

Se puede concluir que el modelo fue exitoso en la generación de imágenes que se asemejan a los dígitos MNIST al completar las 100 épocas. Sin embargo, la perfección no se alcanzó, y agregar más épocas no aseguraría una mejora significativa. Además, el tiempo de entrenamiento podría aumentar considerablemente. Por lo tanto, se considera que el modelo cumplió con éxito su tarea de generar imágenes de dígitos MNIST.

El laboratorio 5 nos demostró cómo una GAN puede generar datos realistas a partir de ruido aleatorio y cómo la estructura del generador y el discriminador es crucial para el éxito del proceso de generación. La GAN puede ser una herramienta poderosa para crear datos sintéticos en diversas aplicaciones, como la generación de imágenes, el aumento de datos y la creación de ejemplos de entrenamiento en campos como el aprendizaje automático y la visión por computadora.

## Laboratorio 6: Sistemas de Recomendación para Libros

En este laboratorio, se construyeron dos sistemas de recomendación para libros utilizando un dataset que consta de información sobre libros, usuarios y calificaciones de libros. Se realizaron dos tipos de sistemas de recomendación: basados en contenido y basados en filtros colaborativos.

### Análisis Exploratorio de Datos (EDA) del Dataset
Antes de profundizar en los modelos, se realizó un análisis exploratorio de datos para comprender la información disponible en los conjuntos de datos. Se identificaron características relevantes y se realizaron transformaciones, como la codificación de etiquetas para las columnas 'Publisher', 'Book-Title' y 'Book-Author'. Los conjuntos de datos se dividieron en conjuntos de entrenamiento y prueba, con diferentes divisiones en función de cada modelo.

### Estructura de las Redes y Funcionamiento
**Modelo Basado en Contenido**
- El modelo basado en contenido utiliza una red neuronal con capas densas.
- Capas del modelo: Capa de entrada, capa oculta con 64 neuronas (ReLU), capa oculta con 32 neuronas (ReLU), y capa de salida con una sola neurona (lineal).
- El modelo se compila utilizando la función de pérdida de error cuadrático medio (MSE) y el optimizador Adam.
- Se entrena el modelo durante 5 épocas.

**Modelo Basado en Filtros Colaborativos**
- El modelo basado en filtros colaborativos involucra la incorporación de usuarios y libros.
- Capas del modelo: Capas de entrada para usuario y libro, capas de incorporación (embedding), capas densas y capa de salida.
- El modelo se compila utilizando la función de pérdida de error cuadrático medio (MSE) y el optimizador Adam.
- Se entrena el modelo durante 3 épocas.

### Resultados y Comparación de Modelos
**Modelo Basado en Contenido**
- Error Cuadrático Medio (MSE): 14.8576
- Raíz del Error Cuadrático Medio (RMSE): 3.8546
- Coeficiente de Determinación (R^2): -1.9383e-05
- Observaciones: Las recomendaciones generadas por este modelo presentan predicciones de calificación que superan el rango válido (1-10). El rendimiento es deficiente en este contexto.

**Modelo Basado en Filtros Colaborativos**
- Error Cuadrático Medio (MSE): 1.5684
- Raíz del Error Cuadrático Medio (RMSE): 1.2531
- Coeficiente de Determinación (R^2): 0.6858
- Observaciones: Este modelo demuestra un rendimiento sustancialmente mejor, con métricas MSE y RMSE más bajos y un R^2 más alto. A pesar de un posible sobreajuste debido a las épocas de entrenamiento, las recomendaciones son más precisas y efectivas en este caso.

**Conclusión:**
El modelo basado en filtros colaborativos sobresale como la mejor opción debido a su capacidad para aprender de las interacciones pasadas y personalizar recomendaciones en función de las preferencias específicas de los usuarios. En contraste, el sistema de recomendaciones basado en contenido presenta desventajas notables, incluyendo un bajo rendimiento en este contexto y la limitación de depender en gran medida de las características del contenido.

Los resultados indican que los filtros colaborativos son más efectivos para sistemas de recomendación de libros, ya que consideran las interacciones entre los usuarios y los libros, lo que permite una mayor personalización y precisión en las recomendaciones.

Este laboratorio nos ofrecio una visión profunda de las estrategias para la construcción de sistemas de recomendación y las consideraciones clave al elegir entre modelos basados en contenido y filtros colaborativos. Estas habilidades son valiosas en la industria y pueden aplicarse a una amplia gama de aplicaciones donde la personalización y la recomendación son esenciales para mejorar la experiencia del usuario y aumentar la satisfacción del cliente.