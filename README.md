# Reconhecimento do Alfabeto de Libras em Tempo Real

Sistema em Python para coleta de dados, treinamento de modelo e reconhecimento em tempo real de letras estáticas do alfabeto de Libras via webcam, utilizando **MediaPipe Hand Landmarker** e **Random Forest (scikit-learn)**. O sistema também conta com montagem de frases e integração via **WebSocket** com um servidor de áudio/ESP32 para síntese de voz.

## Funcionalidades

- **Detecção de mão independente de lateralidade e distância:** os 21 landmarks da mão são centralizados no punho, normalizados pela escala e espelhados automaticamente quando a mão esquerda é detectada.
- **Engenharia de atributos:** além das 63 coordenadas `(x, y, z)`, o modelo utiliza as 210 distâncias euclidianas entre todos os pares de pontos da mão (total de 273 features), melhorando a distinção entre sinais parecidos.
- **Coleta assistida:** captura contínua de amostras em CSV com opção de salvar imagens para auditoria.
- **Validação realista no treino:** utiliza *Leave-One-Group-Out* quando há 2 ou mais pessoas no dataset (evitando vazamento de dados entre treino e teste) ou divisão por blocos sequenciais quando há apenas 1 pessoa, além de *data augmentation* (rotação de ±15° e ruído gaussiano).
- **Construção de frases e integração WebSocket:** janela deslizante de votação para confirmar letras com estabilidade e envio da frase para um servidor local (`ws://localhost:8765`) conectado a um ESP32.

## Estrutura do Projeto

```text
.
├── comum.py               # Detector MediaPipe, normalização e extração de features
├── coletar.py             # Script para coletar amostras pela webcam (gera dataset/amostras.csv)
├── treinar.py             # Treina o RandomForestClassifier e salva o modelo.joblib
├── reconhecer.py          # Reconhecimento em tempo real, montagem de frases e cliente WebSocket
├── modelo.joblib          # Modelo treinado (gerado pelo treinar.py)
├── requirements.txt       # Dependências do projeto
└── README.md