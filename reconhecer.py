"""Reconhece letras pela webcam, monta a frase e conversa com o servidor_audio.py.

Teclas:
    ENTER     = finaliza a frase e fala no alto-falante do ESP32
    ESPAÇO    = adiciona um espaço entre as palavras
    BACKSPACE = apaga a ultima letra
    C         = limpa a frase inteira
    ESC       = sai
"""
import asyncio
from collections import Counter, deque
from pathlib import Path
from queue import Empty, Queue
import threading

import cv2
import joblib
import numpy as np
import websockets

from comum import DetectorMao, desenhar_mao, enriquecer, extrair_features

URL_SERVIDOR = "ws://localhost:8765"

LIMITE_CONFIANCA = 0.6   # confianca minima do modelo para um frame contar
MARGEM = 0.25            # folga minima da 1a letra sobre a 2a (evita letra indecisa)
JANELA = 25              # frames avaliados (cerca de 0,8 s a 30 fps)
PROPORCAO = 0.85         # fracao da janela que precisa concordar


class ClienteServidor:
    """Mantem a conexao WebSocket com o servidor_audio.py em uma thread separada."""

    def __init__(self, url: str):
        self.url = url
        self.fila_envio = Queue()
        self.fila_comandos = Queue()
        self.conectado = False
        self.modo = "libras"
        threading.Thread(target=self._rodar, daemon=True).start()

    def _rodar(self):
        asyncio.run(self._loop())

    async def _loop(self):
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    self.conectado = True
                    await ws.send("hello:libras")
                    print(f"[WS] Conectado ao servidor ({self.url})")

                    t_rec = asyncio.create_task(self._receber(ws))
                    t_env = asyncio.create_task(self._enviar(ws))
                    _, pendentes = await asyncio.wait(
                        [t_rec, t_env], return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in pendentes:
                        t.cancel()
            except Exception:
                self.conectado = False
                await asyncio.sleep(2)

    async def _receber(self, ws):
        async for msg in ws:
            if not isinstance(msg, str):
                continue
            if msg.startswith("modo:"):
                self.modo = msg.split(":", 1)[1]
            elif msg.startswith("cmd:"):
                self.fila_comandos.put(msg.split(":", 1)[1])

    async def _enviar(self, ws):
        while True:
            while not self.fila_envio.empty():
                msg = self.fila_envio.get_nowait()
                await ws.send(msg)
            await asyncio.sleep(0.03)

    def atualizar_frase(self, frase: str):
        self.fila_envio.put(f"frase:{frase}")

    def falar_frase(self, frase: str):
        frase = frase.strip()
        if frase:
            print(f"Enviando para falar: {frase}")
            self.fila_envio.put(f"falar:{frase}")

    def ler_comando(self):
        try:
            return self.fila_comandos.get_nowait()
        except Empty:
            return None


def main():
    caminho_modelo = Path(__file__).with_name("modelo.joblib")
    modelo = joblib.load(caminho_modelo)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    ws = ClienteServidor(URL_SERVIDOR)

    frase = ""
    votos = deque(maxlen=JANELA)
    ultima = None      # ultima letra aceita
    liberado = True    # True depois que a mao saiu do quadro (permite letra repetida)

    def definir_frase(nova_frase: str):
        nonlocal frase
        if nova_frase != frase:
            frase = nova_frase
            ws.atualizar_frase(frase)

    with DetectorMao() as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            achou = det.detectar(frame)

            letra_atual, conf = None, 0.0
            segunda_letra, segunda = "-", 0.0
            if achou:
                pontos, lado = achou
                desenhar_mao(frame, pontos)

                prob = modelo.predict_proba(enriquecer(extrair_features(pontos, lado).reshape(1, -1)))[0]
                ordem = np.argsort(prob)[::-1]
                conf = float(prob[ordem[0]])
                segunda = float(prob[ordem[1]])
                segunda_letra = str(modelo.classes_[ordem[1]])
                if conf >= LIMITE_CONFIANCA and conf - segunda >= MARGEM:
                    letra_atual = str(modelo.classes_[ordem[0]])
                votos.append(letra_atual)
            else:
                votos.clear()
                liberado = True

            validos = [v for v in votos if v is not None]
            topo, n = Counter(validos).most_common(1)[0] if validos else (None, 0)
            progresso = min(1.0, n / (PROPORCAO * JANELA))

            if len(votos) == JANELA and topo is not None and progresso >= 1.0:
                if topo != ultima or liberado:
                    definir_frase(frase + topo)
                    ultima = topo
                    liberado = False
                    votos.clear()

            # --- Comandos vindos do Arduino/ESP32 (ex: botao Falar na tela touch) ---
            cmd = ws.ler_comando()
            if cmd == "falar" and frase.strip():
                ws.falar_frase(frase)
                definir_frase("")
                ultima, liberado = None, True
            elif cmd == "espaco":
                if not frase.endswith(" "):
                    definir_frase(frase + " ")
                ultima = None
            elif cmd == "apagar":
                definir_frase(frase[:-1])
                ultima = None
            elif cmd == "limpar":
                definir_frase("")
                ultima, liberado = None, True

            # --- MELHORIA DE INTERFACE (Backgrounds semitransparentes) ---
            overlay = frame.copy()
            altura, largura = frame.shape[:2]
            cv2.rectangle(overlay, (0, 0), (largura, 80), (0, 0, 0), -1)
            cv2.rectangle(overlay, (0, altura - 85), (largura, altura), (0, 0, 0), -1)

            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # -------------------------------------------------------------

            # Textos da Interface
            cv2.putText(frame, f"Letra: {letra_atual or '-'} ({conf:.0%})", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"2a opcao: {segunda_letra} ({segunda:.0%})", (260, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Indicador de conexao com o servidor_audio.py no canto superior direito
            cor_ws = (0, 255, 0) if ws.conectado else (0, 0, 255)
            texto_ws = "Servidor: OK" if ws.conectado else "Servidor: OFF"
            cv2.putText(frame, texto_ws, (largura - 145, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_ws, 2)

            # Barra de Progresso
            cv2.rectangle(frame, (15, 50), (215, 65), (255, 255, 255), 1)
            cv2.rectangle(frame, (15, 50), (15 + int(200 * progresso), 65), (0, 255, 0), -1)

            # Frase construída
            cv2.putText(frame, f"Frase: {frase}", (15, altura - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Rodapé de instruções
            instrucoes = "ESPACO = espaco | BACKSPACE = apagar | ENTER = falar | C = limpar | ESC = sair"
            cv2.putText(frame, instrucoes, (15, altura - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Libras - Construcao de Frases", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == 13 and frase.strip():  # ENTER
                ws.falar_frase(frase)
                definir_frase("")
                ultima, liberado = None, True
            elif k == 32:  # ESPAÇO
                if not frase.endswith(" "):
                    definir_frase(frase + " ")
                ultima = None
            elif k == 8:  # BACKSPACE (Apaga letra por letra)
                definir_frase(frase[:-1])
                ultima = None
            elif k in (ord("c"), ord("C")):  # C (Limpa tudo)
                definir_frase("")
                ultima, liberado = None, True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()