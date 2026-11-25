"""Treina o classificador com o dataset/amostras.csv e salva modelo.joblib.

Se houver 2 ou mais pessoas no banco, a avaliacao deixa uma pessoa de fora por vez,
que e o jeito honesto de saber como o modelo vai se sair com alguem novo.
Com uma pessoa so, a avaliacao separa blocos de amostras seguidas (e nao frames
soltos), para nao testar com frames quase iguais aos do treino.
"""
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

from comum import enriquecer

rng = np.random.default_rng(0)
TAM_BLOCO = 25  # amostras seguidas que ficam juntas na avaliacao de uma pessoa so


def criar_modelo():
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)


def augmentar(X, y, copias=2):
    """Gira a mao um pouco (+-15 graus) e adiciona ruido leve, so no treino."""
    pts = X.reshape(-1, 21, 3)
    saidas, ys = [X], [y]
    for _ in range(copias):
        ang = np.radians(rng.uniform(-15, 15, size=(len(pts), 1)))
        c, s = np.cos(ang), np.sin(ang)
        novo = pts.copy()
        novo[:, :, 0] = pts[:, :, 0] * c - pts[:, :, 1] * s
        novo[:, :, 1] = pts[:, :, 0] * s + pts[:, :, 1] * c
        novo += rng.normal(0, 0.01, novo.shape)
        saidas.append(novo.reshape(len(pts), -1))
        ys.append(y)
    return np.vstack(saidas), np.concatenate(ys)


def treinar_e_prever(Xtr, ytr, Xte):
    Xa, ya = augmentar(Xtr, ytr)
    return criar_modelo().fit(enriquecer(Xa), ya).predict(enriquecer(Xte))


def avaliar(real, previsto):
    print(f"\nAcuracia: {accuracy_score(real, previsto):.3f}")
    print(classification_report(real, previsto, zero_division=0))
    erros = Counter((r, p) for r, p in zip(real, previsto) if r != p)
    if erros:
        print("Confusoes mais comuns (real -> previsto):")
        for (r, p), n in erros.most_common(8):
            print(f"  {r} -> {p}: {n}")


def main():
    df = pd.read_csv("dataset/amostras.csv")
    X = df.drop(columns=["pessoa", "letra"]).values.astype(np.float32)
    y = df["letra"].values
    grupos = df["pessoa"].values

    print("Amostras por letra:", dict(sorted(Counter(y).items())))
    poucas = [l for l, n in Counter(y).items() if n < 100]
    if poucas:
        print("Aviso: menos de 100 amostras em", sorted(poucas))

    if len(set(grupos)) >= 2:
        previsto = np.empty_like(y)
        for tr, te in LeaveOneGroupOut().split(X, y, grupos):
            previsto[te] = treinar_e_prever(X[tr], y[tr], X[te])
        print("\nValidacao deixando uma pessoa de fora por vez:")
        avaliar(y, previsto)
    else:
        print("\nAviso: so ha uma pessoa no banco. A avaliacao separa blocos de",
              TAM_BLOCO, "amostras seguidas,")
        print("o que e mais honesto que um sorteio simples, mas ainda otimista:")
        print("outra pessoa, outra luz ou outro dia vao dar resultado pior.")
        pos = df.groupby("letra").cumcount()
        blocos = (df["letra"] + "_" + (pos // TAM_BLOCO).astype(str)).values
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, blocos))
        avaliar(y[te], treinar_e_prever(X[tr], y[tr], X[te]))

    Xa, ya = augmentar(X, y)
    modelo = criar_modelo().fit(enriquecer(Xa), ya)
    joblib.dump(modelo, "modelo.joblib")
    print("\nModelo salvo em modelo.joblib")


if __name__ == "__main__":
    main()
