"""
Práctica 3.1: Flujo de Inferencia y Predicciones Causales
=========================================================

ARCHIVO DE SOLUCIONES -

Este archivo contiene las soluciones completas de los ejercicios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys
from pathlib import Path
# Importar ModeloLineal desde software/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / 'software' / 'ModeloLineal'))
from ModeloLinealFull import BayesianLinearModel
import ModeloLineal as ml

np.random.seed(42)
CMAP = plt.get_cmap("tab10")
N = 5000

# EJERCICIO 1: Realidad causal básica
# Cargar datos: df = pd.read_csv('../datos/modelo_basico.csv')
# Variables: z, x, y
# Estructura causal: Z -> X, Z -> Y, X -> Y
# Objetivo: Estimar el efecto causal de X sobre Y (valor real: -2)

def estimar_efecto_causal_ej1(df):
    """
    Estima el efecto causal de X sobre Y.
    
    Backdoor: Z es confounder, hay que controlarlo.
    Phi debe incluir: intercepto, X, Z^2 (por la relación no lineal)
    """
    N = len(df)
    Z = df['z'].values
    X = df['x'].values
    Y = df['y'].values
    
    # Matriz de diseño con intercepto, X, y Z^2
    PHI = np.column_stack([
        np.ones(N),      # c_0: intercepto
        X,               # c_x: efecto de X (queremos estimar -2)
        Z**2             # c_z: control del confounder Z
    ])
    
    # Ajustar modelo bayesiano
    blm = BayesianLinearModel(basis=lambda x: x)
    blm.update(PHI, Y.reshape(N, 1))
    
    media = blm.location
    
    return {
        'coef_x': float(media[1]),  # Segundo coeficiente es X
        'media': media,
        'covarianza': blm.dispersion,
        'evidencia': blm.evidence()
    }


# EJERCICIO 2: Realidad causal compleja

def identificar_variables_backdoor_ej2():
    """
    Analisis del DAG para X -> Y:
    
    Estructura (segun consigna):
        z1 -> w1 -> x         (coef: 6, -1)
        z2 -> w2 -> y         (coef: 5, 1)
        z1, z2 -> z3 -> x, y  (coef: -4, 3, 2, -1)
        x -> w3 -> y          (coef: 2, -1)
    
    Camino causal: x -> w3 -> y (NO bloquear)
    Caminos backdoor: x <- z3 -> y (z3 es confounder)
    
    CONJUNTOS VALIDOS de variables de control (cualquiera cierra el backdoor):
        - {z3, z1}       -> z3 + z1
        - {z3, z2}       -> z3 + z2
        - {z3, w1}       -> z3 + w1
        - {z3, w2}       -> z3 + w2
    
    NO incluir: w3 (mediador - bloquea el efecto causal)
    
    Efecto causal real de X sobre Y: 2 * (-1) = -2
    """
    # Esta es UNA de las respuestas validas. El evaluador acepta cualquiera de:
    # ['z3', 'z1'], ['z3', 'z2'], ['z3', 'w1'], ['z3', 'w2']
    return {
        'variables_control': ['z3', 'z1'],  # z3 + z1
        'mediadores': ['w3'],          # w3 es el mediador (no incluir)
        'justificacion': (
            "z3 es el confounder que crea el camino backdoor x <- z3 -> y. "
            "Alternativas validas: controlar por {z1, z2} que son ancestros de z3. "
            "w3 es el mediador en el camino causal x -> w3 -> y, NO debe incluirse."
        )
    }


def estimar_efecto_causal_ej2(df):
    """
    Estima el efecto causal de X sobre Y controlando por z3.
    NO incluir w3 (mediador).
    
    Efecto causal real: -2
    """
    N = len(df)
    
    # Matriz de diseno: intercepto, x, z3 (control)
    PHI = np.column_stack([
        np.ones(N),
        df['x'].values,
        df['z3'].values,  # Control del confounder
    ])
    
    Y = df['y'].values.reshape(N, 1)
    
    blm = BayesianLinearModel(basis=lambda x: x)
    blm.update(PHI, Y)
    
    return {
        'coef_x': float(blm.location[1]),
        'variables_usadas': ['intercepto', 'x', 'z3'],
        'evidencia': blm.evidence()
    }


# EJERCICIO 3: Efecto causal del sexo biológico sobre la altura

def construir_modelo_base(df):
    """
    Modelo Base: altura = h0 + h1 * altura_madre
    """
    N = len(df)
    
    PHI = np.column_stack([
        np.ones(N),
        df['altura_madre'].values
    ])
    Y = df['altura'].values.reshape(N, 1)
    
    return PHI, Y


def construir_modelo_biologico(df):
    """
    Modelo Biológico: altura = h0 + h1 * altura_madre + h2 * I(sexo=F)
    """
    N = len(df)
    
    PHI = np.column_stack([
        np.ones(N),
        df['altura_madre'].values,
        (df['sexo'] == 'F').astype(float).values
    ])
    Y = df['altura'].values.reshape(N, 1)
    
    return PHI, Y


def construir_modelo_identitario(df):
    """
    Modelo Identitario: grupos de 2 personas con ordenadas diferentes.
    """
    N = len(df)
    
    # Altura de la madre como primera columna
    columnas = [df['altura_madre'].values]
    
    # Una columna por cada grupo (N/2 grupos de 2 personas)
    n_grupos = N // 2
    for g in range(n_grupos):
        indicadora = np.zeros(N)
        indicadora[2*g] = 1
        indicadora[2*g + 1] = 1
        columnas.append(indicadora)
    
    PHI = np.column_stack(columnas)
    Y = df['altura'].values.reshape(N, 1)
    
    return PHI, Y


def comparar_modelos_altura(df):
    """
    Compara los 3 modelos usando evidencia bayesiana.
    """
    # Construir modelos
    PHI_base, Y = construir_modelo_base(df)
    PHI_bio, _ = construir_modelo_biologico(df)
    PHI_ident, _ = construir_modelo_identitario(df)
    
    # Calcular evidencia
    ev_base = float(np.squeeze(ml.log_evidence(Y, PHI_base)))
    ev_bio = float(np.squeeze(ml.log_evidence(Y, PHI_bio)))
    ev_ident = float(np.squeeze(ml.log_evidence(Y, PHI_ident)))
    
    evidencias = {'Base': ev_base, 'Biológico': ev_bio, 'Identitario': ev_ident}
    
    # Calcular posterior con truco del máximo
    L_max = max(ev_base, ev_bio, ev_ident)
    scores = {
        'Base': np.exp(ev_base - L_max),
        'Biológico': np.exp(ev_bio - L_max),
        'Identitario': np.exp(ev_ident - L_max)
    }
    total = sum(scores.values())
    posterior = {k: v/total for k, v in scores.items()}
    
    mejor = max(evidencias, key=evidencias.get)
    
    return {
        'evidencia_base': ev_base,
        'evidencia_biologico': ev_bio,
        'evidencia_identitario': ev_ident,
        'mejor_modelo': mejor,
        'posterior_modelos': posterior
    }


def calcular_media_geometrica(log_evidencia, N):
    """
    Media geométrica = exp(log_evidencia / N)
    """
    return np.exp(log_evidencia / N)


# EJERCICIO 4: Falacia de la Tabla 2

def estimar_efecto_S_sobre_Y(df):
    """
    Estima el efecto causal de S sobre Y.
    
    DAG: U -> F, S, Y
         E -> F, S, Y
         F -> S
         S -> Y
    
    Para estimar S -> Y, controlamos por E y F (cierran backdoors).
    El coeficiente real de S es -2.
    """
    N = len(df)
    
    # Variables de control: e, f (cierran el camino backdoor)
    PHI = np.column_stack([
        np.ones(N),
        df['s'].values,  # Variable de interés
        df['e'].values,  # Control
        df['f'].values   # Control
    ])
    
    Y = df['y'].values.reshape(N, 1)
    
    blm = BayesianLinearModel(basis=lambda x: x)
    blm.update(PHI, Y)
    media = blm.location
    cov = blm.dispersion
    std_s = float(np.sqrt(cov[1, 1]))
    return {
        'coef_s': float(media[1]),  # Coeficiente de S
        'std_s': std_s,             # Desviación estándar del coeficiente de S
        'coef_otros': {
            'intercepto': float(media[0]),
            'e': float(media[2]),
            'f': float(media[3])
        },
        'advertencia': (
            "Los coeficientes de E y F NO son sus efectos causales sobre Y. "
            "Son solo controles para aislar el efecto de S. "
            "Para estimar el efecto de E o F, hay que diseñar otro modelo."
        )
    }


if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICACIÓN SOLUCIONES - Práctica 3.1")
    print("=" * 60)
    
    # Ejercicio 1
    print("\n--- Ejercicio 1: Modelo básico ---")
    try:
        df = pd.read_csv('../datos/modelo_basico.csv')
        resultado = estimar_efecto_causal_ej1(df)
        print(f"Coeficiente de X estimado: {resultado['coef_x']:.4f}")
        print(f"Valor real: -2")
        print(f"Error: {abs(resultado['coef_x'] - (-2)):.4f}")
    except FileNotFoundError:
        print("Ejecute primero generar_datos.py")
    
    # Ejercicio 2
    print("\n--- Ejercicio 2: Realidad compleja ---")
    try:
        df = pd.read_csv('../datos/realidad_compleja.csv')
        backdoor = identificar_variables_backdoor_ej2()
        print(f"Variables de control: {backdoor['variables_control']}")
        resultado = estimar_efecto_causal_ej2(df)
        print(f"Coeficiente de X estimado: {resultado['coef_x']:.4f}")
    except FileNotFoundError:
        print("Ejecute primero generar_datos.py")
    
    # Ejercicio 3
    print("\n--- Ejercicio 3: Modelos de altura ---")
    try:
        df = pd.read_csv('../datos/alturas.csv')
        resultado = comparar_modelos_altura(df)
        print(f"Evidencias:")
        print(f"  Base: {resultado['evidencia_base']:.2f}")
        print(f"  Biológico: {resultado['evidencia_biologico']:.2f}")
        print(f"  Identitario: {resultado['evidencia_identitario']:.2f}")
        print(f"Mejor modelo: {resultado['mejor_modelo']}")
    except FileNotFoundError:
        print("Ejecute primero generar_datos.py")
    
    # Ejercicio 4
    print("\n--- Ejercicio 4: Falacia Tabla 2 ---")
    try:
        df = pd.read_csv('../datos/falacia_tabla2.csv')
        resultado = estimar_efecto_S_sobre_Y(df)
        print(f"Coeficiente de S: {resultado['coef_s']:.4f} (real: -2)")
    except FileNotFoundError:
        print("Ejecute primero generar_datos.py")
    
    print("\n" + "=" * 60)
