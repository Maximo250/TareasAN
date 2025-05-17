{python}
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numdifftools as nd

x_vals = np.linspace(1e-6, np.pi, 200)

f = lambda x: np.sqrt(x) * np.sin(x**2)

def derf(x):
  res = np.zeros_like(x)
mask = x > 0
res[mask] = 2*x[mask]*np.sqrt(x[mask])*np.cos(x[mask]**2) + (np.sin(x[mask]**2)/(2*np.sqrt(x[mask])))
return res

def dderf(x):
  res = np.zeros_like(x)
mask = x > 0
res[mask] = 4*np.sqrt(x[mask])*np.cos(x[mask]**2) - np.sin(x[mask]**2)*(4*x[mask]**2*np.sqrt(x[mask]) + 1/(4*x[mask]*np.sqrt(x[mask])))
return res

def analizar_funcion_c(f, derf, dderf, x_values, nombre):
  plt.figure(figsize=(8,6))
plt.plot(x_values, f(x_values), color="darkred", linewidth=1.5)
plt.title(f"Función {nombre}")
plt.grid()
plt.show()
df_01 = nd.Derivative(f, step=0.1, method='central', order=2)
df_005 = nd.Derivative(f, step=0.05, method='central', order=2)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=df_01(x_values), mode='lines', name='h=0.1', line=dict(color='teal')))
fig.add_trace(go.Scatter(x=x_values, y=df_005(x_values), mode='lines', name='h=0.05', line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=x_values, y=derf(x_values), mode='lines', name='Derivada exacta', line=dict(color='goldenrod')))
fig.update_layout(title=f"Aproximación de la derivada de {nombre}", xaxis_title="x", yaxis_title="f'(x)", template="plotly_white")
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=np.abs(derf(x_values)-df_01(x_values)), mode='lines', name='Error h=0.1', line=dict(color='teal')))
fig.add_trace(go.Scatter(x=x_values, y=np.abs(derf(x_values)-df_005(x_values)), mode='lines', name='Error h=0.05', line=dict(color='royalblue')))
fig.update_layout(title=f"Errores absolutos de la derivada de {nombre}", xaxis_title="x", yaxis_title="Error", template="plotly_white")
fig.show()
ddf_01 = nd.Derivative(f, step=0.1, method='central', order=2, n=2)
ddf_005 = nd.Derivative(f, step=0.05, method='central', order=2, n=2)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=ddf_01(x_values), mode='lines', name='h=0.1', line=dict(color='teal')))
fig.add_trace(go.Scatter(x=x_values, y=ddf_005(x_values), mode='lines', name='h=0.05', line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=x_values, y=dderf(x_values), mode='lines', name='2da derivada exacta', line=dict(color='goldenrod')))
fig.update_layout(title=f"Aproximación de la 2da derivada de {nombre}", xaxis_title="x", yaxis_title="f''(x)", template="plotly_white")
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=np.abs(dderf(x_values)-ddf_01(x_values)), mode='lines', name='Error h=0.1', line=dict(color='teal')))
fig.add_trace(go.Scatter(x=x_values, y=np.abs(dderf(x_values)-ddf_005(x_values)), mode='lines', name='Error h=0.05', line=dict(color='royalblue')))
fig.update_layout(title=f"Errores absolutos 2da derivada de {nombre}", xaxis_title="x", yaxis_title="Error", template="plotly_white")
fig.show()

# Llama la función así, sin espacios antes:
analizar_funcion_c(f, derf, dderf, x_vals, "c) $\\sqrt{x}\\sin(x^2)$")
