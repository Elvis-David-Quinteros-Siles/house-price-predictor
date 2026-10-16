FROM python:3.11-slim

# Evitar bytecode y salida buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
# Copiar requirements primero (aprovecha cache de Docker)
COPY requirements.txt .

# Instalar dependencias sin cache
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación y su tema
COPY app.py .
COPY .streamlit/ .streamlit/

RUN useradd --create-home --uid 1000 app
USER app

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
