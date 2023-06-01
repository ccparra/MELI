# MELI
## Requisitos del sistema

- Python versión 3.9.16
- PIP versión 23.0.1

## Instalación

1. Clona el repositorio en tu máquina local:

   ```shell
   git clone <https://github.com/ccparra/MELI.git>

2. Navega al directorio del proyecto:
   
    ```shell
     cd MELI

3. Crea un entorno virtual (opcional pero se recomienda):

     ```shell
    python -m venv --python=3.9.16 myenv

4. Activa el entorno virtual (si se creó uno):

    En Windows:
     ```shell
    .\myenv\Scripts\activate
     ```

    En Linux/Mac:
     ```shell
    source ./myenv/bin/activate
    ```

5. Instala las dependencias del proyecto utilizando el archivo requirements.txt:

     ```shell
     python -m pip install --upgrade pip==23.0.1
     pip install -r requirements.txt