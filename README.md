<!-- 
Comando de instalación de ROOT local con conda
conda --version # para verificar si se tiene instalada conda

conda config --set channel_priority strict
conda create -n lhcb-root -c conda-forge root xrootd python=3.11
conda activate lhcb-root

# verificación de que pyroot funciona
python -c "import ROOT; print (ROOT.gROOT.GetVersion())"

# En caso de que no corra inicialmente el entorno conda
source ~/.bashrc
conda activate lhcb-root

 -->
 # ...
Consideraciones: el siguiente procedimiento se está realizando en el entorno de Ubuntu 24.04, de manera que puede no funcionar en alguna otra distribución de Linux.

 ## Instalación de Conda
 1. Instalación de dependencias básicas
    ```bash
    sudo apt update
    sudo apt install -y wget curl bzip2
    ```
2. Descargar Anaconda
    ```bash
    wget https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh
    ```
3. Verificación del archivo descargado 
    ```bash
    sha256sum Anaconda3-2025.12-2-Linux-x86_64.sh
    ```
4. Ejecución del instalador
    ```bash
    bash Anaconda3-2025.12-2-Linux-x86_64.sh
    ```
    durante la instalación
    ```
    * Presionar `Enter` para continuar con la instalación.
    * Escribir `yes` para aceptar la licencia.
    * Presionar `Enter` para instalar en /home/usuario/anaconda3.
    * En la opción de ejecutar conda init escribir `yes`.
    * Recargar laterminal con `source ~/.bashrc` o cerrando y abriendo la terminal.
    ```
5. Verificación de la instalación de conda
    ```bash
    conda --version
    ```
6. Configuración del entorno de conda
    ```bash
    conda config --set channel_priority strict
    conda create -n lhcb-root -c conda-forge root xrootd python=3.11
    ```
7. Activar entorno de conda (recomendación, que sea en el entorno de trabajo vscode)
    ```bash
    conda activate lhcb-root
    ```
8. Caso de que aparezca `conda: command nod found`
    ```bash
    source ~anaconda3/bin/activate
    conda init bash
    source ~/.bashrc
    ```
9. En caso de que aparezca 
    ```
    no charge usr/anaconda3/condabin/conda
    no charge usr/anaconda3/bin/conda
    no charge usr/anaconda3/bin/conda-env
    no charge usr/anaconda3/bin/activate
    no charge usr/anaconda3/bin/deactivate
    no charge usr/anaconda3/etc/profile.d/conda.sh
    no charge usr/anaconda3/etc/fish/conf.d/conda.fish
    no charge usr/anaconda3/shell/condabin/Cpnda.psm1
    no charge usr/anaconda3/shell/condabin/conda-hook.ps1
    no charge usr/anaconda3/lib/python3.13/site-packages/xontrib/conda.xsh
    no charge usr/anaconda3/etc/profile.d/conda.csh
    no charge usr/bashrc
    No action taken
    ```
    se tiene que conda está inicializado, pero la terminal no ha cargado la configuración. Introducir
    ```bash
    source ~/.bashrc
    conda activate lhcb-root
    ```

 ## Instalación de ROOT-System