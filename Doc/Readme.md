# Documentación
Carpéta dedicada a la documentación del proyecto y a definiciones teóricas.

1. [Instalación de dependencias](conda-root.md)

## Documentación para la práctica ROOT/PyROOT para el análisis $B^{\pm}\to J/\psi(\to\mu^{+}\mu^{-}) K^{\pm}$
1. **Importar ROOT**
    ```python
    import ROOT                     # Compando para cargar la librería PyROOT dentro de Python o Jupyter.
    ROOT.gROOT.SetBatch(True)       # Evita que ROOT intente abrir ventanas gráficas interactivas.
    ```

2. **Definición de los archivos de ROOT**
    ```python
    # Sintaxis remota
    root_files = [
        "root://eospublic.cern.ch/<ruta del archivo>.root"
    ]
    # Sintaxis local
    root_files = [
        "<carpeta>/<archivo>.root"
    ]
    ```
    * La sintaxis remota permite utilizar archivos que estén en la página del CERN. ROOT puede leer los archivos directamente usando el protocolo `root://`, siempre que la instalación tenga soporte XRootD.
    * La sintaxis local se utiliza cuando se descarga el archivo de manera local, es útil en caso de que este lenta la conexión, para análisis pesados o para pruebas rápidas.

3. **Creación del DataFrame**
    ```python
    rdf = ROOT.RDataFrame("nombre_del_tree", lista_de_archivos)
    ```

4. **Busqueda de columnas específicas**

    Se buscan los nombres de todas las columnas/branches del arbol con el comando
    ```python
    columns = sorted([str(c) for c in rdf.GetColumnNames()])
    for col in columns:
        print(col)
    ```
    * Para variables de masa
        ```python
        for col in columns:
            if "_M" in col:
                print(col)
        ```
    * Para momento transversal
        ```python
        for col in columns:
            if "_PT" in col:
                print(col)
        ```
    * Para identificación de partículas
        ```python
        for col in columns:
            if "PID" in col or "ProbNN" in col or "DLL" in col:
                print(col)
        ```
    * Para variables de calidad de vértice
        ```python
        for col in columns:
            if "CHI2" in col or "VERTEX" in col or "IP" in col:
                print(col)
        ```
    Esto permite explorar las variables que tiene el ntuple antes de decidir el análisis.

5. **Contar candidatos/eventos**

    Para determinar la cantidad de entradas del dataframe
    ```python
    n = rdf.Count().GetValue()
    ```
    * `Count()` devuelve un resultado diferido.
    * `GetValue()` fuerza la ejecución del análisis.

    ROOT suele usar una ejecución diferida en `RDataFrame`, lo que implica que muchas operaciones no se ejecutan de manera inmediata, sino cuando se accede al resultado, como con los comandos `Draw()` o `GetValue()`.

6. **Generación de histograma 1D**

    Sintaxis general para un histograma en 1D
    ```python
    hist = rdf.Histo(
        (
            "nombre",
            "titulo; eje_x; eje_y",
            bins,
            xmin,
            xmax
        ),
        "columna"
    )
    ```
    Parámetros:
    * `"nombre"`:     Nombre interno del histograma
    * `"titulo"`:     Título visible de la gráfica
    * `eje_x`:        Etiqueta del eje X
    * `eje_y`:        Etiqueta del eje Y
    * `bins`:         Número de divisiones
    * `xmin`:         Valor mínimo del eje X
    * `xmax`:         Valor máximo del eje X
    * `"columna"`:    Variable a graficar

    Relleno del histograma:
    ```python
    hist.SetFillColor(ROOT.<color>)
    ```

7. **Creación del canvas**
    
    Área donde ROOT dibuja los histogramas:
    ```python
    # Sintaxis
    canvas = ROOT.TCanvas("nombre", "titulo", ancho, alto)
    ```
    Parámetros
    * `"nombre"`:       Nombre interno de la gráfica
    * `"titulo"`:       Título visible de la gráfica
    * `ancho`:          Ancho en pixeles
    * `alto`:           Alto de pixeles
    
    Un `TCanvas` es el área gráfica donde ROOT dibuja objetos; puede contener uno o varios pads, y unaa sesión de ROOT puede tener varios canvases abiertos.

8. **Dibujar histograma**
    ```python
    hist.Draw()
    ```
    Opciones viables dentro del paréntesis `Draw()`:
    * `"HIST"`: Dibuja como histograma de líneas.
    * `"E"`: Dibuja con barras de error.
    * `"SAME"`: Dibuja encima de lo que ya existe en el canvas.
    * `"COLZ"`: Para histogramas 2D, dibuja colores con barra Z.

9. **Guardar gráfica**
    ```python
    canvas.SaveAs("nombre_archivo.extension")
    ```
    Formatos de extensión posibles: `.png`, `.pdf`, `.svg`, `.root`.

    Par quitar la caja de estadística:
    ```python
    ROOT.gStyle.SetOptStat(0)   # Oculta la caja que muestra entries, mean, std dev, etc.
    ```

10. **Formato de curvas**
    * Color de curvas
        ```python
        hist.SetLineColor(ROOT.<color>)
        ```
        Colores comunes para <color>: `kBlack`, `kRed`, `kBlue`, `kGreen`, `kMagenta`, `kCyan`, `kOrange`, `kViolet`, `kGray`.
        Variaciones para el color: `SetLineColor(ROOT.<color> +/- n)`, donde `n` es un número entero.
    * Grosor de línea
        ```python
        hist.SetLineWidth(n)
        ```
        El parámetro `n`:
        * Para `n=1` se tiene una línea sólida.
        * Para `n=2` se tiene una línea punteada/guión.
        * Para `n=3` la línea son puntos.

        Los histogramas heredan atributos gráficos como líneas, relleno y marcadores. ROOT documenta que las clases que heredan de `TAttLine` pueden cambiar color, grosor y estilo con métodos como `SetLineColor()`, `SetLineWidth()` y `SetLineStyle()`.

11. **Leyenda de la gráfica con `TLegend`**
    ```python
    # Sintaxis
    legend = ROOT.TLegend(x1, y1, x2, y2)
    ```
    Parámetros:
    * `x1`: borde izquierdo
    * `y1`: borde inferior
    * `x2`: borde derecho
    * `y2`: borde superior

    las coordenadas van normalmente de `0` a `1`.

    Para añadir elementos a la leyenda
    ```python
    legend.AddEntry(objeto, "texto", "opcion")
    ```
    Parámetros:
    * `objeto`: Histograma, función o gráfico representado
    * `"texto"`: Texto que aparece en la leyenda
    * `"opcion"`: Forma de representación visual

    Para el parámetro de `"opcion"` suele usarse: `"l"` para líneas, `"p"` para un marcador/punto, `"f"` para relleno y `"e"` para la barra de error.

    ROOT documenta que cada entrada de una TLegend se agrega con `AddEntry()`, indicando el objeto, la etiqueta y una opción como línea, marcador, relleno o barra de error.

    Para el tamaño de la fuente:
    ```python
    legend.SetTextSize()
    ```
    los valores recomendados son: `0.025` $\to$ pequeño, `0.035` $\to$ normal, `0.045` $\to$ grande, `0.060` $\to$ muy grande. Así mismo, se puede cambiar la fuente
    ```python
    legend.SetTextFont()
    ```
    Se puede quitar el borde con el comando
    ```python
    legend.SetBorderSize(0)
    ```
    Para un fondo transparente
    ```python
    legend.SetFillStyle(0)
    ```

12. **Aplicación de Filtros**

    Para seleccionar únicamente los candidatos que cumplen la condición
    ```python
    # Sintaxis
    rdf_filtrado = rdf.Filter("condicion", "nombre_del_corte")
    ```
    Operadores comunes para la condición: `<`, `>`, `<=`, `>=`, `==`, `!=`, `&&`, `||`.

13. **Reporte de recortes**
    ```python
    report = rdf_filtrado.Report()
    report.Print()
    ```
    Muestra los candidatos que pasan cada corte.

14. **Histograma 2D**
    ```python
    hist_2d = rdf.Histo2D(
        (
            "nombre",
            "titulo; eje_x; eje_y",
            bins_x, xmin, xmax,
            bins_y, ymin, ymax,
        ),
        "columna_x",
        "columna_y"
    )
    ```
    Para usar una escala logarítmica en histogramas 2D
    ```python
    # Sintaxis
    canvas.SetLogz()
    ```
    Es util cuando hay zonas con muchos candidatos y otras con pocos candidatos.

15. **Ajuste de curva a un histograma**
    ```python
    hist.Fit("func", "S", "", xmin, xmax)
    ```
    Parámetros:
    * `"func"`: función para ajustar al histograma:
        * `gaus`: Función gausiana.
        * `expo`: Función exponencial.
        * `landau`: Función de Landau.
        * `polN`: función personalizada con `TF1`, donde `pol0` para una función constante, `pol1` para una función lineal, `pol2` para una función cuadrática, `pol3` para una función cúbica, etc.
    * `"S"`: Devuelve objeto con resultado del fit. Otras opciones son
        * `"R"`: Usa el rango definido en la `TF1`.
        * `"+"`: Conserva fits anteriores en el histograma.
        * `"L"`: Usa likelihood, util para conteos Poisson o baja estadística. Es recomendable usar este método cuado el histograma representa conteos con estadística de Poisson, especialmente si hay baja estadística.
    * `""`: Opción gráfica adicional vacía.
    * `xmin`: Inicio del rango de ajuste.
    * `xmax`: Fin del rango de ajuste

    ROOT permite ajustar histogramas 1D, 2D, 3D y perfiles usando funciones definidas por el usuario o funciones predefinidas mediante `TH1::Fit`

    Para un ajuste personalizado con `TF1`
    ```python
    # Sintaxis para definir un ajuste personalizado
    func = ROOT.TF1(
        "nombre_funcion",
        "formula",
        xmin,
        xmax
    )

    # Configuración de parámetros
    func.SetParameters(
        "parametros_funcion"
    )

    # Aplicación al histograma/gráfica
    hist.Draw()
    hist.Fit(func, "S", "", xmin, xmax)
    ```
    ROOT permite crear funciones propias con `TF1` y luego pasarlas al método `Fit()`. Para funciones no predefinidas, los parámetros iniciales deben configurarse antes del ajuste.

<[Regresar al inicio](https://github.com/MASilva139/ECFM-PF)>