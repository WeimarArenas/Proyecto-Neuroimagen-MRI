# Guía para convertir datos DICOM a BIDS usando BIDScoin

Esta guía proporciona los pasos detallados para convertir datos DICOM al formato BIDS (Brain Imaging Data Structure) utilizando BIDScoin.

## Paso 1: Organizar los archivos DICOM en la carpeta principal

### Crear la carpeta principal:
Cree una carpeta que contendrá todos los datos y subcarpetas. Puede nombrar esta carpeta principal, por ejemplo, `Proyecto_BIDS`.

### Crear la subcarpeta raw:
Dentro de la carpeta principal `Proyecto_BIDS`, cree una subcarpeta llamada `raw`. Coloque todos los archivos DICOM sin modificar en esta subcarpeta. Asegúrese de que esta carpeta contenga las imágenes DICOM tal como las recibió de los dispositivos de adquisición, sin alteraciones.

## Paso 2: Organizar los archivos DICOM con DICOMSorter

### Instalar DICOMSorter:
Para instalar DICOMSorter, ejecute el siguiente comando:

```bash
pip install dicomsorter==2.3.1
```

### Ejecutar DICOMSorter:
Una vez instalado, ejecute DICOMSorter utilizando el siguiente comando:

```bash
dicomsorter <ruta_original> <ruta_salida>
```

Asegúrese de reemplazar `<ruta_original>` con la ubicación de la carpeta `raw` y `<ruta_salida>` con la carpeta en la que desee organizar los archivos.

## Paso 3: Crear la carpeta BIDS

### Crear la carpeta vacía BIDS:
Regrese a la carpeta principal `Proyecto_BIDS` y cree una subcarpeta vacía llamada `BIDS`. Esta carpeta será la que contendrá los archivos convertidos al formato BIDS.

## Paso 4: Configurar BIDScoin y dcm2niix

### Instalar BIDScoin y dcm2niix:
Para instalar BIDScoin, ejecute:

```bash
pip install bidscoin
```

Para instalar dcm2niix, asegúrese de seguir las instrucciones disponibles en la publicación anterior.

```bash
pip install dcm2niix
```


### Verificar las instalaciones:
Asegúrese de que ambas herramientas estén correctamente instaladas ejecutando los siguientes comandos en su terminal:

```bash
bidscoin --version
dcm2niix --version
```

Estos comandos deben devolver las versiones instaladas de BIDScoin y dcm2niix, respectivamente.

## Paso 5: Ejecutar el comando bidsmapper

### Ejecutar bidsmapper:
Una vez que las herramientas estén instaladas, ejecute el comando `bidsmapper` desde la terminal, dentro de la carpeta principal de su proyecto. El comando analizará los archivos DICOM y los mapeará hacia la estructura BIDS.

En la terminal, ejecute lo siguiente:

```bash
bidsmapper raw BIDS
```

Donde `raw` es la carpeta de los archivos DICOM y `BIDS` es la carpeta donde se guardarán los datos en formato final. Esto iniciará el proceso de conversión y mapeo de los datos DICOM hacia el formato BIDS.

## Paso 6: Configuración de la GUI de BIDScoin

### Abrir la interfaz gráfica de usuario (GUI):
El comando anterior abrirá la GUI de BIDScoin.

### Configurar la GUI:
En la GUI, seleccione el tipo de datos y las configuraciones necesarias para que BIDScoin realice la conversión de manera correcta. Revise y corrija cualquier error o ajuste en la organización de los datos, si es necesario, antes de proceder.

## Paso 7: Guardar la configuración y ejecutar BIDScoiner

### Guardar el archivo de configuración:
Después de configurar la GUI, guarde el archivo de configuración. Este archivo contendrá la información sobre cómo se deben organizar los datos y se guardará con la extensión `.YAML`.

### Ejecutar BIDScoiner:
Ahora puede ejecutar BIDScoiner utilizando el archivo de configuración que guardó. En la terminal, ejecute el siguiente comando para iniciar la conversión:

```bash
bidscoiner -p <sujeto> -b <ruta_del_config> raw BIDS
```

Reemplace `<sujeto>` con el identificador correspondiente al sujeto y `<ruta_del_config>` con la ruta del archivo de configuración que guardó en el paso anterior.

## Paso 8: Verificar los resultados

### Verificar la carpeta BIDS:
Una vez completado el proceso de conversión, verifique la carpeta `BIDS` para asegurarse de que los archivos se hayan convertido correctamente. La estructura de carpetas debe seguir el formato estándar de BIDS, con subcarpetas como `sub-SUJETO`, y los archivos correspondientes organizados de acuerdo con las especificaciones de BIDS.
